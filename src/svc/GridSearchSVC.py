import jax
import jax.numpy as jnp

import pandas as pd
import warnings
import logging
import time

from typing import Optional
from typing import Iterable
from typing import Dict
from typing import Union

from sklearn.model_selection import ParameterGrid

from .LinearSVC import LinearSVC
from .KernelSVC import KernelSVC

from ..features.RandomCDE import RandomCDE
from ..features.RandomRDE import RandomRDE
from ..features.RandomFourierFeatures import RandomFourierFeatures

from ..utils.random import KeyGen
from ..utils.cache import Cache
from ..utils.lie_algebra import get_logsig_dimension
from ..utils.logger import Logger
from ..preprocessing import Preprocessor

from ..utils.hyperparams import suggest_bandwidth

from ..configs import DEFAULT_CDE_GS
from ..configs import DEFAULT_RDE_GS
from ..configs import DEFAULT_SVC_GS
from ..configs import DEFAULT_LIN_SVC_GS
from ..configs import DEFAULT_PRE_GS

EPS_ = 1e-10


class GridSearchSVC:
    """
    Grid search over estimator and feature-extractor hyperparameters, pure Torch CV.

    Parameters
    ----------
    extractor : class
        Feature-extractor class with `get_features(X: Tensor) -> Tensor`.

    """
    def __init__(self,
                 type,
                 param_grid : dict,
                 gpu : bool = False,
                 linear_svc : bool = True,
                 rff_type : str = '1D',
                 seed : int = 42,
                 verbose : Union[bool, Logger] = False,
                 batch_size : int = 100,
                 stratified : bool = True,
                 n_splits : int = 3,
                 shuffle : bool = False,
                 max_dim_logsigs : int = 1500,
                 random_state : Optional[int] = None):
        
        
        assert type.lower() in ['rde', 'cde'], "type must be 'rde' or 'cde'"
        assert rff_type.lower() in ['1d', '2d'], "rff_type must be '1D' or '2D'"

        self.type = type.lower()
        self.rff_type = rff_type.lower()
        self.batch_size = batch_size
        self.key = KeyGen(seed)
        self.linear_svc = linear_svc
        self.max_dim_logsigs = max_dim_logsigs

        if isinstance(verbose, Logger):
            self.verbose = 'logger'
            self.logger = verbose
        else:
            self.verbose = verbose
            self.logger = None

        self._get_param_dicts(param_grid.copy())

        # CV splitter params
        self.stratified = stratified
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        # Device
        self.gpu = gpu and any(d.platform == "gpu" for d in jax.devices())
        if gpu and not self.gpu:
            warnings.warn("CUDA not available; falling back to CPU.")

    # ==============================================================================
    # Parameter grid preprocessing
    # ==============================================================================

    def _get_param_dicts(self, param_grid):

        _default_dict_pre = DEFAULT_PRE_GS 
        _default_dict = DEFAULT_CDE_GS if self.type == 'cde' else DEFAULT_RDE_GS
        _default_dict_svc = DEFAULT_LIN_SVC_GS if self.linear_svc else DEFAULT_SVC_GS
        

        # Ensure all values in param_grid are lists
        for keys, values in param_grid.items():
            if not isinstance(values, list):
                param_grid[keys] = [values]

        self.normalize_feat_list = param_grid.pop('normalize_feat', _default_dict['normalize_feat'])

        # ++++++++++++++++++++++ Differential Equation params ++++++++++++++++++++++

        # Only RDE has an order parameter
        if self.type == 'rde':
            self.orders_list = param_grid.pop('order', _default_dict['order'])
            self.step_list = param_grid.pop('step', _default_dict['step'])
        else:
            self.orders_list = [None]
            self.step_list = [None]

        # CDE/RDE features dimension
        self.n_features_list = param_grid.pop('n_features', _default_dict['n_features'])

        # Random Differential Equations params
        self.extractor_param_grid = {
            'stdA': param_grid.pop('stdA', _default_dict['stdA']),
            'stdB': param_grid.pop('stdB', _default_dict['stdB']),
            'std0': param_grid.pop('std0', _default_dict['std0']),
            'activation': param_grid.pop('activation', _default_dict['activation']) 
            }
        

        # ++++++++++++++++++++++ Random Fourier Features params ++++++++++++++++++++++

        self.n_fourier_features_list = param_grid.pop('n_fourier_features', 
                                                      _default_dict['n_fourier_features'])

        self.bandwidth_list = param_grid.pop('bandwidth', _default_dict['bandwidth'])


        # ++++++++++++++++++++++ SVC params ++++++++++++++++++++++
        
        self.svc_param_grid = {
            'C': param_grid.pop('C', _default_dict_svc['C']),  
        }

        other_svc_possible_params = [
            'tol'
            'max_iter',
            'fit_intercept',
            'dual',
        ]

        for key, val in param_grid.items():
            if key in other_svc_possible_params:
                self.svc_param_grid[key] = val

        if self.linear_svc:
            self.svc_param_grid['penalty'] = param_grid.pop('penalty', _default_dict_svc['penalty'])
        else:
            self.svc_param_grid['gamma'] = param_grid.pop('gamma', _default_dict_svc['gamma'])

        # ++++++++++++++++++++++ Preprocessing params ++++++++++++++++++++++

        # Preprocessing params
        self.pre_param_grid = {
            'add_time': param_grid.pop('add_time', _default_dict_pre['add_time']),
            'lead_lag': param_grid.pop('lead_lag', _default_dict_pre['lead_lag']),
            'basepoint': param_grid.pop('basepoint', _default_dict_pre['basepoint']),
            'normalize': param_grid.pop('normalize', _default_dict_pre['normalize']),
            'max_time': param_grid.pop('max_time', _default_dict_pre['max_time']),
            'max_len': param_grid.pop('max_len', _default_dict_pre['max_len'])
        }

        
    # ==============================================================================
    # Utils methods
    # ==============================================================================  

    def _validate_input(self, X, y):
        """
        Validate the input types for X and y.
        """
        if not isinstance(X, jnp.ndarray):
            raise ValueError("X must be a jnp.ndarray")
        if not isinstance(y, jnp.ndarray):
            raise ValueError("y must be a jnp.ndarray")

        if X.ndim != 3:
            if X.ndim == 2:
                if self.type == 'rde':
                    raise ValueError("X must be a 3D tensor for RDE")
                X = X[..., None]
            else:
                raise ValueError("X must be a 2D or 3D tensor")

        return X, y


    def _get_feature_extractor(self, 
                               n_features, 
                               extractor_params, 
                               order=None, 
                               step=None):
        """
        Get the feature extractor from the parameter dictionaries.
        """

        # Create the feature extractor
        if self.type == 'rde':
            assert order is not None, "order must be specified for RDE"
            assert step is not None, "step must be specified for RDE"
            
            feature_extractor = RandomRDE(self.key(),
                                          n_features=n_features,
                                          order=order,
                                          step=step,
                                          config=extractor_params,
                                          cache=self.cache,
                                          **extractor_params)
            
        elif self.type == 'cde':
            feature_extractor = RandomCDE(self.key(),
                                          n_features=n_features,
                                          config=extractor_params,
                                          cache=self.cache)

        return feature_extractor
    

    def _get_extractor_params_combinations(self) -> Iterable:
        """
        Get all combinations of extractor parameters for the grid search 
        such that the activation are in the inner loop.
        """

        param_grid = dict(self.extractor_param_grid)
        activation_list = param_grid.pop('activation')

        for params in ParameterGrid(param_grid):
            for activation in activation_list:
                params = dict(params)
                params['activation'] = activation

                yield params


    def _verbose_helper(self, msg: str, level: int = logging.INFO):
        if self.verbose is False:
            return
        elif self.verbose is True:
            print(msg)
        elif self.verbose == "logger":
            self.logger.log(msg, level=level) 


    # ==============================================================================
    # Fit methods
    # ============================================================================== 

    def _get_svc(self, svc_params : Dict = {}):

        if self.linear_svc:
            svc = LinearSVC(gpu=self.gpu, **svc_params)
        else:
            # CHANGE HERE
            svc = KernelSVC(gpu=False, **svc_params)

        return svc

    def evaluate_extractor_svc(self, X : jnp.ndarray, y : jnp.ndarray, sig_params : dict):
        """
        Loops over extractor params (outer) and svc params (inner), computing mean train/val scores.
        Returns DataFrame of results and best-params dict based on val accuracy.
        """
    
        records = []

        order = sig_params['order']
        step = sig_params['step']
        n_features = sig_params['n_features']


        for extractor_params in self._get_extractor_params_combinations():

            try:
                
                # start_time = time.time()

                # Get the feature extractor
                extractor = self._get_feature_extractor(n_features, 
                                                        extractor_params,
                                                        order=order,
                                                        step=step)

                # Get the features
                if self.linear_svc:
                    svc_input = extractor.get_features(X, 
                                                       batch_size=self.batch_size,
                                                       return_interval=False,
                                                       use_cache=True)
                else:
                    svc_input = extractor.get_gram(X, 
                                                   batch_size=self.batch_size,
                                                   return_interval=False,
                                                   use_cache=True)
                    
                # end_time = time.time()

                # print(end_time - start_time)

            except Exception as e:
                warnings.warn(f"Failed to get features for extractor_params={extractor_params}: {e}")

                _params = {**sig_params, **extractor_params, **self.svc_param_grid,
                           'normalize_feat': None, 'train_score': None}
                
                records.append(_params)
                continue

            for normalize_feat in self.normalize_feat_list:

                if normalize_feat:
                    if self.linear_svc:   
                        svc_input = svc_input / (jnp.linalg.norm(svc_input, axis=1, keepdims=True) + EPS_)
                    else:
                        _diag = jnp.sqrt(jnp.diag(svc_input) + EPS_)
                        svc_input = svc_input / (_diag[:, None] * _diag[None, :])

                
                # start_time = time.time()

                # Fit grid search
                svc = self._get_svc()
                svc.fit_gridsearch(svc_input, 
                                   y, 
                                   self.svc_param_grid, 
                                   cv=self.n_splits, 
                                   stratified=self.stratified)

                results_ = {**sig_params,
                            **extractor_params,
                            **svc.best_params,
                            'normalize_feat': normalize_feat,
                            'train_score': svc.best_score}

                # end_time = time.time()
                # print('svc', end_time - start_time)

            records.append(results_)

        # Create DataFrame from records
        df = pd.DataFrame(records)

        # Get the best model based on validation score
        if df.empty or all(df.train_score.isna()):
            best_model = {}
        else:
            best_model_idx = df['train_score'].idxmax()
            best_model = df.loc[best_model_idx].to_dict() 
        
        return df, best_model


    def _fit(self,
             X : jnp.ndarray,
             y : jnp.ndarray):
        """
        Loops over n_features and n_features_fourier, calls evaluate_extractor_svc,
        concatenates DataFrames and collects best-lines.
        Returns (all_results_df, best_params_dict).
        """

        all_dfs = []
        best_models = []

        self.cache = Cache()

        for n_fourier_feat in self.n_fourier_features_list:

            self._verbose_helper(f"N_fourier_features = {n_fourier_feat}")
            
            # If n_fourier_feat is None, we skip the RFF part
            bandwidth_list = self.bandwidth_list if n_fourier_feat is not None else [None]
            for bandwidth in bandwidth_list:

                if n_fourier_feat is not None: 

                    # Adjust for number of fourier features
                    _n_fourier_feat = n_fourier_feat // 2 if self.rff_type == '2d' else n_fourier_feat
                 
                    rff_cls = RandomFourierFeatures(self.key(),
                                                    method=self.rff_type,
                                                    n_features=_n_fourier_feat,
                                                    bandwidth=bandwidth,
                                                    cache=self.cache)


                    X_rff = rff_cls.get_features(X, use_cache=True)

                else:
                    X_rff = X
                    n_fourier_feat = 'None'
                    bandwidth = 'None'

                X_rff = X_rff / X_rff.max()

                for order in self.orders_list:

                    if self.type == 'rde':
                        _dim_logsigs = get_logsig_dimension(order, X_rff.shape[-1])
                        
                        if _dim_logsigs > self.max_dim_logsigs:
                            continue

                    for step in self.step_list:
                        for n_feat in self.n_features_list:

                            self._verbose_helper(f"  Order = {order}, Bandwidth = {bandwidth}, N_features = {n_feat}")

                            sig_params = {'n_fourier_features': n_fourier_feat,
                                          'bandwidth': bandwidth,
                                          'n_features': n_feat,
                                          'order': order,
                                          'step': step}

                            df, best = self.evaluate_extractor_svc(X_rff, y, sig_params)

                            all_dfs.append(df)
                            best_models.append(best)

        df_all_results = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        df_best_models = pd.DataFrame(best_models) if best_models else pd.DataFrame()
        df_best_models.dropna(axis=0, subset='train_score', inplace=True)

        return df_all_results, df_best_models


    def _test(self, 
              X : jnp.ndarray,
              y : jnp.ndarray,
              X_test : jnp.ndarray,
              y_test : jnp.ndarray,
              df_best_models : pd.DataFrame
              ):
        
        self.cache = Cache()

        test_scores = []

        for _, row in df_best_models.iterrows():
            
            # Extract parameters from the row
            n_features = row['n_features']
            n_fourier_feat = row['n_fourier_features']
            bandwidth = row['bandwidth']
            order = row['order']
            step = row['step']
            normalize_feat = row['normalize_feat']

            preprocess_params = {}
            for key in self.pre_param_grid.keys():
                preprocess_params[key] = row[key]

            extractor_params = {}
            for key in self.extractor_param_grid.keys():
                extractor_params[key] = row[key]
        
            svc_params = {}
            for key in self.svc_param_grid.keys():
                svc_params[key] = row[key] 


            # Apply the preprocessing_class
            preprocessing_class = Preprocessor(**preprocess_params)
            X_transformed = preprocessing_class.fit_transform(X)
            X_test_transformed = preprocessing_class.transform(X_test)

            # Apply the random Fourier features if needed
            if n_fourier_feat != 'None':

                # Adjust for number of fourier features
                _n_fourier_feat = n_fourier_feat // 2 if self.rff_type == '2d' else n_fourier_feat

                rff_cls = RandomFourierFeatures(self.key(),
                                                method=self.rff_type,
                                                n_features=_n_fourier_feat,
                                                bandwidth=bandwidth,
                                                cache=self.cache)
                
                X_rff = rff_cls.get_features(X_transformed, use_cache=True)
                X_rff_test = rff_cls.get_features(X_test_transformed, use_cache=True)

            else:
                X_rff = X_transformed
                X_rff_test = X_test_transformed

            X_rff = X_rff / X_rff.max()
            X_rff_test = X_rff_test / X_rff.max()

            # Apply the feature extractor
            extractor = self._get_feature_extractor(n_features, 
                                                    extractor_params,
                                                    order=order,
                                                    step=step)

            X_feat_train = extractor.get_features(X_rff,
                                                  batch_size=self.batch_size,
                                                  return_interval=False,
                                                  use_cache=True)
            X_feat_test = extractor.get_features(X_rff_test,
                                                 batch_size=self.batch_size,
                                                 return_interval=False,
                                                 use_cache=True)

            if normalize_feat:
                X_feat_train = X_feat_train / (jnp.linalg.norm(X_feat_train, axis=1, keepdims=True) + EPS_)
                X_feat_test = X_feat_test / (jnp.linalg.norm(X_feat_test, axis=1, keepdims=True) + EPS_)

            if not self.linear_svc:
                svc_input_train = X_feat_train @ X_feat_train.T
                svc_input_test = X_feat_test @ X_feat_train.T
            else:
                svc_input_train = X_feat_train
                svc_input_test = X_feat_test

            # Fit svc
            svc = self._get_svc(svc_params)
            svc.fit(svc_input_train, y)
            test_score = svc.score(svc_input_test, y_test)
            test_scores.append(test_score)

        df_best_models['test_score'] = test_scores

        return df_best_models
    

    def fit(self, 
            X : jnp.ndarray,
            y : jnp.ndarray,
            X_test : Optional[jnp.ndarray] = None,
            y_test : Optional[jnp.ndarray] = None,
            name : str = 'results',
            save : bool = True):
        """
        Fit the GridsearchSVC model.

        Parameters
        ----------
        X : Tensor
            Training data.
        y : Tensor
            Training labels.
        X_test : Optional[Tensor]
            Test data.
        y_test : Optional[Tensor]
            Test labels.
        name : str
            Name for saving results.
        save : bool
            Whether to save results to CSV files.
        """
        # Validate input
        X, y = self._validate_input(X, y)
        
        if X_test is None or y_test is None:
            testing = False
        else:
            testing = True
            X_test, y_test = self._validate_input(X_test, y_test)

        # Adjust number of splits for imbalanced classes
        self.n_splits = min(self.n_splits, jnp.min(jnp.bincount(y)).item())

        all_dfs = []
        best_models = []

        self._verbose_helper("Starting grid search...")
        _pre_paramgrid_size = len(ParameterGrid(self.pre_param_grid))

        # Grid search over preprocessing parameters
        for i, pre_params in enumerate(ParameterGrid(self.pre_param_grid)):

            self._verbose_helper(f"Starting Preprocessing combo {i+1} out of {_pre_paramgrid_size}")

            # Preprocessing
            preprocessing_class = Preprocessor(**pre_params)
            X_transformed = preprocessing_class.fit_transform(X)

            # Update bandwidth - TO BE CHANGED
            suggested_bandwidth = suggest_bandwidth(X_transformed)
            _bandwidth_list = [suggested_bandwidth * br for br in self.bandwidth_list]
            self.bandwidth_list = _bandwidth_list + [1.0, 1.25, 0.75]

            # Fit model
            df_all_results, df_best_models = self._fit(X_transformed, y)

            # Store results
            all_dfs.append(df_all_results.assign(**pre_params))
            best_models.append(df_best_models.assign(**pre_params))

        df_all_results = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        df_best_models = pd.concat(best_models, ignore_index=True) if best_models else pd.DataFrame()

        # We should filter here for ALL the pre_params
        params_to_filter = [
            'n_features',
            'n_fourier_features',
            'order',
            'step',
            'lead_lag'
        ]

        idx = df_best_models.groupby(params_to_filter, dropna=False)['train_score'].idxmax()
        df_best_models = df_best_models.loc[idx.values].reset_index(drop=True)

        if testing:
            df_best_models = self._test(X, y, X_test, y_test, df_best_models)

        _mask = (df_best_models.nunique(dropna=False) > 1)
        _mask.loc[['train_score', 'test_score', 'activation']] = True
        df_best_models = df_best_models.loc[:, _mask]
        df_best_models.reset_index(drop=True, inplace=True)

        if save:
            df_all_results.to_csv(name + '_results.csv', index=False)
            df_best_models.to_csv(name + '_best_results.csv', index=False)

        else:
            return df_all_results, df_best_models

