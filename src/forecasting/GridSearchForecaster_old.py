import jax
import jax.numpy as jnp

import pandas as pd
import numpy as np
import warnings
import logging
import time

from typing import Optional
from typing import Iterable
from typing import Dict
from typing import Union

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from .LinearSVC import LinearSVC
from .KernelSVC import KernelSVC

from ..features.RandomCDE import RandomCDE
from ..features.RandomRDE import RandomRDE
from ..features.RandomFourierFeatures import RandomFourierFeatures

from ..utils.random import KeyGen
from ..utils.cache import Cache
from ..utils.lie_algebra import get_logsig_dimension
from ..utils.logger import Logger
from ..utils.preprocessing import Preprocessor

from ..utils.hyperparams import suggest_bandwidth

from ..configs import DEFAULT_CDE_GS
from ..configs import DEFAULT_RDE_GS
from ..configs import DEFAULT_SVC_GS
from ..configs import DEFAULT_LIN_SVC_GS
from ..configs import DEFAULT_PRE_GS

EPS_ = 1e-10

# print times for debug


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
                 max_dim_logsigs : int = 1000,
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

        self.bandwidth_ratios = param_grid.pop('bandwidth', _default_dict['bandwidth'])


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
                features = extractor.get_features(X, 
                                                    batch_size=self.batch_size,
                                                    return_interval=False,
                                                    use_cache=True)
                    
                # end_time = time.time()
                # print(end_time - start_time)

            except Exception as e:
                
                # warnings.warn(f"Failed to get features for extractor_params={extractor_params}: {e}")

                _dict = {_key : None for _key in self.svc_param_grid.keys()}
                _params = {**sig_params, **extractor_params}

                self._verbose_helper(f"Failed to get features for params={_params}", level=logging.WARNING)

                # To be removed
                print(f"Failed to get features for params={_params}")


                results_ = {**sig_params, **extractor_params, **_dict,
                            'normalize_feat': False,
                            'cv_score': -2.0}
                    
                records.append(results_)

                continue

            for normalize_feat in self.normalize_feat_list:
                    
                try:
                    
                    if normalize_feat:        
                        features = features / (jnp.linalg.norm(features, axis=1, keepdims=True) + EPS_)
                    
                    # start_time = time.time()

                    if not self.linear_svc:
                        svc_input = features @ features.T
                    else:
                        svc_input = features 

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
                                'cv_score': svc.best_score}
                    
                    records.append(results_)

                    # end_time = time.time()
                    # print('svc', end_time - start_time)

                except Exception as e:

                    _dict = {_key : None for _key in self.svc_param_grid.keys()}
                    _params = {**sig_params, **extractor_params}

                    self._verbose_helper(f"Failed to fit SVC for params={_params}", level=logging.DEBUG)

                    results_ = {**sig_params, **extractor_params, **_dict,
                                'normalize_feat': normalize_feat,
                                'cv_score': -1.0}
                    
                    records.append(results_)

                    print(e)

                    # To be removed
                    # print(f"Failed to fit SVC for params={_params}")
                    continue

        # Create DataFrame from records
        df = pd.DataFrame(records)

        # Get the best model based on validation score
        if df.empty or all(df.cv_score.isna()):
            best_model = {}
        else:
            best_model_idx = df['cv_score'].idxmax()
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

                            if self.type == 'cde':
                                self._verbose_helper(f"  Bandwidth = {bandwidth}")
                            else:
                                self._verbose_helper(f"  Order = {order}, Step = {step}, Bandwidth = {bandwidth}")

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
        df_best_models.dropna(axis=0, subset='cv_score', inplace=True)

        return df_all_results, df_best_models


    def _test(self, 
              X : jnp.ndarray,
              y : jnp.ndarray,
              X_test : jnp.ndarray,
              y_test : jnp.ndarray,
              df_best_models : pd.DataFrame
              ):
        
        self.cache = Cache()

        train_scores = []
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

            # Normalize the random Fourier features
            _max = X_rff.max()
            X_rff = X_rff / _max
            X_rff_test = X_rff_test / _max

            # Apply the feature extractor
            extractor = self._get_feature_extractor(n_features, 
                                                    extractor_params,
                                                    order=order,
                                                    step=step)

            features_train = extractor.get_features(X_rff,
                                                  batch_size=self.batch_size,
                                                  return_interval=False,
                                                  use_cache=True)
            
            if self.type == 'cde':
                features_test = extractor.get_features(X_rff_test,
                                                       batch_size=self.batch_size,
                                                       return_interval=False,
                                                       use_cache=True)
            else:
                features_test = extractor.get_features(X_rff_test,
                                                       batch_size=self.batch_size,
                                                       return_interval=False,
                                                       use_cache=True,
                                                       testing=True)

            if normalize_feat:
                features_train = features_train / (jnp.linalg.norm(features_train, axis=1, keepdims=True) + EPS_)
                features_test = features_test / (jnp.linalg.norm(features_test, axis=1, keepdims=True) + EPS_)


            if not self.linear_svc:
                svc_input_train = features_train @ features_train.T
                svc_input_test = features_test @ features_train.T
            else:
                svc_input_train = features_train
                svc_input_test = features_test

            # Fit svc
            svc = self._get_svc(svc_params)
            svc.fit(svc_input_train, y)
            train_score = svc.score(svc_input_train, y)
            test_score = svc.score(svc_input_test, y_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

        df_best_models['train_score'] = train_scores
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

        # Update bandwidth - TO BE CHANGED
        suggested_bandwidth = suggest_bandwidth(X)
        _bandwidth_list = [suggested_bandwidth * br for br in self.bandwidth_ratios]
        self.bandwidth_list = _bandwidth_list + [1.0, 1.25, 0.75]

        # Grid search over preprocessing parameters
        for i, pre_params in enumerate(ParameterGrid(self.pre_param_grid)):

            self._verbose_helper(f"Starting Preprocessing combo {i+1} out of {_pre_paramgrid_size}")

            # Preprocessing
            preprocessing_class = Preprocessor(**pre_params)
            X_transformed = preprocessing_class.fit_transform(X)

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

        idx = df_best_models.groupby(params_to_filter, dropna=False)['cv_score'].idxmax()
        df_best_models = df_best_models.loc[idx.values].reset_index(drop=True)

        if testing:
            df_best_models = self._test(X, y, X_test, y_test, df_best_models)

        _mask = (df_best_models.nunique(dropna=False) > 1)
        _mask.loc[['train_score', 'cv_score', 'test_score', 'activation', 'stdA', 'stdB', 'std0']] = True
        df_best_models = df_best_models.loc[:, _mask]
        df_best_models.reset_index(drop=True, inplace=True)

        if save:
            df_all_results.to_csv(name + '_results.csv', index=False)
            df_best_models.to_csv(name + '_best_results.csv', index=False)

        else:
            return df_all_results, df_best_models


# ==============================================================================
# GridSearchForecaster: Forecasting grid search with Ridge regression head
# ==============================================================================

class GridSearchForecaster:
    """
    Grid search over extractor and linear regression forecaster hyperparameters.

    This is the forecasting analogue of GridSearchSVC:
    - Uses the same RandomCDE / RandomRDE feature extractors
    - Uses the same preprocessing and Random Fourier Features stacks
    - Replaces the SVC head with a (potentially multi-output) Ridge regression head

    It assumes that the user has already built input/output pairs suitable for
    forecasting (e.g. sliding windows on ETTh, ILI, UCR-TSF, etc.).

    Parameters
    ----------
    type : str
        Either 'rde' or 'cde', controlling which feature extractor is used.
    param_grid : dict
        Grid of hyperparameters. Keys can include:
        - 'normalize_feat' : list[bool]
        - 'order', 'step' (for RDE)
        - 'n_features', 'stdA', 'stdB', 'std0', 'activation'
        - 'n_fourier_features', 'bandwidth'
        - 'alpha' (for Ridge head)
        plus any of the preprocessing keys in DEFAULT_PRE_GS.
    """

    def __init__(self,
                 type: str,
                 param_grid: dict,
                 rff_type: str = "1D",
                 seed: int = 42,
                 verbose: Union[bool, Logger] = False,
                 batch_size: int = 100,
                 n_splits: int = 3,
                 shuffle: bool = False,
                 max_dim_logsigs: int = 1000,
                 random_state: Optional[int] = None):

        assert type.lower() in ["rde", "cde"], "type must be 'rde' or 'cde'"
        assert rff_type.lower() in ["1d", "2d"], "rff_type must be '1D' or '2D'"

        self.type = type.lower()
        self.rff_type = rff_type.lower()
        self.batch_size = batch_size
        self.key = KeyGen(seed)
        self.max_dim_logsigs = max_dim_logsigs

        if isinstance(verbose, Logger):
            self.verbose = "logger"
            self.logger = verbose
        else:
            self.verbose = verbose
            self.logger = None

        self._get_param_dicts(param_grid.copy())

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    # ==============================================================================
    # Parameter grid preprocessing
    # ==============================================================================

    def _get_param_dicts(self, param_grid):

        _default_dict_pre = DEFAULT_PRE_GS
        _default_dict = DEFAULT_CDE_GS if self.type == "cde" else DEFAULT_RDE_GS

        # Ensure all values in param_grid are lists
        for key, values in param_grid.items():
            if not isinstance(values, list):
                param_grid[key] = [values]

        self.normalize_feat_list = param_grid.pop("normalize_feat", _default_dict["normalize_feat"])

        # ++++++++++++++++++++++ Differential Equation params ++++++++++++++++++++++

        if self.type == "rde":
            self.orders_list = param_grid.pop("order", _default_dict["order"])
            self.step_list = param_grid.pop("step", _default_dict["step"])
        else:
            self.orders_list = [None]
            self.step_list = [None]

        self.n_features_list = param_grid.pop("n_features", _default_dict["n_features"])

        self.extractor_param_grid = {
            "stdA": param_grid.pop("stdA", _default_dict["stdA"]),
            "stdB": param_grid.pop("stdB", _default_dict["stdB"]),
            "std0": param_grid.pop("std0", _default_dict["std0"]),
            "activation": param_grid.pop("activation", _default_dict["activation"]),
        }

        # ++++++++++++++++++++++ Random Fourier Features params ++++++++++++++++++++++

        self.n_fourier_features_list = param_grid.pop(
            "n_fourier_features", _default_dict["n_fourier_features"]
        )
        self.bandwidth_ratios = param_grid.pop("bandwidth", _default_dict["bandwidth"])

        # ++++++++++++++++++++++ Ridge forecaster params ++++++++++++++++++++++

        # Minimal head: Ridge regression with alpha grid
        # If user does not provide an 'alpha' grid, fall back to a small default.
        self.reg_param_grid = {
            "alpha": param_grid.pop("alpha", [1.0]),
        }

        # ++++++++++++++++++++++ Preprocessing params ++++++++++++++++++++++

        self.pre_param_grid = {
            "add_time": param_grid.pop("add_time", _default_dict_pre["add_time"]),
            "lead_lag": param_grid.pop("lead_lag", _default_dict_pre["lead_lag"]),
            "basepoint": param_grid.pop("basepoint", _default_dict_pre["basepoint"]),
            "normalize": param_grid.pop("normalize", _default_dict_pre["normalize"]),
            "max_time": param_grid.pop("max_time", _default_dict_pre["max_time"]),
            "max_len": param_grid.pop("max_len", _default_dict_pre["max_len"]),
        }

    # ==============================================================================
    # Utils methods
    # ==============================================================================

    def _validate_input(self, X, y):
        """
        Validate the input types for X and y.

        X is expected to be a 3D tensor: (n_samples, length, channels)
        y can be 1D or 2D: (n_samples,) or (n_samples, out_dim)
        """
        if not isinstance(X, jnp.ndarray):
            raise ValueError("X must be a jnp.ndarray")
        if not isinstance(y, jnp.ndarray):
            raise ValueError("y must be a jnp.ndarray")

        if X.ndim != 3:
            if X.ndim == 2 and self.type == "cde":
                X = X[..., None]
            else:
                raise ValueError("X must be a 3D tensor (or 2D for CDE, which will be promoted)")

        if y.ndim == 1:
            y = y[:, None]
        elif y.ndim > 2:
            raise ValueError("y must be 1D or 2D for forecasting")

        return X, y

    def _get_feature_extractor(self, n_features, extractor_params, order=None, step=None):
        """
        Get the feature extractor from the parameter dictionaries.
        """

        if self.type == "rde":
            assert order is not None, "order must be specified for RDE"
            assert step is not None, "step must be specified for RDE"

            feature_extractor = RandomRDE(
                self.key(),
                n_features=n_features,
                order=order,
                step=step,
                config=extractor_params,
                cache=self.cache,
                **extractor_params,
            )
        else:
            feature_extractor = RandomCDE(
                self.key(),
                n_features=n_features,
                config=extractor_params,
                cache=self.cache,
            )

        return feature_extractor

    def _get_extractor_params_combinations(self) -> Iterable:
        """
        Get all combinations of extractor parameters for the grid search
        such that the activations are in the inner loop.
        """

        param_grid = dict(self.extractor_param_grid)
        activation_list = param_grid.pop("activation")

        for params in ParameterGrid(param_grid):
            for activation in activation_list:
                params = dict(params)
                params["activation"] = activation
                yield params

    def _verbose_helper(self, msg: str, level: int = logging.INFO):
        if self.verbose is False:
            return
        elif self.verbose is True:
            print(msg)
        elif self.verbose == "logger":
            self.logger.log(msg, level=level)

    # ==============================================================================
    # Fit / evaluation helpers
    # ==============================================================================

    def _get_regressor(self, reg_params: Dict = {}):
        """
        Build a (potentially multi-output) Ridge regressor from parameters.
        """
        return Ridge(**reg_params)

    def evaluate_extractor_forecaster(self, X: jnp.ndarray, y: jnp.ndarray, sig_params: dict):
        """
        Loops over extractor params (outer) and forecaster params (inner),
        computing mean CV scores.

        We use negative mean squared error as the score (higher is better).
        """

        records = []

        order = sig_params["order"]
        step = sig_params["step"]
        n_features = sig_params["n_features"]

        for extractor_params in self._get_extractor_params_combinations():

            try:
                extractor = self._get_feature_extractor(
                    n_features,
                    extractor_params,
                    order=order,
                    step=step,
                )

                features = extractor.get_features(
                    X,
                    batch_size=self.batch_size,
                    return_interval=False,
                    use_cache=True,
                )

            except Exception as e:
                _dict = {_key: None for _key in self.reg_param_grid.keys()}
                _params = {**sig_params, **extractor_params}

                self._verbose_helper(
                    f"Failed to get features for params={_params}", level=logging.WARNING
                )

                results_ = {
                    **sig_params,
                    **extractor_params,
                    **_dict,
                    "normalize_feat": False,
                    "cv_score": -jnp.inf,
                }
                records.append(results_)
                continue

            # Convert to numpy for sklearn
            X_feat = np.asarray(jax.device_get(features))
            y_np = np.asarray(jax.device_get(y))

            for normalize_feat in self.normalize_feat_list:

                try:
                    X_feat_norm = X_feat
                    if normalize_feat:
                        norms = np.linalg.norm(X_feat_norm, axis=1, keepdims=True) + EPS_
                        X_feat_norm = X_feat_norm / norms

                    # Cross-validation
                    kf = KFold(
                        n_splits=self.n_splits,
                        shuffle=self.shuffle,
                        random_state=self.random_state,
                    )

                    cv_mses = []

                    for train_idx, val_idx in kf.split(X_feat_norm):
                        X_tr, X_val = X_feat_norm[train_idx], X_feat_norm[val_idx]
                        y_tr, y_val = y_np[train_idx], y_np[val_idx]

                        for reg_params in ParameterGrid(self.reg_param_grid):
                            reg = self._get_regressor(reg_params)
                            reg.fit(X_tr, y_tr)
                            y_pred = reg.predict(X_val)
                            mse = mean_squared_error(y_val, y_pred)
                            cv_mses.append((mse, reg_params))

                    if len(cv_mses) == 0:
                        best_cv_score = -jnp.inf
                        best_reg_params = {k: None for k in self.reg_param_grid.keys()}
                    else:
                        # Choose reg_params with minimal MSE (max negative MSE)
                        best_mse, best_reg_params = min(cv_mses, key=lambda t: t[0])
                        best_cv_score = -best_mse

                    results_ = {
                        **sig_params,
                        **extractor_params,
                        **best_reg_params,
                        "normalize_feat": normalize_feat,
                        "cv_score": best_cv_score,
                    }
                    records.append(results_)

                except Exception as e:
                    _dict = {_key: None for _key in self.reg_param_grid.keys()}
                    _params = {**sig_params, **extractor_params}

                    self._verbose_helper(
                        f"Failed to fit forecaster for params={_params}",
                        level=logging.DEBUG,
                    )

                    results_ = {
                        **sig_params,
                        **extractor_params,
                        **_dict,
                        "normalize_feat": normalize_feat,
                        "cv_score": -jnp.inf,
                    }
                    records.append(results_)

        df = pd.DataFrame(records)

        if df.empty or all(df.cv_score.isna()):
            best_model = {}
        else:
            best_model_idx = df["cv_score"].idxmax()
            best_model = df.loc[best_model_idx].to_dict()

        return df, best_model

    def _fit(self, X: jnp.ndarray, y: jnp.ndarray):
        """
        Loops over n_features and n_features_fourier, calls evaluate_extractor_forecaster,
        concatenates DataFrames and collects best-lines.
        """

        all_dfs = []
        best_models = []
        self.cache = Cache()

        for n_fourier_feat in self.n_fourier_features_list:

            self._verbose_helper(f"N_fourier_features = {n_fourier_feat}")

            bandwidth_list = self.bandwidth_list if n_fourier_feat is not None else [None]
            for bandwidth in bandwidth_list:

                if n_fourier_feat is not None:
                    _n_fourier_feat = (
                        n_fourier_feat // 2 if self.rff_type == "2d" else n_fourier_feat
                    )

                    rff_cls = RandomFourierFeatures(
                        self.key(),
                        method=self.rff_type,
                        n_features=_n_fourier_feat,
                        bandwidth=bandwidth,
                        cache=self.cache,
                    )

                    X_rff = rff_cls.get_features(X, use_cache=True)
                else:
                    X_rff = X
                    n_fourier_feat = "None"
                    bandwidth = "None"

                X_rff = X_rff / X_rff.max()

                for order in self.orders_list:

                    if self.type == "rde":
                        _dim_logsigs = get_logsig_dimension(order, X_rff.shape[-1])
                        if _dim_logsigs > self.max_dim_logsigs:
                            continue

                    for step in self.step_list:
                        for n_feat in self.n_features_list:

                            if self.type == "cde":
                                self._verbose_helper(f"  Bandwidth = {bandwidth}")
                            else:
                                self._verbose_helper(
                                    f"  Order = {order}, Step = {step}, Bandwidth = {bandwidth}"
                                )

                            sig_params = {
                                "n_fourier_features": n_fourier_feat,
                                "bandwidth": bandwidth,
                                "n_features": n_feat,
                                "order": order,
                                "step": step,
                            }

                            df, best = self.evaluate_extractor_forecaster(X_rff, y, sig_params)

                            all_dfs.append(df)
                            best_models.append(best)

        df_all_results = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        df_best_models = pd.DataFrame(best_models) if best_models else pd.DataFrame()
        if not df_best_models.empty and "cv_score" in df_best_models.columns:
            df_best_models.dropna(axis=0, subset=["cv_score"], inplace=True)

        return df_all_results, df_best_models

    def _test(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray,
        df_best_models: pd.DataFrame,
    ):

        self.cache = Cache()

        train_scores = []
        test_scores = []

        for _, row in df_best_models.iterrows():

            n_features = row["n_features"]
            n_fourier_feat = row["n_fourier_features"]
            bandwidth = row["bandwidth"]
            order = row["order"]
            step = row["step"]
            normalize_feat = row["normalize_feat"]

            preprocess_params = {key: row[key] for key in self.pre_param_grid.keys()}
            extractor_params = {key: row[key] for key in self.extractor_param_grid.keys()}
            reg_params = {key: row[key] for key in self.reg_param_grid.keys()}

            preprocessing_class = Preprocessor(**preprocess_params)
            X_transformed = preprocessing_class.fit_transform(X)
            X_test_transformed = preprocessing_class.transform(X_test)

            if n_fourier_feat != "None":
                _n_fourier_feat = (
                    n_fourier_feat // 2 if self.rff_type == "2d" else n_fourier_feat
                )

                rff_cls = RandomFourierFeatures(
                    self.key(),
                    method=self.rff_type,
                    n_features=_n_fourier_feat,
                    bandwidth=bandwidth,
                    cache=self.cache,
                )

                X_rff = rff_cls.get_features(X_transformed, use_cache=True)
                X_rff_test = rff_cls.get_features(X_test_transformed, use_cache=True)
            else:
                X_rff = X_transformed
                X_rff_test = X_test_transformed

            _max = X_rff.max()
            X_rff = X_rff / _max
            X_rff_test = X_rff_test / _max

            extractor = self._get_feature_extractor(
                n_features,
                extractor_params,
                order=order,
                step=step,
            )

            features_train = extractor.get_features(
                X_rff,
                batch_size=self.batch_size,
                return_interval=False,
                use_cache=True,
            )

            if self.type == "cde":
                features_test = extractor.get_features(
                    X_rff_test,
                    batch_size=self.batch_size,
                    return_interval=False,
                    use_cache=True,
                )
            else:
                features_test = extractor.get_features(
                    X_rff_test,
                    batch_size=self.batch_size,
                    return_interval=False,
                    use_cache=True,
                    testing=True,
                )

            X_feat_train = np.asarray(jax.device_get(features_train))
            X_feat_test = np.asarray(jax.device_get(features_test))
            y_np = np.asarray(jax.device_get(y))
            y_test_np = np.asarray(jax.device_get(y_test))

            if normalize_feat:
                norms_tr = np.linalg.norm(X_feat_train, axis=1, keepdims=True) + EPS_
                X_feat_train = X_feat_train / norms_tr
                norms_te = np.linalg.norm(X_feat_test, axis=1, keepdims=True) + EPS_
                X_feat_test = X_feat_test / norms_te

            reg = self._get_regressor(reg_params)
            reg.fit(X_feat_train, y_np)

            y_pred_train = reg.predict(X_feat_train)
            y_pred_test = reg.predict(X_feat_test)

            # Use negative MSE for consistency with cv_score (higher is better)
            train_score = -mean_squared_error(y_np, y_pred_train)
            test_score = -mean_squared_error(y_test_np, y_pred_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

        df_best_models["train_score"] = train_scores
        df_best_models["test_score"] = test_scores

        return df_best_models

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        X_test: Optional[jnp.ndarray] = None,
        y_test: Optional[jnp.ndarray] = None,
        name: str = "results_forecasting",
        save: bool = True,
    ):
        """
        Fit the GridSearchForecaster model.

        X : (n_samples, length, channels) jnp.ndarray
            Training input paths (already windowed for forecasting).
        y : (n_samples,) or (n_samples, out_dim) jnp.ndarray
            Training targets (single or multi-output).
        X_test, y_test : optional
            Test set for out-of-sample evaluation.
        """

        X, y = self._validate_input(X, y)

        if X_test is None or y_test is None:
            testing = False
        else:
            testing = True
            X_test, y_test = self._validate_input(X_test, y_test)

        self._verbose_helper("Starting forecasting grid search...")
        _pre_paramgrid_size = len(ParameterGrid(self.pre_param_grid))

        # Update bandwidth similarly to GridSearchSVC
        suggested_bandwidth = suggest_bandwidth(X)
        _bandwidth_list = [suggested_bandwidth * br for br in self.bandwidth_ratios]
        self.bandwidth_list = _bandwidth_list + [1.0, 1.25, 0.75]

        all_dfs = []
        best_models = []

        for i, pre_params in enumerate(ParameterGrid(self.pre_param_grid)):

            self._verbose_helper(
                f"Starting Preprocessing combo {i + 1} out of {_pre_paramgrid_size}"
            )

            preprocessing_class = Preprocessor(**pre_params)
            X_transformed = preprocessing_class.fit_transform(X)

            df_all_results, df_best_models = self._fit(X_transformed, y)

            all_dfs.append(df_all_results.assign(**pre_params))
            best_models.append(df_best_models.assign(**pre_params))

        df_all_results = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        df_best_models = pd.concat(best_models, ignore_index=True) if best_models else pd.DataFrame()

        params_to_filter = [
            "n_features",
            "n_fourier_features",
            "order",
            "step",
            "lead_lag",
        ]

        if not df_best_models.empty and "cv_score" in df_best_models.columns:
            idx = df_best_models.groupby(params_to_filter, dropna=False)["cv_score"].idxmax()
            df_best_models = df_best_models.loc[idx.values].reset_index(drop=True)

        if testing and not df_best_models.empty:
            df_best_models = self._test(X, y, X_test, y_test, df_best_models)

        if not df_best_models.empty:
            _mask = df_best_models.nunique(dropna=False) > 1
            for col in [
                "train_score",
                "cv_score",
                "test_score",
                "activation",
                "stdA",
                "stdB",
                "std0",
            ]:
                if col in _mask.index:
                    _mask.loc[col] = True
            df_best_models = df_best_models.loc[:, _mask]
            df_best_models.reset_index(drop=True, inplace=True)

        if save:
            df_all_results.to_csv(name + "_results.csv", index=False)
            df_best_models.to_csv(name + "_best_results.csv", index=False)
        else:
            return df_all_results, df_best_models

