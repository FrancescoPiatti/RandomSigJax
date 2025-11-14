import jax
import jax.numpy as jnp

import pandas as pd
import warnings
import logging

from typing import Optional
from typing import Iterable
from typing import Dict
from typing import Union

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

from .Lasso import LassoRegressor
from .Ridge import RidgeRegressor

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
from ..configs import DEFAULT_PRE_GS

EPS_ = 1e-8

class GridSearchForecasting:
    """
    Grid search over RandomCDE / RandomRDE feature extractors and
    linear regression forecasting heads (Ridge or Lasso).

    This class mirrors the design philosophy of GridSearchSVC, but
    targets *regression* tasks such as time-series forecasting. It wraps
    the full pipeline:

    - optional preprocessing of input paths (time augmentation,
      lead-lag transform, basepoint, normalisation, truncation),
    - optional Random Fourier Features (RFF) layer,
    - RandomCDE or RandomRDE feature extractor,
    - a linear regression head (RidgeRegressor or LassoRegressor,
      supporting both CPU and cuML GPU backends),

    and performs cross-validated grid search over all corresponding
    hyperparameters using regression metrics (e.g. negative MSE).

    Parameters
    ----------
    type : str
        Either 'rde' or 'cde', selecting the RandomRDE or RandomCDE
        feature extractor respectively.
    param_grid : dict
        Dictionary specifying the hyperparameter grid. It may include keys for:
          - preprocessing (e.g. 'add_time', 'lead_lag', 'basepoint',
            'normalize', 'max_time', 'max_len'),
          - differential equation features (e.g. 'n_features', 'order',
            'step', 'stdA', 'stdB', 'std0', 'activation'),
          - random Fourier features (e.g. 'n_fourier_features', 'bandwidth'),
          - regression head (e.g. 'alpha', 'max_iter', 'fit_intercept').
        Missing keys fall back to defaults from the corresponding
        DEFAULT_*_GS configurations.
    gpu : bool, optional
        If True, attempt to use GPU-accelerated regression heads via
        cuML (when available). Falls back to CPU if no compatible GPU is detected.
    head : {'ridge', 'lasso'}, optional
        Regression head type. 'ridge' uses RidgeRegressor, 'lasso' uses LassoRegressor.
    rff_type : {'1D', '2D'}, optional
        Type of Random Fourier Features to apply when 'n_fourier_features' > 0.
    seed : int, optional
        Random seed used to initialise the internal key generator for
        random CDE/RDE weights.
    verbose : bool or Logger, optional
        If False, run silently. If True, print progress messages.
        If a Logger instance is provided, messages are routed through that logger.
    batch_size : int, optional
        Batch size used when computing random CDE/RDE features.
    n_splits : int, optional
        Number of cross-validation folds. Uses standard K-fold
        (not stratified) since this is a regression task.
    shuffle : bool, optional
        Whether to shuffle data before splitting into folds.
    max_dim_logsigs : int, optional
        Maximum allowed log-signature dimension for RDE configurations.
        Larger expansions are automatically skipped.
    random_state : int or None, optional
        Random seed passed to the CV splitter when shuffling is enabled.

    Notes
    -----
    - This class expects forecasting-ready data:
        X : (n_samples, length, channels)
        y : (n_samples,) or (n_samples, n_targets)
      The user is responsible for constructing sliding windows or other
      forecasting formulations prior to calling `.fit`.
    - The optimisation criterion used during grid search is negative
      mean squared error, so higher `cv_score` corresponds to better
      performance.
    """
    def __init__(self,
                 type: str,
                 param_grid: dict,
                 gpu: bool = False,
                 head: str = "ridge",
                 rff_type: str = "1D",
                 seed: int = 42,
                 verbose: Union[bool, Logger] = False,
                 batch_size: int = 100,
                 n_splits: int = 3,
                 shuffle: bool = True,
                 max_dim_logsigs: int = 1000,
                 random_state: Optional[int] = None):

        assert type.lower() in ["rde", "cde"], "type must be 'rde' or 'cde'"
        assert rff_type.lower() in ["1d", "2d"], "rff_type must be '1D' or '2D'"
        assert head.lower() in ["ridge", "lasso"], "head must be 'ridge' or 'lasso'"

        self.type = type.lower()
        self.rff_type = rff_type.lower()
        self.head = head.lower()
        self.batch_size = batch_size
        self.key = KeyGen(seed)
        self.max_dim_logsigs = max_dim_logsigs

        # Verbosity
        if isinstance(verbose, Logger):
            self.verbose = "logger"
            self.logger = verbose
        else:
            self.verbose = verbose
            self.logger = None

        # Regression CV setup
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        # Device
        self.gpu = gpu and any(d.platform == "gpu" for d in jax.devices())
        if gpu and not self.gpu:
            warnings.warn("CUDA not available; falling back to CPU for forecasting head.")

        # Parse param grid into internal dicts
        self._get_param_dicts(param_grid.copy())


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

        # Feature normalization flag
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

        self.n_fourier_features_list = param_grid.pop("n_fourier_features", 
                                                      _default_dict["n_fourier_features"]
        )
        self.bandwidth_ratios = param_grid.pop("bandwidth", _default_dict["bandwidth"])

        # ++++++++++++++++++++++ Regression head params ++++++++++++++++++++++

        # Minimal head grid: regularisation strength alpha
        self.reg_param_grid = {
            "alpha": param_grid.pop("alpha", [1.0]),
        }

        # Optionally allow other regression head params to be grid-searched
        # other_reg_possible_params = [
        #     "max_iter",
        #     "fit_intercept",
        #     "tol",
        # ]

        # for key in list(param_grid.keys()):
        #     if key in other_reg_possible_params:
        #         self.reg_param_grid[key] = param_grid.pop(key)

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
    # Regression head helper
    # ==============================================================================

    def _get_regressor(self, reg_params: Dict = {}):
        """
        Build a Ridge or Lasso regressor (CPU or GPU) from parameters.
        """
        if self.head == "ridge":
            return RidgeRegressor(gpu=self.gpu, **reg_params)
        else:
            return LassoRegressor(gpu=self.gpu, **reg_params)


    # ==============================================================================
    # Core evaluation (CV) with fixed extractor and RFF config
    # ==============================================================================

    def evaluate_extractor_forecasting(self, X: jnp.ndarray, y: jnp.ndarray, sig_params: dict):
        """
        Loops over extractor params (outer) and regression head params (inner),
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
                    n_features=n_features,
                    extractor_params=extractor_params,
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

                self._verbose_helper(f"Failed to get features for params={_params}", level=logging.WARNING)

                results_ = {
                    **sig_params,
                    **extractor_params,
                    **_dict,
                    "normalize_feat": False,
                    "cv_score": -jnp.inf,
                }
                records.append(results_)
                continue

            for normalize_feat in self.normalize_feat_list:

                try:
                    features_norm = features
                    if normalize_feat:
                        norms = jnp.linalg.norm(features_norm, axis=1, keepdims=True) + EPS_
                        features_norm = features_norm / norms

                    # Regression head grid search (uses KFold internally)
                    reg = self._get_regressor()
                    reg.fit_gridsearch(
                        features_norm,
                        y,
                        self.reg_param_grid,
                        cv=self.n_splits,
                        shuffle=self.shuffle,
                        random_state=self.random_state,
                        scoring="neg_mean_squared_error",
                    )

                    results_ = {
                        **sig_params,
                        **extractor_params,
                        **reg.best_params,
                        "normalize_feat": normalize_feat,
                        "cv_score": reg.best_score,
                    }
                    records.append(results_)

                except Exception as e:
                    _dict = {_key: None for _key in self.reg_param_grid.keys()}
                    _params = {**sig_params, **extractor_params}

                    self._verbose_helper(f"Failed to fit forecaster for params={_params}", level=logging.DEBUG)

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

    # ==============================================================================
    # Inner fit over RFF + extractor configs
    # ==============================================================================

    def _fit(self, X: jnp.ndarray, y: jnp.ndarray):
        """
        Loops over n_fourier_features and extractor hyperparameters, calling
        evaluate_extractor_forecasting and collecting results.
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
                                self._verbose_helper(f"  Order = {order}, Step = {step}, Bandwidth = {bandwidth}")

                            sig_params = {
                                "n_fourier_features": n_fourier_feat,
                                "bandwidth": bandwidth,
                                "n_features": n_feat,
                                "order": order,
                                "step": step,
                            }

                            df, best = self.evaluate_extractor_forecasting(X_rff, y, sig_params)

                            all_dfs.append(df)
                            best_models.append(best)

        df_all_results = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        df_best_models = pd.DataFrame(best_models) if best_models else pd.DataFrame()
        if not df_best_models.empty and "cv_score" in df_best_models.columns:
            df_best_models.dropna(axis=0, subset=["cv_score"], inplace=True)

        return df_all_results, df_best_models

    # ==============================================================================
    # Test phase (refit on train, evaluate on test)
    # ==============================================================================

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

            # RFF layer
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

            # Extractor
            extractor = self._get_feature_extractor(
                n_features=n_features,
                extractor_params=extractor_params,
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

            if normalize_feat:
                norms_tr = jnp.linalg.norm(features_train, axis=1, keepdims=True) + EPS_
                features_train = features_train / norms_tr
                norms_te = jnp.linalg.norm(features_test, axis=1, keepdims=True) + EPS_
                features_test = features_test / norms_te

            # Regressor fit on full train and test evaluation
            reg = self._get_regressor(reg_params)
            reg.fit(features_train, y)

            y_pred_train = reg.predict(features_train)
            y_pred_test = reg.predict(features_test)

            train_score = -mean_squared_error(y, y_pred_train)
            test_score = -mean_squared_error(y_test, y_pred_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

        df_best_models["train_score"] = train_scores
        df_best_models["test_score"] = test_scores

        return df_best_models

    # ==============================================================================
    # Public API
    # ==============================================================================

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
        Fit the GridSearchForecasting model.

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

        # Bandwidth suggestion (same as GridSearchSVC)
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

        # Filter best per (n_features, n_fourier_features, order, step, lead_lag)
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