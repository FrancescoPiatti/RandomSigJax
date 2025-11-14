import jax
import jax.numpy as jnp

try:
    from cuml.linear_model import Lasso as LassoOnGPU
    cuml_available = True
except Exception:
    cuml_available = False

from sklearn.linear_model import Lasso as LassoOnCPU
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from typing import Optional
from typing import Dict


class LassoRegressor:
    """
    Wrapper for Lasso regression (L1-regularised linear regression)
    that uses scikit-learn on CPU and cuML on GPU (if available).

    - Accepts jnp.ndarray inputs.
    - Standard Lasso is single-output; y is expected to be 1D.
    """

    def __init__(self,
                 gpu: bool = False,
                 alpha: float = 1.0,
                 max_iter: int = 1000,
                 fit_intercept: bool = True,
                 **kwargs):

        # Check GPU is available and cuML is installed
        if gpu and any(d.platform == "gpu" for d in jax.devices()) and cuml_available:
            gpu = True
        else:
            gpu = False

        self.hparams: Dict = {}

        if alpha is not None:
            self.hparams["alpha"] = alpha
        if max_iter is not None:
            self.hparams["max_iter"] = max_iter
        if fit_intercept is not None:
            self.hparams["fit_intercept"] = fit_intercept

        # Common extra kwargs
        if "tol" in kwargs and kwargs["tol"] is not None:
            self.hparams["tol"] = kwargs["tol"]
        if "selection" in kwargs and kwargs["selection"] is not None:
            self.hparams["selection"] = kwargs["selection"]
        if "warm_start" in kwargs and kwargs["warm_start"] is not None:
            self.hparams["warm_start"] = kwargs["warm_start"]

        # Initialize the model
        if gpu:
            # cuML Lasso
            self.model = LassoOnGPU(**self.hparams)
        else:
            # scikit-learn Lasso
            self.model = LassoOnCPU(**self.hparams)

    # ----------------------------- Validation methods -----------------------------

    def _validate_input_type(self,
                             features: jnp.ndarray,
                             y: Optional[jnp.ndarray] = None) -> None:
        """
        Validate the input types for X and y.
        """
        if not isinstance(features, jnp.ndarray):
            raise TypeError("X must be a jnp.ndarray")
        if y is not None and not isinstance(y, jnp.ndarray):
            raise TypeError("y must be a jnp.ndarray")

        if features.ndim != 2:
            raise ValueError("Features must be a 2D tensor")

    # ----------------------------- Model methods -----------------------------

    def fit(self,
            features: jnp.ndarray,
            y: jnp.ndarray) -> None:
        """
        Fit the Lasso regression model.
        """
        self._validate_input_type(features, y)
        self.model.fit(features, y)


    def fit_gridsearch(self,
                       features: jnp.ndarray,
                       y: jnp.ndarray,
                       reg_grid: Dict,
                       cv: int = 4,
                       shuffle: bool = True,
                       scoring: Optional[str] = None):
        """
        Fit the model using grid search over reg_grid.

        Parameters
        ----------
        features : jnp.ndarray
            Training features (n_samples, n_features).
        y : jnp.ndarray
            Training targets (n_samples).
        reg_grid : dict
            Dictionary/parameter grid for Lasso hyperparameters.
        cv : int
            Number of KFold splits.
        shuffle : bool
            Whether to shuffle data before splitting in KFold.
        scoring : Optional[str]
            Scikit-learn scoring string. If None, uses estimator's default (R^2 for Lasso).
        """

        self._validate_input_type(features, y)

        self.kfold = KFold(n_splits=cv, shuffle=shuffle)

        # Wrap current model (CPU or GPU) into GridSearchCV
        self.model = GridSearchCV(self.model, reg_grid, cv=self.kfold, scoring=scoring)
        self.model.fit(features, y)


    def predict(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Predict using the fitted Lasso regression model.
        """
        self._validate_input_type(features, None)
        return self.model.predict(features)


    def score(self, features: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Compute the score of the model on the given features and targets.

        By default (if no scoring was specified), this is R^2.
        """
        self._validate_input_type(features, y)
        return self.model.score(features, y)

    # ----------------------------- Properties -----------------------------

    @property
    def coef_(self) -> jnp.ndarray:
        """
        Get the coefficients of the Lasso regression model.
        """
        return self.model.coef_

    @property
    def intercept_(self) -> jnp.ndarray:
        """
        Get the intercept of the Lasso regression model.
        """
        return self.model.intercept_

    @property
    def get_params(self) -> dict:
        """
        Get the parameters of the Lasso regression model.
        """
        return self.model.get_params()

    @property
    def best_params(self) -> Dict:
        """
        Get the best parameters from GridSearchCV.
        """
        return self.model.best_params_

    @property
    def best_score(self) -> float:
        """
        Get the best score from GridSearchCV.
        """
        return self.model.best_score_

    @property
    def best_estimator(self):
        """
        Get the best estimator from GridSearchCV.
        """
        return self.model.best_estimator_

    @property
    def cv_results(self) -> Dict:
        """
        Get the cross-validation results from GridSearchCV.
        """
        return self.model.cv_results_

    @property
    def refit_time(self) -> float:
        """
        Get the refit time from GridSearchCV.
        """
        return self.model.refit_time_