import jax
import jax.numpy as jnp

try:
    from cuml.svm import LinearSVC as LinearSVConGPU
    cuml_available = True
except:
    cuml_available = False

from sklearn.svm import LinearSVC as LinearSVConCPU
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from typing import Optional
from typing import Dict


class LinearSVC:
    """
    Wrapper for a linear SVM that uses scikit-learn on CPU and cuML on GPU.
    """
    def __init__(self,
                 gpu : bool = False,
                 penalty : str = 'l2',
                 C : float = 1.0,
                 max_iter : int = 1000,
                 fit_intercept : bool = True,
                 **kwargs):

        if gpu and any(d.platform == "gpu" for d in jax.devices()) and cuml_available:
            # Use GPU if available and cuml is installed
            self.gpu = True
        else:
            self.gpu = False

        self.hparams = {}
        
        if penalty is not None:
            self.hparams['penalty'] = penalty
        if C is not None:
            self.hparams['C'] = C   
        if max_iter is not None:
            self.hparams['max_iter'] = max_iter
        if fit_intercept is not None:
            self.hparams['fit_intercept'] = fit_intercept
            
        # Initialize the model
        if self.gpu:
            
            self.hparams['tol'] = kwargs.get('tol', None)
            self.model = LinearSVConGPU(**self.hparams)

        else:

            self.hparams['tol'] = kwargs.get('tol', 1e-4)
            self.hparams['intercept_scaling'] = kwargs.get('intercept_scaling', 1.0)
            self.hparams['dual'] = kwargs.get('dual', 'auto')

            self.model = LinearSVConCPU(**self.hparams)

    # ----------------------------- Validation methods ----------------------------- 

    def _validate_input_type(self, 
                             features : jnp.ndarray, 
                             y : Optional[jnp.ndarray] = None) -> None:
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
        Fit the linear SVC model, dispatching to GPU or CPU implementation.
        """
        self._validate_input_type(features, y)
        self.model.fit(features, y)


    def fit_gridsearch(self,
                       features: jnp.ndarray,
                       y: jnp.ndarray,
                       svc_grid : Dict,
                       cv : int = 4,
                       stratified : bool = True):
        """
        Fit the model using grid search.
        """

        self._validate_input_type(features, y)

        # Set up folds
        splits = min(cv, jnp.min(jnp.bincount(y)).item())
        self.kfold = StratifiedKFold(n_splits=splits) if stratified else KFold(n_splits=splits)

        self.model = GridSearchCV(self.model, svc_grid, cv=self.kfold)
        self.model.fit(features, y)


    def predict(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Predict using the fitted linear SVC model.
        """
        self._validate_input_type(features, None) 
        return self.model.predict(features)

    def score(self, features: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Compute the accuracy of the model on the given features and labels.
        """
        self._validate_input_type(features, y)
        return self.model.score(features, y)
    
    @property
    def coef_(self) -> jnp.ndarray:
        """
        Get the coefficients of the linear SVC model.
        """
        return self.model.coef_
    
    @property
    def intercept_(self) -> jnp.ndarray:
        """
        Get the intercept of the linear SVC model.
        """
        return self.model.intercept_
    
    @property
    def get_params(self) -> dict:
        """
        Get the parameters of the linear SVC model.
        """
        return self.model.get_params()
    
    
    @property
    def best_params(self) -> Dict:
        """
        Get the best parameters of the linear SVC model.
        """
        return self.model.best_params_
    
    @property
    def best_score(self) -> float:
        """
        Get the best score of the linear SVC model.
        """
        return self.model.best_score_

    @property
    def best_estimator(self):
        """
        Get the best estimator of the linear SVC model.
        """
        return self.model.best_estimator_
    
    @property
    def cv_results(self) -> Dict:
        """
        Get the cross-validation results of the linear SVC model.
        """
        return self.model.cv_results_
    
    @property
    def refit_time(self) -> float:
        """
        Get the refit time of the linear SVC model.
        """
        return self.model.refit_time_


    


