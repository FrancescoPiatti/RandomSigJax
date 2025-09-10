import jax
import jax.numpy as jnp

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from typing import Optional
from typing import Dict


class KernelSVC:
    """
    Wrapper for a linear SVM that uses scikit-learn on CPU and cuML on GPU.
    """
    def __init__(self,
                 gpu : bool = False,
                 C : float = 1.0,
                 gamma : str = 'scale',
                 decision_function_shape : Optional[str] = 'ovo',
                 max_iter : int = 1000,
                 tol :  Optional[float] = None,
                 dual : Optional[float] = None,
                 random_state: int = None):

        # ON GPU CUML DOES NOT SUPPORT PRECOMPUTED KERNELS

        self.hparams = {}

        if C is not None:
            self.hparams['C'] = C
        if gamma is not None:
            self.hparams['gamma'] = gamma
        if max_iter is not None:
            self.hparams['max_iter'] = max_iter
        if decision_function_shape is not None:
            self.hparams['decision_function_shape'] = decision_function_shape
        if tol is not None:
            self.hparams['tol'] = tol
        if dual is not None:
            self.hparams['dual'] = dual
        if random_state is not None:
            self.hparams['random_state'] = random_state

        # if not self.gpu:
        #     self.hparams['kernel'] = 'precomputed'  # Set kernel to precomputed for SVC

        # Initialize the model
        self.model = SVC(kernel='precomputed', **self.hparams)

    # ----------------------------- Validation methods ----------------------------- 
    
    def _validate_input_type(self, 
                             gram_matrix : jnp.ndarray, 
                             y : Optional[jnp.ndarray] = None) -> None:
        """
        Validate the input types for X and y.
        """
        if not isinstance(gram_matrix, jnp.ndarray):
            raise TypeError("X must be a jnp.ndarray")
        if y is not None and not isinstance(y, jnp.ndarray):
            raise TypeError("y must be a jnp.ndarray")

        if gram_matrix.ndim != 2:
            raise ValueError("gram_matrix must be a 2D tensor")
                    

    # ----------------------------- Model methods -----------------------------

    def fit(self, 
            gram_matrix: jnp.ndarray, 
            y: jnp.ndarray) -> None:
        """
        Fit the linear SVC model, dispatching to GPU or CPU implementation.
        """
        self._validate_input_type(gram_matrix, y)
        self.model.fit(gram_matrix, y)

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

    def predict(self, gram_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Predict using the fitted SVC model.
        """
        # Validate input types
        self._validate_input_type(gram_matrix, None) 
        return self.model.predict(gram_matrix)

    def score(self, gram_matrix: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Compute the accuracy of the model on the given gram_matrix and labels.
        """
        self._validate_input_type(gram_matrix, y)
        return self.model.score(gram_matrix, y)
    
    @property
    def coef_(self) -> jnp.ndarray:
        """
        Get the coefficients of the SVC model.
        """
        return self.model.coef_
    
    @property
    def intercept_(self) -> jnp.ndarray:
        """
        Get the intercept of the SVC model.
        """
        return self.model.intercept_
    
    @property
    def get_params(self) -> dict:
        """
        Get the parameters of the SVC model.
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

    

    


