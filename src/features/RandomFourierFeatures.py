import jax
import jax.numpy as jnp
import math
from functools import partial

from typing import Optional
from typing import Union

from ..utils.cache import Cache
from ..utils.checks import _check_positive_integer_value, _check_positive_value

from ..utils.random import scale_matrices
from ..utils.random import gaussian_matrix_sampler
from ..utils.random import uniform_vector_sampler
from ..utils.random import KeyGen

class RandomFourierFeatures2D:

    def __init__(self, 
                 key : KeyGen,
                 n_features : int, 
                 bandwidth : float = 1.0, 
                 cache : Optional[Cache] = None):
        
        self.n_features = n_features
        self.bandwidth = bandwidth
        self.cache = cache
        self.key = key

        self._validate_params()


    # ----------------------------- Validation methods ----------------------------- 

    def _validate_params(self):
        """
        Validates the parameters of the class.
        """
        _check_positive_integer_value(self.n_features, "n_features")
        _check_positive_value(self.bandwidth, "bandwidth")


    def _validate_input(self, X: jnp.ndarray):
        """
        Validates the input data. Moves it to self device

        Raises:
            ValueError: If the input data is not a Tensor, or if the input data is not 2D or 3D.
        """
        if not isinstance(X, jnp.ndarray):
            raise TypeError("Input must be a JAX ndarray.")
        
        if X.ndim == 2:
            return X[None, :]
        elif X.ndim == 3:
            return X
        else:
            raise ValueError("Input data must be either 2D or 3D (batch)")
        
    # ----------------------------- Cache methods ----------------------------- 

    def get_cache(self) -> Cache:
        """
        Returns the cache object.
        """
        if self.cache is None:
            self.cache = Cache()
        
        return self.cache
    
    
    def set_cache(self, cache : Cache) -> None:
        """
        Sets the cache object.
        """
        if not isinstance(cache, Cache):
            raise ValueError("Cache must be of type Cache")
        self.cache = cache
    

    def _update_cache(self, matrices : jnp.ndarray, bandwidth : float) -> None:
        """
        Updates the cache with the given matrices and bandwidth.
        """

        self.cache.set('fourier_matrices', matrices)
        self.cache.set('bandwidth', bandwidth)


    def _validate_cache(self, input_dim : int) -> bool:
        """
        Validates the cache.
        """
        if self.cache is None:
            self.cache = Cache()
            return False
        
        if not self.cache.has('fourier_matrices'):
            return False
        if not self.cache.has('bandwidth'):
            return False
        
        matrix_input_dim, matrix_n_features = self.cache.get('fourier_matrices').shape

        if matrix_n_features != self.n_features:
            return False
        
        if matrix_input_dim != input_dim:
            return False
        
        return True
    
    # ----------------------------- Main methods ----------------------------- 

    def _get_random_matrices(self, input_dim : int, use_cache : bool) -> jnp.ndarray:
        """
        Initializes the random weights for the Fourier features.

        Args:
            input_dim : The dimension of the input data.

        Returns:
            An array of shape (input_dim, n_features) with random weights.
        """

        if use_cache and self._validate_cache(input_dim):

            return scale_matrices(self.cache.get('fourier_matrices'), 
                                  1/self.bandwidth,
                                  1/self.cache.get('bandwidth'),
                                  self.key())
            
        return gaussian_matrix_sampler(self.key(),
                                       input_dim,
                                       self.n_features,
                                       std = 1/self.bandwidth)
    
    @staticmethod
    @partial(jax.jit, static_argnames=['n_features'])
    def _compute_features(X : jnp.ndarray, 
                          n_features : int, 
                          fourier_matrices : jnp.ndarray) -> jnp.ndarray:

        scale = math.sqrt(1 / n_features)

        proj = jnp.matmul(X, fourier_matrices)
        features = jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)
        features = features * scale
        return features
    
    def get_features(self, X : jnp.ndarray, use_cache : bool = False) -> jnp.ndarray:
        """
        Computes the random Fourier features for the input data.

        Args:
            X : The input data of shape (batch_size, input_dim) or (input_dim,).
            use_cache : Whether to use the cache for the random matrices.

        Returns:
            An array of shape (batch_size, 2 * n_features) or (2 * n_features,) with the random Fourier features.
        """

        X = self._validate_input(X)
        input_dim = X.shape[-1]

        fourier_matrices = self._get_random_matrices(input_dim, use_cache)

        if use_cache:
            self._update_cache(fourier_matrices, self.bandwidth)

        return self._compute_features(X, self.n_features, fourier_matrices)
    
    

class RandomFourierFeatures1D:

    def __init__(self, 
                 key : KeyGen,
                 n_features : int, 
                 bandwidth : float = 1.0, 
                 cache : Optional[Cache] = None):
        
        self.n_features = n_features
        self.bandwidth = bandwidth
        self.cache = cache
        self.key = key

        self._validate_params()


    # ----------------------------- Validation methods ----------------------------- 

    def _validate_params(self):
        """
        Validates the parameters of the class.
        """
        _check_positive_integer_value(self.n_features, "n_features")
        _check_positive_value(self.bandwidth, "bandwidth")


    def _validate_input(self, X: jnp.ndarray):
        """
        Validates the input data. Moves it to self device

        Raises:
            ValueError: If the input data is not a Tensor, or if the input data is not 2D or 3D.
        """
        if not isinstance(X, jnp.ndarray):
            raise TypeError("Input must be a JAX ndarray.")
        
        if X.ndim == 2:
            return X[None, :]
        elif X.ndim == 3:
            return X
        else:
            raise ValueError("Input data must be either 2D or 3D (batch)")
        
    # ----------------------------- Cache methods ----------------------------- 

    def get_cache(self) -> Cache:
        """
        Returns the cache object.
        """
        if self.cache is None:
            self.cache = Cache()
        
        return self.cache
    
    
    def set_cache(self, cache : Cache) -> None:
        """
        Sets the cache object.
        """
        if not isinstance(cache, Cache):
            raise ValueError("Cache must be of type Cache")
        self.cache = cache
    

    def _update_cache(self, matrices : jnp.ndarray, shift : jnp.ndarray, bandwidth : float) -> None:
        """
        Updates the cache with the given matrices and bandwidth.
        """

        self.cache.set('fourier_matrices', matrices)
        self.cache.set('fourier_shift', shift)
        self.cache.set('bandwidth', bandwidth)


    def _validate_cache(self, input_dim : int) -> bool:
        """
        Validates the cache.
        """
        if self.cache is None:
            self.cache = Cache()
            return False
        
        if not self.cache.has('fourier_matrices'):
            return False
        if not self.cache.has('bandwidth'):
            return False

        if not self.cache.has('fourier_shift'):
            return False

        matrix_input_dim, matrix_n_features = self.cache.get('fourier_matrices').shape

        if matrix_n_features != self.n_features:
            return False
        
        if matrix_input_dim != input_dim:
            return False
        
        shift_dim = self.cache.get('fourier_shift').shape[0]
        if shift_dim != self.n_features:
            return False

        return True
    
    # ----------------------------- Main methods ----------------------------- 

    def _get_random_matrices(self, input_dim : int, use_cache : bool) -> jnp.ndarray:
        """
        Initializes the random weights for the Fourier features.

        Args:
            input_dim : The dimension of the input data.

        Returns:
            An array of shape (input_dim, n_features) with random weights.
        """

        if use_cache and self._validate_cache(input_dim):

            fourier_matrices = scale_matrices(self.cache.get('fourier_matrices'), 
                                              1/self.bandwidth,
                                              1/self.cache.get('bandwidth'),
                                              self.key())
            
            fourier_shift = self.cache.get('fourier_shift')

            return fourier_matrices, fourier_shift
            
        fourier_matrices = gaussian_matrix_sampler(self.key(),
                                                   input_dim,
                                                   self.n_features,
                                                   std = 1/self.bandwidth)

        fourier_shift = 2*jnp.pi * uniform_vector_sampler(self.key(), self.n_features)

        return fourier_matrices, fourier_shift
    
    @staticmethod
    @partial(jax.jit, static_argnames=['n_features'])
    def _compute_features(X : jnp.ndarray, 
                          n_features : int,
                          fourier_matrices : jnp.ndarray, 
                          fourier_shift : jnp.ndarray) -> jnp.ndarray:
        """
        Computes the random Fourier features for the input data.
        """
        scale = math.sqrt(2 / n_features)

        proj = jnp.matmul(X, fourier_matrices) 
        features = jnp.cos(proj + jnp.expand_dims(fourier_shift, (0,1)))
        features = features * scale
        return features
    

    def get_features(self, X : jnp.ndarray, use_cache : bool = False) -> jnp.ndarray:
        """
        Gets the random Fourier features for the input data.
        """

        X = self._validate_input(X)
        input_dim = X.shape[-1]

        fourier_matrices, fourier_shift = self._get_random_matrices(input_dim, use_cache)

        if use_cache:
            self._update_cache(fourier_matrices, fourier_shift, self.bandwidth)

        return self._compute_features(X, self.n_features, fourier_matrices, fourier_shift)


class RandomFourierFeatures:
    """
    A class to compute random Fourier features for 1D and 2D data.
    """

    def __new__(cls,
                key : Union[int, jax.Array] = 42,
                method : str = '1D',
                n_features : int = 100,
                bandwidth : float = 1.0,
                cache : Optional[Cache] = None):

        assert method.lower() in ['1d', '2d'], "Method must be either '1D' or '2D'"

        key = KeyGen(key)

        if method.lower() == '1d':
            return RandomFourierFeatures1D(key=key,
                                           n_features=n_features,
                                           bandwidth=bandwidth,
                                           cache=cache)
        else:
            return RandomFourierFeatures2D(key=key,
                                           n_features=n_features,
                                           bandwidth=bandwidth,
                                           cache=cache)
    
