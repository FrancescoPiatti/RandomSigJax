import jax
import jax.numpy as jnp
import jax.random as random

from typing import Optional
from typing import Union
from typing import Callable

from dataclasses import dataclass

from ..utils.checks import _check_positive_integer_value, _check_non_negative_value
from ..utils.random import KeyGen, gaussian_matrices_sampler_CDE, scale_matrices_cde
from ..utils.activation_dict import ACTIVATION_DICT
from ..utils.cache import Cache

from ..configs import DEFAULT_CONFIG_RCDE

@dataclass
class VectorFieldCDE:
    """
    CDE driving vector field.

    Parameters
    ----------
    input_dim : int
        d — driving path dimension.
    n_features : int
        n — feature dimension for z.
    activation : str | callable
        Nonlinearity φ applied to z before augmentation with 1.
    matrices : jax.Array
        Parameter block (d, n, n+1) holding [A | b].
    """
    
    input_dim: int
    n_features: int
    activation: Callable
    matrices: jnp.ndarray

    # Build from random init (preferred)
    @classmethod
    def from_random(cls,
                    input_dim : int,
                    n_features : int,
                    key : jax.Array,
                    activation : Callable,
                    stdA : float = 1.0,
                    stdB : float = 1.0):

        matrices = gaussian_matrices_sampler_CDE(key, n_features, input_dim, stdA, stdB)

        return cls(input_dim=input_dim,
                   n_features=n_features,
                   activation=activation,
                   matrices=matrices)


    # Build from cache (with rescaling)
    @classmethod
    def from_cache(cls,
                   cache : Cache,
                   input_dim : int,
                   n_features : int,
                   key : jax.Array,
                   activation : Callable,
                   stdA : float = 1.0,
                   stdB : float = 1.0):
        """
        Creates the vector fileds from the cache
        """

        # Expected cache keys: 'matrices', 'stdA', 'stdB'
        mats_cached = cache.get('matrices')
        stdA_cached, stdB_cached = cache.get('stdA'), cache.get('stdB')
        mats_scaled = scale_matrices_cde(mats_cached, stdA, stdB, stdA_cached, stdB_cached, key)

        return cls(input_dim=input_dim,
                   n_features=n_features,
                   activation=activation,
                   matrices=mats_scaled)

    # @staticmethod
    # @partial(jax.jit, static_argnames=('activation'))
    # def apply(z, activation, A, b, dx):
    #     return jnp.einsum('dnm, d -> nm', A, dx) @ activation(z) + jnp.einsum('dn,d->n', b, dx)



# def cdeint(field, X, features_0):
#     """
#     Computes the path solving the CDE
#         dY_t = \sum_{i=1}^d ( A_i@activation(Y_t) + b_i )*dx^i_t
#     with initial value Y_0

#     :param activation: (1) -> (1)
#     :param AA: (d, N, N)
#     :param bb: (N, d)
#     :param dx: (times, d)
#     :param Y_0: (N,)
#     :return: Y : (times, N) 
#     """

#     @partial(jax.jit, static_argnames=('act'))
#     def _integrate(act, A, b, z_0, dx):

#         times = dx.shape[0] + 1
#         z_t = jnp.zeros(shape=(times, A.shape[1])).at[0,:].set(z_0)

#         def body_fun(t, array):
#             return array.at[t+1, :].set(array[t] + field.apply(array[t], act, A, b, dx[t]))

#         return jax.lax.fori_loop(0, times-1, body_fun, z_t)
    
#     @partial(jax.jit, static_argnames=('act'))
#     def _integrate_all(act, A, b, z_0, dx):

#         dx = jnp.diff(X, axis=1)

#         features = jax.vmap(lambda dx_i : _integrate(act, A, b, z_0, dx_i), in_axes=0)(dx)

#         return features

#     features = _integrate_all(field.activation, 
#                               field.matrices[..., :-1], 
#                               field.matrices[..., -1], 
#                               features_0, 
#                               X)

#     return features

def cdeint(field, X, features_0):
    """
    Solve dZ_t = sum_i (A_i @ act(Z_t) + b_i) dX_t^i, with Z_0 = features_0.
    X: (B, T, d), features_0: (B, n)
    field.matrices: (d, n, n+1) with last col = b_i
    """
    A_full = field.matrices           # (d, n, n+1)
    A = A_full[..., :-1]              # (d, n, n)
    b = A_full[..., -1]               # (d, n)
    act = field.activation            # capture in closure (no static arg)
    dx = jnp.diff(X, axis=1)          # (B, T-1, d)

    @jax.jit
    def _integrate_one(A, b, z0, dx_seq):
        """
        Single trajectory integrator.
        A: (d,n,n), b: (d,n), z0: (n,), dx_seq: (T-1,d)
        return: (T,n)
        """
        # Precompute per-step operators
        mat_dx = jnp.einsum('td,dnm->tnm', dx_seq, A)   # (T-1, n, n)
        bias_dx = jnp.einsum('td,dn->tn',  dx_seq, b)    # (T-1, n)

        def step(z_t, inputs):
            M_t, b_t = inputs
            z_next = z_t + M_t @ act(z_t) + b_t
            return z_next, z_next

        _, traj = jax.lax.scan(step, z0, (mat_dx, bias_dx))     # traj: (T-1, n)
        return jnp.concatenate([z0[None, :], traj], axis=0)

    # Batch over trajectories (batch axis 0 on z0 and dx)
    integrate_batched = jax.jit(jax.vmap(_integrate_one, in_axes=(None, None, None, 0)))
    features = integrate_batched(A, b, features_0, dx)   # (B, T, n)
    return features


class RandomCDE:
    """
    This class defines the RandomCDE model, which is a neural CDE with random matrices.

    Args:
        n_features (int): Number of features in the CDE.
        activation (Callable or str): Activation function to apply to the CDE features.
        config (dict): Configuration dictionary with parameters for the CDE.
        cache (Optional[Cache]): Cache object to store precomputed matrices and features.
        device (torch.device): Device to run the model on (default is 'cpu').
    """

    def __init__(self,
                 key : Union[int, jax.Array],
                 n_features : int,
                 activation : Callable = lambda x : x,
                 config : dict = DEFAULT_CONFIG_RCDE,
                 cache : Optional[Cache] = None):

        super(RandomCDE, self).__init__()

        self.n_features = n_features
        self.activation = activation

        self.stdA = config.get('stdA', 1.0)
        self.stdB = config.get('stdB', 0.0)
        self.std0 = config.get('std0', 1.0)

        self.cache = cache
        self.key = KeyGen(key)

        self._validate_params()


    # ----------------------------- Validation methods ----------------------------- 

    def _validate_input(self, X : jnp.ndarray) -> None:
        """
        Validates the input data. Moves it to self device

        Raises:
            ValueError: If the input data is not a jnp.ndarray, or if the input data is not 2D or 3D.
        """
        if not isinstance(X, jnp.ndarray):
            raise ValueError("Input data must be a jnp.ndarray")
        
        if len(X.shape) == 2:
            return X[None, ...]
        elif len(X.shape) == 3: 
            return X
        else:
            raise ValueError("Input data must be either 2D or 3D (batch)")


    def _validate_params(self):

        _check_positive_integer_value(self.n_features, 'n_features')
        _check_non_negative_value(self.std0, 'std0')
        _check_non_negative_value(self.stdA, 'stdA')
        _check_non_negative_value(self.stdB, 'stdB')

        if isinstance(self.activation, str):
            if self.activation not in ACTIVATION_DICT:
                raise ValueError(f"Activation function '{self.activation}' not recognized.")
            self.activation = ACTIVATION_DICT[self.activation]
        
        elif not callable(self.activation):
            raise ValueError("Activation must be a callable or a string representing an activation function.")


    # ----------------------------- Cache methods ----------------------------- 

    def _validate_cache(self, input_dim : int) -> bool:
        """
        Validates the fourier_matrices in Cache.

        Args:
            input_dim (int).
        """

        if self.cache is None:
            self.cache = Cache()
            return False
        
        if not self.cache.has('matrices'):
            return False
        if not self.cache.has('stdA'):
            return False
        if not self.cache.has('stdB'):
            return False
        if not self.cache.has('std0'):
            return False
        if not self.cache.has('features_0'):
            return False
        
        matrix_input_dim, matrix_n_features, _ = self.cache.get('matrices').shape

        if matrix_n_features != self.n_features:
            return False
        if matrix_input_dim != input_dim:
            return False
        
        features_0_dim = self.cache.get('features_0').shape[0]
        if features_0_dim != self.n_features:
            return False
        
        return True


    def _update_cache(self) -> None:
        """
        Updates the cache with the current state of the model.
        """
        
        self.cache.set('stdA', self.stdA)
        self.cache.set('stdB', self.stdB)
        self.cache.set('std0', self.std0)
        self.cache.set('matrices', self.fields.matrices)
        self.cache.set('features_0', self.features_0)


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
    

    # ----------------------------- Main methods ----------------------------- 
    
    def _initialize_fields(self, input_dim, use_cache) -> None:
        """
        Initialize the CDE vector field and initial features, optionally using cached values.

        If `use_cache` is True and a valid cache exists:
          - Construct a `VectorFieldCDE` with the cached matrices.
          - Load `features_0` from the cache.
        Otherwise:
          - Create a fresh `VectorFieldCDE` without cache.
          - Sample new initial features from a normal distribution with std `self.std0`.

        After initialization, if `use_cache` is True, updates the cache with new values.

        Args:
            input_dim (int): Dimensionality of the input signal.
            use_cache (bool): Whether to attempt loading from and updating a cache.
        """

        # Attempt to reuse cached field & initial features
        if use_cache and self._validate_cache(input_dim): 
            
            # Build vector field from stored cache
            self.fields = VectorFieldCDE.from_cache(self.cache,
                                                    input_dim, 
                                                    self.n_features, 
                                                    self.key(),
                                                    self.activation, 
                                                    self.stdA, 
                                                    self.stdB)
            
            # Load initial features from cache
            self.features_0 = self.cache.get('features_0')

        else:
            # Create a new vector field without caching
            self.fields = VectorFieldCDE.from_random(input_dim, 
                                                     self.n_features, 
                                                     self.key(),
                                                     self.activation, 
                                                     self.stdA, 
                                                     self.stdB)
            
            # Sample fresh initial features from Normal(0, std0)
            self.features_0 = self.std0 * random.normal(self.key(), 
                                                        shape=(self.n_features,), 
                                                        dtype=jnp.float32)

        # If caching enabled, store updated parameters
        if use_cache:
            self._update_cache()


    def _get_features(self, 
                      X : jnp.ndarray, 
                      return_interval : bool = False):
        
        """
        Compute the CDE feature for a single batch of paths.

        Args:
            X (Tensor): Input of shape (batch, timesteps, input_channels).
            return_interval (bool): If True, returns the entire feature path over time;

        Returns:
            Tensor:
              - If `return_interval`: shape (batch, timesteps, n_features).
              - Else: shape (batch, n_features), the final timepoint features.
        """

        features = cdeint(field=self.fields, X=X, features_0=self.features_0)

        if return_interval: 
            return features / jnp.asarray(self.n_features, dtype=jnp.float32)
        else:
            return features[:, -1] / jnp.asarray(self.n_features, dtype=jnp.float32)


    def get_features(self, 
                     X : jnp.ndarray, 
                     batch_size : Optional[int] = None,
                     return_interval : bool = False,
                     use_cache : bool = False):
        
        # Ensure input tensor is valid
        X = self._validate_input(X)

        # Sizes
        batch = X.shape[0]
        input_dim = X.shape[2]

        # Initialize the CDE vector field and initial features
        self._initialize_fields(input_dim, use_cache)

        if batch_size is None:
            batch_size = batch 

        # Integrate in one shot or in minibatches
        if batch <= batch_size:
            features = self._get_features(X, return_interval)
        else:
            chunks = []
            for i in range(0, batch, batch_size):
                X_chunk = X[i : i + batch_size]
                chunks.append(self._get_features(X_chunk, return_interval))

            features = jnp.concatenate(chunks, axis=0)

        return features
    

    def get_gram(self,
                 X: jnp.ndarray,
                 Y: Optional[jnp.ndarray] = None,
                 return_interval: bool = False,
                 batch_size: Optional[int] = None,
                 use_cache: bool = False) -> jnp.ndarray:
        """
        Compute the Gram (kernel) matrix between path collections X and Y
        using CDE features extracted by this object.

        Args:
            X: jnp.ndarray of shape (batchX, timeX, input_dim).
            Y: Optional[jnp.ndarray] of shape (batchY, timeY, input_dim).
            If None, defaults to X.
            return_interval: If True, computes kernel for all timepoints:
                returns shape (batchX, batchY, timeX, timeY).
                Otherwise returns shape (batchX, batchY).
            batch_size: Optional minibatch size for feature computation.
            use_cache: Whether to use cached coefficient matrices.

        Returns:
            Gram matrix as jnp.ndarray:
                - (batchX, batchY) if return_interval is False
                - (batchX, batchY, timeX, timeY) if return_interval is True
        """

        # Compute features
        feats_X = self.get_features(X, 
                                    batch_size=batch_size,
                                    return_interval=return_interval,
                                    use_cache=use_cache)

        if Y is not None:
            feats_Y = self.get_features(Y, 
                                        batch_size=batch_size,
                                        return_interval=return_interval,
                                        use_cache=use_cache)
        else:
            feats_Y = feats_X

        if not return_interval:
            # feats_X: (batchX, n_features)
            # feats_Y: (batchY, n_features)
            gram = jnp.einsum("bf,gf->bg", feats_X, feats_Y)
        else:
            # feats_X: (batchX, timeX, n_features)
            # feats_Y: (batchY, timeY, n_features)
            # Compute kernel for all pairs of timepoints
            gram = jnp.einsum("bxf,gyf->bgxy", feats_X, feats_Y)

        return gram
        