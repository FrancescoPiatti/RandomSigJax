import jax
import jax.numpy as jnp
import jax.random as random

from typing import Optional
from typing import Union
from typing import Callable

from dataclasses import dataclass

from ..utils.cache import Cache
from ..utils.random import KeyGen, scale_matrices_rde, gaussian_matrices_sampler_RDE
from ..utils.activation_dict import ACTIVATION_DICT
from ..utils.checks import _check_non_negative_value
from ..utils.lie_algebra import get_logsig_dimension
from ..signatures import get_logsignatures


@dataclass
class VectorFieldRDE:
    n_features: int
    order: int
    activation: Callable
    matrices: jnp.ndarray           # (d_logsig, n, n+1)

    @classmethod
    def from_random(cls,
                    key: jax.Array,
                    input_dim: int,
                    n_features: int,
                    order: int,
                    activation: Callable,
                    stdA: float = 1.0,
                    stdB: float = 1.0):
        
        mats = gaussian_matrices_sampler_RDE(key, n_features, input_dim, order, stdA, stdB)
        return cls(n_features=n_features, order=order, activation=activation, matrices=mats)

    @classmethod
    def from_cache(cls,
                   key: jax.Array,
                   cache: Cache,
                   input_dim: int,
                   n_features: int,
                   order: int,
                   activation: Callable,
                   stdA: float = 1.0,
                   stdB: float = 1.0):
        
        mats = cache.get('rde_matrices')  
        mats = scale_matrices_rde(mats, input_dim, order, stdA, stdB, cache.get('stdA'), cache.get('stdB'), key)
        return cls(n_features=n_features, order=order, activation=activation, matrices=mats)


def log_odeint(field, logsigs: jnp.ndarray, features_0: jnp.ndarray, return_interval: bool = True) -> jnp.ndarray:
    """
    Integrate Z_{t+1} = Z_t + (M_t @ act(Z_t) + b_t),
    where M_t = Σ_i logsigs[t,i] * A_i,   b_t = Σ_i logsigs[t,i] * b_i.

    Args
    ----
    field.matrices : (d, n, n+1)   # last col is b_i
    field.activation : callable     # (n,) -> (n,)
    logsigs : (B, K, d)             # bin-wise integrals (increments)
    features_0 : (B, n)             # initial features per trajectory
    return_interval : bool          # if True -> (B, K+1, n) full path; else -> (B, n) final only

    Returns
    -------
    jnp.ndarray
        (B, K+1, n) if return_interval=True, else (B, n).
    """
    A_full = field.matrices
    A = A_full[..., :-1]  # (d, n, n)
    b = A_full[..., -1]   # (d, n)
    act = field.activation

    @jax.jit
    def _integrate_one_full(A, b, z0, logs_seq):
        """Single trajectory, returns full path (K+1, n)."""
        # Precompute per-step operators (materializes K blocks)
        Ms = jnp.einsum('kd,dnm->knm', logs_seq, A)  # (K, n, n)
        bs = jnp.einsum('kd,dn->kn',  logs_seq, b)   # (K, n)

        def step(z_t, inputs):
            M_t, b_t = inputs
            z_next = z_t + (M_t @ act(z_t) + b_t)
            return z_next, z_next

        _, traj = jax.lax.scan(step, z0, (Ms, bs))       # (K, n)
        return jnp.concatenate([z0[None, :], traj], 0)   # (K+1, n)

    @jax.jit
    def _integrate_one_final(A, b, z0, logs_seq):
        """Single trajectory, returns only final state (n,). Memory-saving."""
        def step(z_t, x_t):  # x_t: (d,)
            M_t = jnp.einsum('d,dnm->nm', x_t, A)  # (n, n)
            b_t = jnp.einsum('d,dn->n',  x_t, b)   # (n,)
            z_next = z_t + (M_t @ act(z_t) + b_t)
            return z_next, None

        zT, _ = jax.lax.scan(step, z0, logs_seq)
        return zT  # (n,)

    if return_interval:
        batched = jax.jit(jax.vmap(_integrate_one_full, in_axes=(None, None, None, 0)))
        return batched(A, b, features_0, logsigs)   # (B, K+1, n)
    else:
        batched = jax.jit(jax.vmap(_integrate_one_final, in_axes=(None, None, None, 0)))
        return batched(A, b, features_0, logsigs)   # (B, n)

    
class RandomRDE:
    """
    Random-feature RDE:
      - Builds coefficient blocks once from input_dim & order
      - Computes log-signatures per mini-batch (or once if no batching)
      - Integrates over bin-wise logsig increments
    """

    def __init__(self,
                 key: Union[int, jax.Array],
                 n_features: int,
                 order: int = 4,
                 step: int = 8,
                 activation: Callable = lambda x: x,
                 config: dict = None,
                 cache: Optional["Cache"] = None,
                 **kwargs):

        super(RandomRDE, self).__init__()

        self.n_features = n_features
        self.activation = activation
        self.order = order
        self.step = step

        self.stdA = config.get('stdA', 1.0)
        self.stdB = config.get('stdB', 0.0)
        self.std0 = config.get('std0', 1.0)

        self.min_length = kwargs.get('min_length', 3)
        self.normalize_logsigs = kwargs.get('normalize_logsigs', True)
        self.cache_logsigs = kwargs.get('cache_logsigs', True)

        self.cache = cache
        self.key = KeyGen(key)

        # set at build time
        self.fields: Optional[VectorFieldRDE] = None
        self.features_0: Optional[jnp.ndarray] = None

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
        
        if X.ndim == 2:
            return X[None, ...]
        if X.ndim == 3:
            return X
        raise ValueError("Input data must be either 2D or 3D (batch)")


    def _validate_params(self):

        _check_non_negative_value(self.std0, 'std0')
        _check_non_negative_value(self.stdA, 'stdA')
        _check_non_negative_value(self.stdB, 'stdB')

        if isinstance(self.activation, str):
            if self.activation not in ACTIVATION_DICT:
                raise ValueError(f"Activation function '{self.activation}' not recognized.")
            self.activation = ACTIVATION_DICT[self.activation]


    def _validate_cache(self,
                        items : str, 
                        logsigs_dim : int, 
                        logsigs_length : Optional[int] = None, 
                        testing : bool = False) -> None:

        if self.cache is None:
            self.cache = Cache()
            return False

        # Check cached matrices
        if items == 'matrices':

            if not self.cache.has('rde_matrices'):
                return False
            if not self.cache.has('stdA'):
                return False
            if not self.cache.has('stdB'):
                return False
            if not self.cache.has('std0'):
                return False
            if not self.cache.has('features_0'):
                return False
            
            # Check cached matrices dimensions
            matrix_input_dim, matrix_n_features, _ = self.cache.get('rde_matrices').shape
            if matrix_n_features != self.n_features:
                return False
            if matrix_input_dim != logsigs_dim:
                return False

            # Check cached features_0 dimensions
            features_0_dim = self.cache.get('features_0').shape[0]
            if features_0_dim != self.n_features:
                return False
            
            return True

        # Check cached logsigs 
        if items == 'logsigs':
            
            tag = 'logsigs_test' if testing else 'logsigs'
            if not self.cache.has(tag):
                return False
            
            Lc, Dc = self.cache.get(tag).shape[1:]
            return (logsigs_dim == Dc) and (logsigs_length == Lc)
            
        return False
        

    # ----------------------------- Cache update -----------------------------
    def _update_cache(self, logsigs: Optional[jnp.ndarray] = None, testing: bool = False) -> None:
        """
        Updates the cache with the current state of the object.
        """
        self.cache.set('rde_matrices', self.fields.matrices)
        self.cache.set('stdA', self.stdA)
        self.cache.set('stdB', self.stdB)
        self.cache.set('std0', self.std0)
        self.cache.set('features_0', self.features_0)

        if self.cache_logsigs and (logsigs is not None):
            tag = 'logsigs_test' if testing else 'logsigs'
            self.cache.set(tag, logsigs)


    # ----------------------------- Logsigs helpers -----------------------------    
    def _initialize_logsigs(self,
                            X: jnp.ndarray,
                            logsigs_dim: int,
                            logsigs_length: int,
                            use_cache: bool,
                            testing: bool,
                            batch_size: Optional[int] = None) -> jnp.ndarray:

        # Try cache 
        if use_cache and self._validate_cache('logsigs', logsigs_dim, logsigs_length,testing=testing):
            tag = 'logsigs' if not testing else 'logsigs_test'
            return self.cache.get(tag)
        
        # Else compute
        return get_logsignatures(X, self.step, self.order, self.min_length, batch_size=batch_size)


    # ----------------------------- Field + f0 init -----------------------------

    def _initialize_fields(self, input_dim : int, logsigs_dim : int, use_cache : bool) -> None:

        if use_cache and self._validate_cache('matrices', logsigs_dim):  
            
            self.fields = VectorFieldRDE.from_cache(self.key(), 
                                                    self.cache,
                                                    input_dim,
                                                    self.n_features, 
                                                    self.order, 
                                                    self.activation, 
                                                    self.stdA, 
                                                    self.stdB)

            self.features_0 = self.cache.get('features_0')
        

        else:
            self.fields = VectorFieldRDE.from_random(self.key(),
                                                     input_dim,
                                                     self.n_features,
                                                     self.order,
                                                     self.activation,
                                                     self.stdA,
                                                     self.stdB)
            
            # Sample fresh initial features from Normal(0, std0)
            self.features_0 = self.std0 * random.normal(self.key(), 
                                                        shape=(self.n_features,), 
                                                        dtype=jnp.float32)
        


    # ----------------------------- Core feature eval -----------------------------

    def _get_features(self, logsigs_batch: jnp.ndarray, return_interval: bool) -> jnp.ndarray:
        """
        Compute features for a **batch** of logsigs (B, K, d).
        """
        
        features = log_odeint(self.fields, logsigs_batch, self.features_0, return_interval=return_interval)  # (B, K+1, n)

        return features / jnp.sqrt(jnp.asarray(self.n_features, dtype=jnp.float32))


    # ----------------------------- Public API -----------------------------
    def get_features(self,
                     X: jnp.ndarray,
                     batch_size: Optional[int] = None,
                     return_interval: bool = False,
                     testing: bool = False,
                     use_cache: bool = False) -> jnp.ndarray:
        """
        X: (B, T, C)
        returns (B, n) or (B, K+1, n) if return_interval.
        """
        X = self._validate_input(X)
        
        B, T, C = X.shape

        logsigs_dim = get_logsig_dimension(self.order, C)
        logsigs_length = (T // self.step) + int((T % self.step) >= self.min_length)

        # Init field & f0 given input space (=logsigs_dim)
        self._initialize_fields(C, logsigs_dim, use_cache)

        # Obtain logsigs for full X (possibly batched by batch_size)
        logsigs = self._initialize_logsigs(X, logsigs_dim, logsigs_length, use_cache, testing, batch_size)

        # Optionally update cache with matrices/f0 and logsigs
        if use_cache:
            self._update_cache(logsigs=logsigs, testing=testing if self.cache_logsigs else False)

        # If no additional batch slicing is required, evaluate once
        if batch_size is None or B <= batch_size:
            return self._get_features(logsigs, return_interval)

        # Else slice **logsigs** in sync with X's batch chunks
        outs = []
        for i in range(0, B, batch_size):
            outs.append(self._get_features(logsigs[i:i+batch_size], return_interval))
        return jnp.concatenate(outs, axis=0)


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
            X: Array of shape (batchX, timeX, input_dim).
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
                                        testing = True,
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
    
