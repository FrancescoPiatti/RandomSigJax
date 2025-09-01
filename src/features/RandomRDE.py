import jax
import jax.numpy as jnp

from typing import Optional
from typing import Union
from typing import Callable

from dataclasses import dataclass
from functools import partial

from ..utils.cache import Cache
from ..utils.random import KeyGen, scale_matrices_rde, gaussian_matrices_sampler_RDE

from ..configs import DEFAULT_CONFIG_RRDE

@dataclass
class VectorFieldRDE:

    logsig_dim : int
    n_features : int
    logsigs : jnp.ndarray
    order : int
    activation : Callable 
    matrices : jnp.ndarray
    # cache : Cache

    @classmethod
    def from_random(cls,
                    key : jax.Array,
                    n_features : int,
                    logsigs : jnp.ndarray,
                    order : int,
                    activation : Callable,
                    stdA : float = 1.0,
                    stdB : float = 1.0):

        logsig_dim = logsigs.shape[-1]
        matrices = gaussian_matrices_sampler_RDE(key, logsig_dim, n_features, order, stdA, stdB)

        return cls(logsig_dim=logsig_dim,
                   n_features=n_features,
                   activation=activation,
                   logsigs=logsigs,
                   matrices=matrices)

    @classmethod
    def from_cache(cls,
                   key : jax.Array,
                   cache : Cache,
                   n_features : int,
                   logsigs : jnp.ndarray,
                   order : int,
                   activation : Callable,
                   stdA : float = 1.0,
                   stdB : float = 1.0):

        logsig_dim = logsigs.shape[-1]

        matrices = cache.get('matrices')

        matrices = scale_matrices_rde(matrices, 
                                      logsig_dim, 
                                      order,
                                      stdA,
                                      stdB,
                                      cache.get('stdA'),
                                      cache.get('stdB'),
                                      key)
        
        return cls(logsig_dim=logsig_dim,
                   n_features=n_features,
                   activation=activation,
                   logsigs=logsig,
                   matrices=matrices)
    


def log_odeint(field, logsigs, features_0):
    """
    Solve dZ_t = sum_i (A_i @ act(Z_t) + b_i) dX_t^i, with Z_0 = features_0.
    X: (B, T, d), features_0: (B, n)
    field.matrices: (d, n, n+1) with last col = b_i
    """
    A_full = field.matrices           # (d, n, n+1)
    A = A_full[..., :-1]              # (d, n, n)
    b = A_full[..., -1]               # (d, n)
    act = field.activation            # capture in closure (no static arg)

    dt = 1.0 / logsigs.shape[1]

    @jax.jit
    def _integrate_one(A, b, z0, logsigs):
        """
        Single trajectory integrator.
        A: (d,n,n), b: (d,n), z0: (n,), dx_seq: (T-1,d)
        return: (T,n)
        """
        # Precompute per-step operators
        mat_logs = jnp.einsum('td,dnm->tnm', logsigs, A)   # (T-1, n, n)
        bias_logs = jnp.einsum('td,dn->tn',  logsigs, b)    # (T-1, n)

        def step(z_t, inputs):
            M_t, b_t = inputs
            z_next = z_t + (M_t @ act(z_t) + b_t) * dt
            return z_next, z_next

        _, traj = jax.lax.scan(step, z0, (mat_logs, bias_logs))     # traj: (T-1, n)
        return jnp.concatenate([z0[None, :], traj], axis=0)

    # Batch over trajectories (batch axis 0 on z0 and dx)
    integrate_batched = jax.jit(jax.vmap(_integrate_one, in_axes=(None, None, None, 0)))
    features = integrate_batched(A, b, features_0, dx)   # (B, T, n)
    return features

    


class RandomRDE(nn.Module):

    """
    This class defines the
    """

    def __init__(self,  
                 n_features : int,
                 order : int = 4,
                 step : int = 8,
                 activation :  Callable = lambda x : x,
                 config : dict = {'stdA' : 1.0, 'stdB' : 0.0, 'std0' : 1.0},
                 cache : Optional[Cache] = None,
                 device : torch.device = torch.device('cpu'),
                 **kwargs):

        super(RandomRDE, self).__init__()

        self.n_features = n_features
        self.activation = activation
        self.order = order
        self.step = step

        self.stdA = config.get('stdA', 1.0)
        self.stdB = config.get('stdB', 0.0)
        self.std0 = config.get('std0', 1.0)

        self.cache = cache
        self.device = device

        self.method = kwargs.get('method', 'rk4')
        self.adjoint = kwargs.get('adjoint', False)
        self.min_length = kwargs.get('min_length', 3)
        
        self._validate_params()


    def _validate_cache(self, logsigs_dim : int, logsigs_length : int, 
                        key : str, testing : bool = False) -> None:

        if self.cache is None:
            self.cache = Cache()
            return False
        
        if key == 'matrices':

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
            
            matrix_input_dim, matrix_n_features, _ = self.cache.get('rde_matrices').shape

            if matrix_n_features != self.n_features:
                return False
            
            if matrix_input_dim != logsigs_dim:
                return False
            
            features_0_dim = self.cache.get('features_0').shape[0]

            if features_0_dim != self.n_features:
                return False
            
            return True
        
        if key == 'logsigs':

            _str = 'logsigs_test' if testing else 'logsigs'
            
            if not self.cache.has(_str):
                return False
            
            _logsigs_length, _logsigs_dim = self.cache.get(_str).shape[1:]
            
            if logsigs_dim != _logsigs_dim:
                return False
            
            if logsigs_length != _logsigs_length:
                return False
            
            return True
        
    
    def get_logsigs(self, X : Tensor) -> Tensor:

        return get_logsigs(X, 
                           self.step, 
                           self.order, 
                           self.min_length, 
                           batch_size=self.batch_size, 
                           device=self.device)
            

    def _initialize_fields(self, X, use_cache, testing : bool,
                           logsigs : Optional[Tensor] = None) -> None:

        input_dim = X.shape[-1]
        logsigs_dim = get_logsig_dimension(self.order, input_dim)
        logsigs_length = X.shape[1] // self.step 
        logsigs_length += 1 if X.shape[1] % self.step >= self.min_length else 0

        if logsigs is None:
            if use_cache:
                _valid_logsigs = self._validate_cache(logsigs_dim, 
                                                      logsigs_length, 
                                                      key='logsigs', 
                                                      testing=testing)
                if _valid_logsigs:
                    _str = 'logsigs' if not testing else 'logsigs_test'
                    logsigs = self.cache.get(_str)
                else:
                    logsigs = self.get_logsigs(X)
            else:
                logsigs = self.get_logsigs(X)

        self.logsigs = logsigs.to(self.device)
        
        if use_cache:

            _valid_matrices = self._validate_cache(logsigs_dim, 
                                                   logsigs_length, 
                                                   key='matrices')  
            
            _cache = self.cache if _valid_matrices else None
            
            self.fields = VectorFieldRDE(input_dim,
                                         self.n_features, 
                                         self.logsigs, 
                                         self.order, 
                                         self.activation, 
                                         self.stdA, 
                                         self.stdB, 
                                         cache=_cache, 
                                         device=self.device)
            if _valid_matrices:
                self.features_0 = self.cache.get('features_0').to(self.device) 
            else:
                self.features_0 = torch.normal(0.0, self.std0, size=(self.n_features,)).to(self.device)
    
            self._update_cache(testing)
        
        else:

            self.fields = VectorFieldRDE(input_dim, 
                                         self.n_features, 
                                         self.logsigs, 
                                         self.order, 
                                         self.activation, 
                                         self.stdA, 
                                         self.stdB, 
                                         cache=None, 
                                         device=self.device)
            
            self.features_0 = torch.normal(0.0, self.std0, size=(self.n_features,)).to(self.device)
        

    def _validate_input(self, X : Tensor) -> None:
        """
        Validates the input data. Moves it to self device

        Args:
            X : A data array on CPU or GPU.

        Raises:
            ValueError: If the input data is not a Tensor, or if the input data is not 2D or 3D.
        """
        if not isinstance(X, Tensor):
            raise ValueError("Input data must be a torch tensor")
        
        if len(X.shape) == 2:
            return X.to(self.device).unsqueeze(0)
        elif len(X.shape) == 3: 
            return X.to(self.device)
        else:
            raise ValueError("Input data must be either 2D or 3D (batch)")


    def _validate_params(self):

        _check_non_negative_value(self.std0, 'std0')
        _check_non_negative_value(self.stdA, 'stdA')
        _check_non_negative_value(self.stdB, 'stdB')
        _check_boolean(self.adjoint, 'adjoint')

        if isinstance(self.activation, str):
            if self.activation not in ACTIVATION_DICT:
                raise ValueError(f"Activation function '{self.activation}' not recognized.")
            self.activation = ACTIVATION_DICT[self.activation]


    def _update_cache(self, testing : bool = False) -> None:
        """
        Updates the cache with the current state of the object.

        Args:
            cache : The cache to update.
        """
        if testing:
            self.cache.set('logsigs_test', self.logsigs)
        else:
            self.cache.set('logsigs', self.logsigs)

        self.cache.set('rde_matrices', self.fields.rde_matrices)
        self.cache.set('stdA', self.stdA)
        self.cache.set('stdB', self.stdB)
        self.cache.set('std0', self.std0)
        self.cache.set('features_0', self.features_0)


    def _set_options(self, logsigs, return_interval=False):
        """
        Sets the options to be passed to the relevant `odeint` function.

        Args:
            logsig (torch.Tensor): The logsignature of the path.
            return_sequences (bool): Set True if a regression problem where we need the full sequence. This requires us
                specifying the time grid as `torch.arange(0, T_final)` which is less memory efficient that specifying
                the times `t = torch.Tensor([0, T_final])` along with an `step_size=1` in the options.
            eps (float): The epsilon perturbation to make to integration points to distinguish the ends.

        Returns:
            torch.Tensor, dict: The integration times and the options dictionary.
        """
        length = logsigs.shape[1] 
        if return_interval:
            t = torch.arange(0, length, dtype=torch.float).to(self.device)
            options = {}
        else:
            options = {'step_size':1}
            t = torch.Tensor([0, length]).to(self.device)
        return t, options

    
    # CHANGE HERE
    def _get_features(self, 
                      X : Tensor, 
                      return_interval : bool = False):
        
        t, options = self._set_options(self.logsigs, return_interval)

        odeint_fn = odeint_adjoint if self.adjoint else odeint
        features_t = odeint_fn(func=self.fields, 
                               y0=self.features_0, 
                               t=t, 
                               method=self.method, 
                               options=options).transpose(0,1)

        if return_interval:
            return features_t 
        
        return features_t[:, -1]
    

    def get_features(self, 
                     X : Tensor, 
                     batch_size : Optional[int] = None,
                     return_interval : bool = False,
                     testing : bool = False,
                     use_cache : bool = False):
        
        self._validate_input(X)

        # Sizes
        batch = X.shape[0] 

        if batch_size is None:
            self.batch_size = batch
        else:
            self.batch_size = batch_size

        self._initialize_fields(X, use_cache, testing)
        self.logsigs = self.logsigs / self.logsigs.max()

        if self.features_0.dim() == 1:
            self.features_0 = self.features_0.repeat(X.shape[0], 1)

        # Get features
        if batch <= self.batch_size:
            features = self._get_features(X, return_interval)
        else:
            features = []
            for i in range(0, batch, batch_size):
                X_batch = X[i:i+batch_size]
                features_batch = self._get_features(X_batch, return_interval)
                features.append(features_batch)

            features = torch.cat(features, dim=0)

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