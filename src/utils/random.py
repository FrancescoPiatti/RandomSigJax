import jax
import jax.numpy as jnp
import jax.random as random
import math

from functools import partial

from dataclasses import dataclass
from typing import Union

from .lie_algebra import get_lie_matrices
from .lie_algebra import get_lyndon_words

@dataclass
class KeyGen:
    """
    PRNG key manager for JAX (to be used OUTSIDE jit).
    """

    def __init__(self, seed_or_key: Union[int, jax.Array] = 0):
        self._key = self._make_key(seed_or_key)

    @staticmethod
    def _is_valid_key(x: jax.Array) -> bool:
        return isinstance(x, jax.Array) and x.dtype == jnp.uint32 and x.shape == (2,)
    
    @classmethod
    def _make_key(cls, seed_or_key: Union[int, jax.Array]) -> jax.Array:
        """
        Return a valid PRNG key from a seed or existing key.
        """
        if isinstance(seed_or_key, int):
            return random.PRNGKey(seed_or_key)
        if cls._is_valid_key(seed_or_key):
            return seed_or_key
        raise TypeError(
            f"Expected int seed or PRNG key of dtype uint32 and shape (2,), "
            f"got dtype={getattr(seed_or_key, 'dtype', None)}, shape={getattr(seed_or_key, 'shape', None)}."
        )
        
    @property
    def current(self) -> jax.Array:
        """
        Peek at the current internal key (no advancement).
        """
        return self._key

    def __call__(self) -> jax.Array:
        """
        Return a fresh subkey and advance the internal key.
        """
        self._key, sub = random.split(self._key)
        return sub

    def split(self, n: int) -> jax.Array:
        """Return `n` fresh subkeys and advance the internal key once."""
        # split into n+1: first is the new carry key, rest are subkeys
        keys = random.split(self._key, n + 1)
        self._key = keys[0]
        return keys[1:]


    def reset(self, seed_or_key: Union[int, jax.Array]) -> None:
        """
        Reset internal key from an int seed or an existing valid key.
        """
        self._key = self._make_key(seed_or_key)


# =============================================================================
# Randemacher matrix sampler
# =============================================================================

@partial(jax.jit, static_argnames=('M'))
def rademacher_sample(key: jax.Array, M: int) -> jnp.ndarray:
    """
    Sample a MxM Rademacher diagonal matrix (±1 on the diagonal).
    Returns (matrix, new_key).
    """
    diag = 2 * random.bernoulli(key, p=0.5, shape=(M,)).astype(jnp.int32) - 1
    mat = jnp.diag(diag.astype(jnp.float32))
    return mat


@partial(jax.jit, static_argnames=('M', 'd'))
def rademacher_matrices_sampler(key: jax.Array, M: int, d: int) -> jnp.ndarray:
    """
    Sample (d, M, M) Rademacher diagonal matrices.
    Returns (stack, new_key).
    """
    diags = 2 * random.bernoulli(key, p=0.5, shape=(d, M)).astype(jnp.int32) - 1  # (d, M)
    mats = jax.vmap(jnp.diag)(diags.astype(jnp.float32))  # (d, M, M)
    return mats


# =============================================================================
# Uniform vector sampler
# =============================================================================

@partial(jax.jit, static_argnames=('N'))
def uniform_vector_sampler(key: jax.Array, N: int) -> jnp.ndarray:
    """
    Sample (d, N, 1) uniform column vectors with entries ~ U(0, 1).
    Returns (stack, new_key).
    """
    return random.uniform(key, shape=(N,), dtype=jnp.float32)

# =============================================================================
# Gaussian matrix sampler
# =============================================================================

@partial(jax.jit, static_argnames=('N', 'M', 'std'))
def gaussian_matrix_sampler(key: jax.Array, N: int, M: int, std: float) -> jnp.ndarray:
    """
    Sample an (N, M) Gaussian matrix with entries ~ N(0, (std/sqrt(N))^2).
    Returns (matrix, new_key).
    """
    scale = std / math.sqrt(N)
    mat = random.normal(key, shape=(N, M), dtype=jnp.float32) * scale
    return mat


@partial(jax.jit, static_argnames=('N', 'd', 'std'))
def gaussian_matrices_sampler(key: jax.Array, N: int, d: int, std: float) -> jnp.ndarray:
    """
    Sample (d, N, N) Gaussian matrices with entries ~ N(0, (std/sqrt(N))^2).
    Returns (stack, new_key).
    """  
    scale = std / math.sqrt(N)
    mats = random.normal(key, shape=(d, N, N), dtype=jnp.float32) * scale
    return mats


@partial(jax.jit, static_argnames=('N', 'd', 'std'))
def gaussian_vectors_sampler(key: jax.Array, N: int, d: int, std: float) -> jnp.ndarray:
    """
    Sample (d, N, 1) Gaussian column vectors with entries ~ N(0, std^2).
    Returns (stack, new_key).
    """
    vecs = random.normal(key, shape=(d, N, 1), dtype=jnp.float32) * std
    return vecs

def gaussian_matrices_sampler_CDE(key : jax.Array,
                                  N : int, 
                                  d : int, 
                                  stdA : float = 1.0, 
                                  stdB : float = 1.0, 
                                  ) -> jnp.ndarray:
    """
    Returns a (d, N, N+1) gaussian matrices.
    """

    key1, key2 = random.split(key, 2)
    A = gaussian_matrices_sampler(key1, N, d, stdA)
    b = gaussian_vectors_sampler(key2, N, d, stdB)
    return jnp.concatenate([A, b], axis=-1)


def gaussian_matrices_sampler_RDE(key : jax.Array,
                                  N : int, 
                                  d : int, 
                                  order : int, 
                                  stdA : float = 1.0,
                                  stdB : float = 1.0,
                                  ) -> jnp.ndarray:
    """
    Returns a (logsig_dim, N, N+1) gaussian matrices.
    """

    key1, key2 = random.split(key, 2)
    matrices = gaussian_matrices_sampler(key1, N, d, stdA)

    # Get the Lie algebra matrices
    lie_matrices = get_lie_matrices(matrices, order)
    
    # Get the bias - dimension is the dim of the logsig
    bias = gaussian_vectors_sampler(key2, N, lie_matrices.shape[0], stdB)

    return jnp.concatenate([lie_matrices, bias], axis=-1)

# =============================================================================
# Gaussian matrix scaler
# =============================================================================

@partial(jax.jit, static_argnames=())
def scale_matrices(
    matrices: jnp.ndarray,
    std_new: float,
    std_old: float,
    key: jax.Array,
) -> jnp.ndarray:
    """
    Rescale entries to achieve target std_new given original std_old.

    If std_old == 0:
      - std_new == 0: return matrices unchanged
      - std_new != 0: return fresh N(0, std_new^2) noise with same shape
    """
    std_new_a = jnp.asarray(std_new, dtype=matrices.dtype)
    std_old_a = jnp.asarray(std_old, dtype=matrices.dtype)

    def when_old_zero(_):
        def when_new_zero(__):
            return matrices
        def when_new_nonzero(__):
            return random.normal(key, matrices.shape, dtype=matrices.dtype) * std_new_a
        return jax.lax.cond(std_new_a == 0.0, when_new_zero, when_new_nonzero, operand=None)

    def when_old_nonzero(_):
        return matrices * (std_new_a / std_old_a)

    return jax.lax.cond(std_old_a == 0.0, when_old_zero, when_old_nonzero, operand=None)


@partial(jax.jit, static_argnames=())
def scale_matrices_cde(
    matrices: jnp.ndarray,
    stdA_new: float,
    stdB_new: float,
    stdA_old: float,
    stdB_old: float,
    key: jax.Array,
) -> jnp.ndarray:
    """
    Scale CDE coefficient blocks of shape (..., m, m) or (..., m, m+1).
    Assumes input is valid; shape validation should happen outside JIT.
    """
    n_rows, n_cols = matrices.shape[-2], matrices.shape[-1]

    def case_square(_):
        # Only A-type scaling
        return scale_matrices(matrices, stdA_new, stdA_old, key)

    def case_augmented(_):
        # [A | b]
        A = matrices[..., :-1]                 # (..., m, m)
        b = matrices[..., -1]                  # (..., m)
        A_scaled = scale_matrices(A, stdA_new, stdA_old, key)
        b_scaled = scale_matrices(b, stdB_new, stdB_old, key)
        b_scaled = jnp.expand_dims(b_scaled, axis=-1)
        return jnp.concatenate([A_scaled, b_scaled], axis=-1)

    return jax.lax.switch(
        jnp.where(n_cols == n_rows, 0, 1),  # 0: square, 1: augmented
        (case_square, case_augmented),
        operand=None,
    )


def scale_matrices_rde(matrices: jax.Array,
                       input_dim: int,
                       order: int,
                       stdA_new: float,
                       stdB_new: float,
                       stdA_old: float,
                       stdB_old: float,
                       key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Scale coefficient matrices for Rough Differential Equation (RDE) models.

    Args:
        matrices (jnp.ndarray): Array of shape (W, batch, m+1), where W = number of Lyndon words.
        input_dim (int): The dimension 'm' of the underlying signal.
        order (int): Truncation level (maximum word length) for the Lyndon basis.
        stdA_new (float): Target standard deviation for A-type coefficients.
        stdB_new (float): Target standard deviation for B-type coefficients.
        stdA_old (float): Original standard deviation for A-type coefficients.
        stdB_old (float): Original standard deviation for B-type coefficients.
        key (jax.Array): Random key for sampling.

    Returns:
        jnp.ndarray: Rescaled tensor of the same shape as input.

    Raises:
        AssertionError: If `matrices` does not have shape [..., m, m+1].
    """
    
    n_rows, n_cols = matrices.shape[-2], matrices.shape[-1]
    
    assert n_cols == n_rows + 1, f"Invalid matrix shape {matrices.shape}. Expected [..., m, m+1]."

    # Handle degenerate cases up front (not jitted; cheap control)
    if stdA_old == 0.0 and stdA_new == 0.0:
        # Only need to scale the bias potentially; A is zero and stays zero.
        b_scaled = scale_matrices(matrices[:, :, -1], stdB_new, stdB_old, key)
        return matrices.at[:, :, -1].set(b_scaled)

    if stdA_old == 0.0 and stdA_new != 0.0:
        return gaussian_matrices_sampler_RDE(n_rows, input_dim, order, stdA_new, stdB_new, key)

    # First scale the bias
    b_scaled = scale_matrices(matrices[:, :, -1], stdB_new, stdB_old, key)
    out = matrices.at[:, :, -1].set(b_scaled)

    # Lyndon block sizes per length 1..L
    words = get_lyndon_words(order, input_dim)                
    lengths = jnp.asarray([len(level) for level in words], jnp.int32) 

    ratio = jnp.asarray(stdA_new / stdA_old, dtype=out.dtype)

    # Rescale each block of rows corresponding to words of increasing length
    ratio = jnp.asarray(stdA_new / stdA_old, dtype=out.dtype)

    @jax.jit
    def apply_blocks(out_in: jax.Array, lengths: jax.Array, ratio: jax.Array) -> jax.Array:
        # carry = (array, current_row_start)
        def body(carry, i):
            arr, row = carry
            blk = lengths[i]  # JAX scalar
            def do_blk(c):
                arr2, r = c
                sf = ratio ** jnp.asarray(i + 1, arr2.dtype)            # (stdA_new/stdA_old)^(i+1)
                arr2 = arr2.at[r : r + blk, :, :-1].multiply(sf)        # scale A part only 
                return (arr2, r + blk)
            def skip(c):
                return c
            return jax.lax.cond(blk > 0, do_blk, skip, (arr, row)), None

        (out_final, _), _ = jax.lax.scan(body, (out_in, 0), jnp.arange(lengths.shape[0]))
        return out_final

    out = apply_blocks(out, lengths, ratio)
    return out