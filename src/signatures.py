import jax
import jax.numpy as jnp
from jax import lax

from functools import partial

from signax import logsignature

# Should change here that you do not get signatures but piecewise linear abelian etc

# -------- fixed-size jitted kernel (slice_size must be static) --------
@partial(jax.jit, static_argnames=('level', 'chunk_size'))
def _logsig_chunks_fixed(X: jnp.ndarray,
                         starts: jnp.ndarray,   # (n_chunks,) int32
                         level: int,
                         chunk_size: int) -> jnp.ndarray:
    """
    Compute logsignatures for chunks of FIXED length chunk_size.
    Returns: (B, n_chunks, D)
    """
    B, T, C = X.shape

    def one_chunk(start):
        x_chunk = lax.dynamic_slice(
            X,
            start_indices=(0, start, 0),               # slice along time
            slice_sizes=(B, chunk_size, C),            # STATIC sizes
        )  # (B, chunk_size, C)
        return logsignature(x_chunk, level)            # (B, D)

    return jax.vmap(one_chunk, in_axes=0, out_axes=1)(starts)


# -------- arithmetic split: only last chunk may differ --------
def _split_last(L: int, step: int, min_len: int):
    """
    Returns:
      n_full  : number of full 'step' chunks (possibly reduced if merge)
      last    : (start, length) or None
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    if min_len < 1:
        min_len = 1

    n_full = L // step
    leftover = L - n_full * step

    if leftover == 0:
        # exactly full chunks
        return n_full, None

    if leftover < min_len and n_full > 0:
        # merge tail into previous full chunk
        n_full -= 1
        last_start = n_full * step
        last_len = step + leftover
        return n_full, (last_start, last_len)

    # keep tail as its own (short) chunk
    last_start = n_full * step
    last_len = leftover
    return n_full, (last_start, last_len)



# ---------- public API with optional batch_size ----------
def get_logsignatures(
    X: jnp.ndarray,         # (B, T, C)
    step: int,
    order: int,
    min_length: int,
    batch_size: int | None = None,
) -> jnp.ndarray:
    """
    Compute per-chunk log-signatures along time.
    All chunks have length==step, except possibly the last one.
    Returns: (B, n_chunks, D)

    If batch_size is provided, processes the batch axis in mini-batches
    to reduce peak memory without changing results.
    """
    if step < min_length:
        min_length = step

    B, T, C = map(int, X.shape)
    n_full, last = _split_last(T, int(step), int(min_length))

    # Precompute start indices for full/last chunks
    starts_full = jnp.arange(n_full, dtype=jnp.int32) * jnp.int32(step) if n_full > 0 else None
    if last is not None:
        last_start, last_len = last
        starts_last = jnp.array([last_start], dtype=jnp.int32)
    else:
        last_len = None
        starts_last = None

    # Helper to process one batch slice
    def run_one_batch(Xb: jnp.ndarray) -> jnp.ndarray:
        outs = []
        if n_full > 0:
            outs.append(_logsig_chunks_fixed(Xb, starts_full, order, step))      # (b, n_full, D)
        if last_len is not None and last_len > 0:
            outs.append(_logsig_chunks_fixed(Xb, starts_last, order, last_len))  # (b, 1, D)
        return outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=1)

    # No batching needed
    if batch_size is None or B <= batch_size:
        return run_one_batch(X)

    # Mini-batch over the batch axis
    chunks = []
    for i in range(0, B, batch_size):
        Xb = X[i : i + batch_size]
        chunks.append(run_one_batch(Xb))
    return jnp.concatenate(chunks, axis=0)