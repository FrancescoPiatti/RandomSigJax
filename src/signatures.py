import jax
import jax.numpy as jnp
from functools import partial

from signax import logsignature


def split_indices(L: int, N: int, min_length: int):
    """
    Helper function that splits a time axis into contiguous index ranges.

    Produces index pairs (start, end) of length `chunk_size`, merging the final
    segment into the previous one if it would be shorter than `min_length`.

    Parameters
    ----------
    length : int
        Total number of time steps.
    chunk_size : int
        Nominal segment length.
    min_length : int
        Minimum length allowed for the final segment.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) index intervals covering [0, length).
    """
    chunks = []
    start = 0
    while start < L:
        leftover = L - start
        if leftover <= N:
            if leftover < min_length and len(chunks) > 0:
                # Merge with the previous chunk
                prev_start, _ = chunks.pop()
                chunks.append((prev_start, L))
            else:
                chunks.append((start, L))
            break
        else:
            chunks.append((start, start + N))
            start += N
    return chunks

@partial(jax.jit, static_argnames=("length",))
def _logsig_chunks(X: jnp.ndarray,
                   starts: jnp.ndarray,   # (n_chunks,)
                   lens: jnp.ndarray,     # (n_chunks,)
                   level: int) -> jnp.ndarray:
    """
    JIT core: vectorize logsignature over time-chunks.
    Returns (batch, n_chunks, logsig_dim).
    """
    def chunk_fn(start, L):
        # X: (batch, time, channels); slice along time
        x_chunk = jax.lax.dynamic_slice_in_dim(X, start_index=start, slice_size=L, axis=1)  # (B, L, C)
        return logsignature(x_chunk, level)  # (B, D)

    # vmap over chunks; place mapped axis at position 1 -> (B, n_chunks, D)
    return jax.vmap(chunk_fn, in_axes=(0, 0), out_axes=1)(starts, lens)


def get_logsignature(X: jnp.ndarray,
                     step: int,
                     order: int,
                     min_length: int) -> jnp.ndarray:
    """
    Compute per-chunk log-signatures along time.
    X: (batch, time, channels)
    Returns: (batch, n_chunks, logsig_dim)
    """
    # ensure final chunk length ≥ min_length, but never exceed 'step' rule
    if step < min_length:
        min_length = step

    # build chunk indices on the host (Python)
    idx = split_indices(int(X.shape[1]), step, min_length)  # [(s,e), ...]
    starts = jnp.asarray([s for s, e in idx], dtype=jnp.int32)
    lens   = jnp.asarray([e - s for s, e in idx], dtype=jnp.int32)

    return _logsig_chunks(X, starts, lens, order)

# @partial(jax.jit, static_argnums=(1, 2, 3))
# def get_logsignature(X: jnp.ndarray, step: int, length: int, min_length: int):
#     """
#     Compute per-chunk log-signatures along the time dimension.

#     Parameters
#     ----------
#     X : jnp.ndarray
#         Input array of shape (batch, time, channels).
#     step : int
#         Chunk length.
#     length : int
#         Log-signature depth.
#     min_length : int
#         Minimum length for the final chunk.

#     Returns
#     -------
#     jnp.ndarray
#         Array of shape (batch, n_chunks, logsig_dim).
#     """
#     if step < min_length:
#         min_length = step

#     idx = split_indices(X.shape[1], step, min_length)
#     logsigs = [
#         jnp.expand_dims(logsignature(X[:, start:end], length), 1)
#         for (start, end) in idx
#     ]

#     return jnp.concatenate(logsigs, axis=1)
