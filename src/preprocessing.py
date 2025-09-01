import jax
import jax.numpy as jnp
import jax.image as jimg

from typing import Optional
from functools import partial


class PreprocessorJAX:
    """
    Initializes the `SequenceAugmentor` object.

    Args:
        add_time: Whether to augment with time coordinate.
        lead_lag: Whether to augment with lead-lag.
        basepoint: Whether to augment with basepoint.
        normalize: Whether to normalize time series.
        max_time: Maximum time if `add_time is True`.
        max_len: Maximum length of sequences.
    """
    def __init__(
        self,
        add_time: bool = True,
        lead_lag: bool = True,
        basepoint: bool = True,
        normalize: bool = True,
        max_time: float = 1.0,
        max_len: Optional[int] = None,
    ):
        
        self.add_time = add_time
        self.lead_lag = lead_lag
        self.basepoint = basepoint
        self.normalize = normalize
        self.max_time = float(max_time)
        self.max_len = max_len
        self.scale_ = None  


    def fit(self, X_seq: jnp.ndarray):
        
        if self.normalize:
            self.scale_ = jnp.max(X_seq)


    def transform(self, X_seq: jnp.ndarray) -> jnp.ndarray:
        """
        Apply augmentation pipeline to X_seq (batch, T, C).
        """
        if self.normalize and self.scale_ is None:
            # lightweight fallback if fit() was skipped
            self.scale_ = jnp.max(X_seq)

        return _transform_core(
            X_seq,
            scale=self.scale_ if self.normalize else jnp.array(1.0, X_seq.dtype),
            max_time=self.max_time,
            add_time=self.add_time,
            lead_lag=self.lead_lag,
            basepoint=self.basepoint,
            normalize=self.normalize,
            max_len=self.max_len,
        )


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _transform_core(
    X_seq: jnp.ndarray,
    scale: jnp.ndarray,
    max_time: float,
    add_time: bool,
    lead_lag: bool,
    basepoint: bool,
    normalize: bool,
    max_len: Optional[int],
) -> jnp.ndarray:
    """
    JITted numeric core implementing the pipeline.
    """
    B, T, C = X_seq.shape
    x = X_seq

    # Normalize
    if normalize:
        x = x / scale

    # Lead-lag: repeat time steps then concat lead/lag along channels
    if lead_lag:
        x_rep = jnp.repeat(x, 2, axis=1)          # (B, 2T, C)
        lead = x_rep[:, 1:, :]                    # (B, 2T-1, C)
        lag  = x_rep[:, :-1, :]                   # (B, 2T-1, C)
        x = jnp.concatenate([lead, lag], axis=-1) # (B, 2T-1, 2C)

    # Add time channel
    if add_time and max_time > 1e-6:
        Tb = x.shape[1]
        t = jnp.linspace(0.0, float(max_time), Tb, dtype=x.dtype)  # (Tb,)
        t = jnp.broadcast_to(t[None, :, None], (x.shape[0], Tb, 1))
        x = jnp.concatenate([t, x], axis=-1)       # (B, Tb, 1+...)

    # Basepoint: prepend zero vector along time axis
    if basepoint:
        zero = jnp.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype)
        x = jnp.concatenate([zero, x], axis=1)

    # Resize along time if needed (linear interpolation)
    if (max_len is not None) and (x.shape[1] > max_len):
        x = jimg.resize(x, shape=(x.shape[0], int(max_len), x.shape[2]), method="linear", antialias=False)

    return x