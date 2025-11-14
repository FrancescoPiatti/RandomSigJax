# src/utils/synthetic_hurst.py

import jax.numpy as jnp
import numpy as np

from typing import Sequence, Tuple


def _fbm_covariance_matrix(length: int, H: float) -> np.ndarray:
    """
    Build the covariance matrix of fractional Brownian motion B^H on a uniform grid
    t_k = k / (length - 1), k = 0, ..., length - 1.
    """
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    t_col = t[:, None]
    t_row = t[None, :]

    # Cov(B^H_t, B^H_s) = 0.5 (t^{2H} + s^{2H} - |t - s|^{2H})
    cov = 0.5 * (
        np.power(t_col, 2.0 * H)
        + np.power(t_row, 2.0 * H)
        - np.power(np.abs(t_col - t_row), 2.0 * H)
    )
    return cov


def generate_hurst_classification_dataset(
    n_per_bin: int = 500,
    length: int = 256,
    d_input: int = 1,
    hurst_grid: Sequence[float] = tuple(0.05 + 0.10 * k for k in range(9)),  # 0.05, 0.15, ..., 0.85
    bin_edges: Sequence[float] = tuple(0.1 * k for k in range(11)),          # [0,0.1), [0.1,0.2), ...
    noise_std: float = 0.0,
    to_jax: bool = True,
    seed: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a synthetic dataset of d-dimensional paths whose roughness is
    controlled by a Hurst exponent, with labels given by Hurst bins.

    For each H in `hurst_grid` we simulate n_per_bin independent sample paths
    of a d-dimensional fractional Brownian motion, where each channel is an
    independent scalar fBM with the same H.

    Each sample is labelled according to which interval [a, b) of `bin_edges`
    its H lies in. For example, with hurst_grid = (0.05, 0.15, 0.25, ...)
    and bin_edges = (0.0, 0.1, 0.2, ...), H = 0.05 -> class 0 (bin [0.0,0.1)),
    H = 0.15 -> class 1 (bin [0.1,0.2)), etc.

    Parameters
    ----------
    n_per_bin : int
        Number of paths to generate per Hurst grid point.
    length : int
        Number of timepoints per path.
    d_input : int
        Input dimension (number of channels). For d_input > 1, we generate
        d_input independent fBM components with the same H.
    hurst_grid : Sequence[float]
        Hurst exponents at which to simulate fractional Brownian motion.
    bin_edges : Sequence[float]
        Edges of the Hurst bins. Must be increasing. Class k corresponds to
        [bin_edges[k], bin_edges[k+1]).
    noise_std : float
        Standard deviation of additive Gaussian noise per timepoint/channel.
    to_jax : bool
        Whether to return JAX arrays.
    seed : int
        Numpy random seed.

    Returns
    -------
    X : jnp.ndarray or np.ndarray
        Array of paths with shape (n_samples, length, d_input).
    y : jnp.ndarray or np.ndarray
        Integer labels giving the Hurst bin index, shape (n_samples,).

    Notes
    -----
    - All channels of a given sample share the same H (same roughness class),
      but are independent across dimensions.
    - For classification, you can feed X directly into your RandomCDE / RDE
      pipeline as a standard multivariate time series.
    """
    rng = np.random.default_rng(seed)

    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    assert np.all(np.diff(bin_edges) > 0), "bin_edges must be strictly increasing"

    hurst_grid = np.asarray(hurst_grid, dtype=np.float64)

    # Map each H in hurst_grid to a bin index
    # For each H, find k s.t. bin_edges[k] <= H < bin_edges[k+1]
    bin_indices = np.digitize(hurst_grid, bin_edges) - 1
    if np.any(bin_indices < 0) or np.any(bin_indices >= len(bin_edges) - 1):
        raise ValueError("Some H values in hurst_grid lie outside the bin_edges range.")

    all_paths = []
    all_labels = []

    for H, bin_idx in zip(hurst_grid, bin_indices):
        # Precompute Cholesky factor for this H (time covariance)
        cov = _fbm_covariance_matrix(length, float(H))
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(length, dtype=np.float64))  # (length, length)

        # We want d-dimensional fBM: simulate d_input independent copies.
        # Sample standard normal increments: (n_per_bin * d_input, length)
        z = rng.standard_normal(size=(n_per_bin * d_input, length), dtype=np.float64)

        # Apply Cholesky: each row is one path (scalar fBM)
        paths_1d = z @ L.T  # (n_per_bin * d_input, length)

        # Reshape to (n_per_bin, d_input, length)
        paths_1d = paths_1d.reshape(n_per_bin, d_input, length)

        # Permute to (n_per_bin, length, d_input)
        paths = np.transpose(paths_1d, (0, 2, 1))

        if noise_std > 0.0:
            paths += noise_std * rng.standard_normal(size=paths.shape)

        all_paths.append(paths.astype(np.float32))
        all_labels.append(np.full((n_per_bin,), bin_idx, dtype=np.int64))

    X_np = np.concatenate(all_paths, axis=0)  # (n_samples, length, d_input)
    y_np = np.concatenate(all_labels, axis=0)  # (n_samples,)

    if to_jax:
        X = jnp.array(X_np, dtype=jnp.float32)
        y = jnp.array(y_np, dtype=jnp.int32)
    else:
        X = X_np
        y = y_np

    return X, y