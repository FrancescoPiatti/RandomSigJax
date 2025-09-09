import jax.numpy as jnp
import jax.random as random

# CHECK VALUE OF EPS
# WRITE DOCSTRINGS WELL AND FORMAT

def suggest_bandwidth(
    X : jnp.ndarray,
    num_samples : int = 1000,
    is_static : bool = True,
    eps : float = 1e-8,
    ) -> float:
    """
    Suggest bandwidth according to the median heuristic.

    Args:
        X: Training data, shape (N, D) if is_static=True, else (B, T, D).
        num_samples: Number of samples to approximate the distance matrix.
        is_static: Whether X is static (N, D) or sequence (B, T, D).
        eps: Small value for numerical stability in sqrt.

    Returns:
        (median_distance, new_key)
    """

    # Flatten to (sample_size, D)
    if is_static:
        sample_size = X.shape[0]
        samples_view = X.reshape(sample_size, -1)
    else:
        sample_size = X.shape[0] * X.shape[1]
        samples_view = X.reshape(sample_size, X.shape[-1])

    # Bound num_samples
    num_samples = int(min(num_samples, sample_size))

    # Edge cases
    if num_samples < 2:
        return 0.0

    # Sample indices without replacement
    perm = random.permutation(random.PRNGKey(42), sample_size)[:num_samples]
    samples = samples_view[perm, :]  

    # Pairwise squared distances: ||x||^2/2 + ||y||^2/2 - x·y
    norms = (jnp.sum(samples * samples, axis=1, keepdims=True)) / 2.0  
    sq_dists = norms + norms.T - samples @ samples.T  

    # Remove diagonal (self-distances)
    mask = ~jnp.eye(num_samples, dtype=bool)
    dists = jnp.sqrt(jnp.clip(sq_dists[mask], a_min=eps))
    med_dist = jnp.median(dists)

    return float(med_dist)

def suggest_stepsize(length : int):

    step_size_ratio = 0.2 
    prev_step_size = length
    step_size_list = []

    while True:
        step_size = int(length * step_size_ratio)

        if step_size != prev_step_size and step_size >= 3:
            step_size_list.append(step_size)
            step_size_ratio /= 2
            prev_step_size = step_size
        else:
            break

    return step_size_list   
