import jax.numpy as jnp
import numpy as np

from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder

from typing import Tuple


def load_dataset(name : str, to_jax : bool = True) -> Tuple[jnp.ndarray]:
    """
    Load a dataset from UEA/UEA repository.
    
    Parameters
    ----------
    name : str
        Name of the dataset to load.
    to_jax : bool
        Whether to convert the data to JAX arrays.
    
    Returns
    -------
    X : jnp.ndarray
        Features of the dataset. Shape: (n_samples, n_timepoints, n_channels)
    y : jnp.ndarray
        Labels of the dataset.
    """
    X_train, y_train = load_classification(name, split='train')
    X_test, y_test = load_classification(name, split='test')

    X_train = np.swapaxes(X_train, 1, 2)  # Ensure shape is (n_samples, n_timepoints, n_channels)
    X_test = np.swapaxes(X_test, 1, 2)    # Ensure shape is (n_samples, n_timepoints, n_channels)
    
    # Encode labels if they are not numeric
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    if to_jax:
        X_train = jnp.array(X_train, dtype=jnp.float32)
        y_train = jnp.array(y_train, dtype=jnp.int32)
        X_test = jnp.array(X_test, dtype=jnp.float32)
        y_test = jnp.array(y_test, dtype=jnp.int32)

    return X_train, y_train, X_test, y_test