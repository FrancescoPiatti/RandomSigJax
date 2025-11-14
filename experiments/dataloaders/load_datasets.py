# src/utils/load_ucr_uea_datasets.py

import jax.numpy as jnp
import numpy as np
import pandas as pd

from aeon.datasets import load_forecasting
from aeon.datasets import load_classification

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from typing import Tuple
from typing import Optional


def load_ucr_uea_dataset(name : str, to_jax : bool = True) -> Tuple[jnp.ndarray]:
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

    # # Shuffle the data
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

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


def load_etth1(
    csv_path: Optional[str] = None,
    to_jax: bool = True,
) -> Tuple[jnp.ndarray, np.ndarray]:
    """
    Load the ETTh1 dataset as a single multivariate time series.

    The canonical CSV can be obtained from the ETT repository
    (Zhou et al., 2021). It has a 'date' column plus several
    covariates (e.g. 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT').

    Parameters
    ----------
    csv_path : Optional[str]
        Path (or URL) to 'ETTh1.csv'. If None, a default GitHub URL is used.
    to_jax : bool
        Whether to convert the output to JAX arrays.

    Returns
    -------
    X : jnp.ndarray
        Multivariate time series of shape (n_timepoints, n_channels),
        containing all numeric columns except the timestamp.
    time_index : np.ndarray
        Array of timestamps corresponding to each row (dtype='datetime64').
    """
    if csv_path is None:
        csv_path = (
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
            "ETT-small/ETTh1.csv"
        )

    df = pd.read_csv(csv_path)
    # Assume there is a 'date' column
    if "date" in df.columns:
        time_index = pd.to_datetime(df["date"].values)
        df_numeric = df.drop(columns=["date"])
    else:
        time_index = np.arange(len(df))
        df_numeric = df

    X_np = df_numeric.to_numpy(dtype=np.float32)  # (n_timepoints, n_channels)

    if to_jax:
        return jnp.array(X_np, dtype=jnp.float32), time_index
    else:
        return X_np, time_index


def load_weather_monash(
    to_jax: bool = True,
    return_metadata: bool = False,
):
    """
    Load the Monash 'weather' dataset via aeon, and reshape it into
    a UCR-like 3D array.

    The Monash 'weather' dataset contains 3010 daily time series of four
    weather variables across Australian stations.

    Parameters
    ----------
    to_jax : bool
        Whether to convert the output to JAX arrays.
    return_metadata : bool
        If True, also return the metadata dictionary from aeon.

    Returns
    -------
    X : jnp.ndarray
        Array of shape (n_series, n_timepoints, 1), where each series
        is a univariate time series (Monash already encodes multivariate
        structure at the metadata level; here we treat each series as 1D).
    time_index : np.ndarray
        Array of time stamps shared by all series.
    metadata : dict, optional
        Aeon metadata, returned only if `return_metadata=True`.
    """
    # aeon >= 0.8.0: load_forecasting(name, return_metadata=True)
    X_df, metadata = load_forecasting("weather_dataset", return_metadata=True)

    # X_df: index = time, columns = series_id
    time_index = X_df.index.to_numpy()
    data_np = X_df.to_numpy(dtype=np.float32)  # (n_timepoints, n_series)

    # Reshape to (n_series, n_timepoints, 1)
    data_np = np.transpose(data_np, (1, 0))  # (n_series, n_timepoints)
    data_np = data_np[..., None]             # (n_series, n_timepoints, 1)

    if to_jax:
        X = jnp.array(data_np, dtype=jnp.float32)
    else:
        X = data_np

    if return_metadata:
        return X, time_index, metadata
    else:
        return X, time_index
    


def load_ili(
    csv_path: str,
    date_col: str = "date",
    value_col: str = "ILI",
    to_jax: bool = True,
) -> Tuple[jnp.ndarray, np.ndarray]:
    """
    Load an ILI (influenza-like illness) dataset from a CSV file as a single
    univariate time series.

    This is intentionally simple and assumes a CSV with at least:
    - a date column (e.g. 'date' or 'week')
    - a numeric column with the ILI measure (e.g. 'ILI' or 'value')

    Parameters
    ----------
    csv_path : str
        Path or URL to the ILI CSV file.
    date_col : str
        Name of the time index column.
    value_col : str
        Name of the numeric ILI column to use as the target.
    to_jax : bool
        Whether to convert the output to JAX arrays.

    Returns
    -------
    X : jnp.ndarray
        Univariate time series of shape (n_timepoints, 1).
    time_index : np.ndarray
        Array of timestamps corresponding to each row.
    """
    df = pd.read_csv(csv_path)

    if date_col in df.columns:
        time_index = pd.to_datetime(df[date_col].values)
        series = df[value_col].astype(np.float32).to_numpy()
    else:
        time_index = np.arange(len(df))
        series = df[value_col].astype(np.float32).to_numpy()

    X_np = series[:, None]  # (n_timepoints, 1)

    if to_jax:
        X = jnp.array(X_np, dtype=jnp.float32)
    else:
        X = X_np

    return X, time_index