# from src.svc.LinearSVC import LinearSVC
from src.features.RandomCDE import RandomCDE
from src.features.RandomRDE import RandomRDE
from src.features.RandomFourierFeatures import RandomFourierFeatures
import argparse
from utils.preprocessing import Preprocessor
from experiments.dataloaders.load_datasets import load_dataset

import jax
import jax.numpy as jnp


# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC   

from src.svc.LinearSVC import LinearSVC
from src.svc.KernelSVC import KernelSVC 

# n_fourier,  bandwidth,  n_features, std0,stdA,stdB,  activation,   C,    tol,        train_score, val_score
# 128,        1.5,        128,        0.5, 1.5, 0.0,   id,          1.0,   0.001,       0.976,       0.6778




def main(X_train, X_test, y_train, y_test):

    config_cde = {
        'stdA' : 0.25, 
        'stdB' : 0.1, 
        'std0' : 0.0,
    }

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    rff = RandomFourierFeatures(subkey,
        n_features=64,
        bandwidth=1.8,
        method = '2d'
    )

    key, subkey = jax.random.split(key)

    rcde = RandomCDE(key=subkey,
        n_features=250,
        config=config_cde,
        activation='relu'
    )

    svc = KernelSVC(
        C=1000.0,
        gamma='scale'
    )

    print('Starting Random Fourier Features')
    X_rff_train = rff.get_features(X_train, use_cache=True)
    X_rff_test = rff.get_features(X_test, use_cache=True)

    _max = jnp.max(X_rff_train)
    X_rff_train /= _max
    X_rff_test /= _max

    print('Starting Random CDE')
    _X_rcde_train = rcde.get_features(X_rff_train, use_cache=True)
    _X_rcde_test = rcde.get_features(X_rff_test, use_cache=True)

    _X_rcde_train = _X_rcde_train / jnp.linalg.norm(_X_rcde_train, axis=1, keepdims=True + 1e-10)
    _X_rcde_test = _X_rcde_test / jnp.linalg.norm(_X_rcde_test, axis=1, keepdims=True + 1e-10)

    X_rcde_train = _X_rcde_train @ _X_rcde_train.T
    X_rcde_test = _X_rcde_test @ _X_rcde_train.T

    # _diag = jnp.sqrt(jnp.diag(svc_input) + EPS_)
    # svc_input = svc_input / (_diag[:, None] * _diag[None, :])

    print('Starting Linear SVC')
    svc.fit(X_rcde_train, y_train)
    train_score = svc.score(X_rcde_train, y_train)
    test_score = svc.score(X_rcde_test, y_test)

    print(f"Train score: {train_score:.4f}, Test score: {test_score:.4f}")

    svc_grid = {
        'C': [0.1, 1.0, 10.0, 100, 1000],
        'gamma': ['scale', 'auto'],
    }

    svc = KernelSVC()

    svc.fit_gridsearch(X_rcde_train, y_train, svc_grid)

    print(f"Best parameters found: {svc.best_params}")
    print(f"Best score: {svc.best_score:.4f}")


def main_rde(X_train, X_test, y_train, y_test):

    config_cde = {
        'stdA' : 1.5, 
        'stdB' : 0.1, 
        'std0' : 0.5,
    }

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    rcde = RandomRDE(key=subkey,
        n_features=250,
        order = 2,
        step = 4,
        activation = 'tanh',
        config=config_cde,
    )

    svc = KernelSVC(
        C=100.0,
        gamma='scale'
    )

    # print('Starting Random Fourier Features')
    # X_rff_train = rff.get_features(X_train, use_cache=True)
    # X_rff_test = rff.get_features(X_test, use_cache=True)
    # X_rff_train /= X_rff_train.max()
    # X_rff_test /= X_rff_test.max()

    print('Starting Random CDE')
    _X_rcde_train = rcde.get_gram(X_train, use_cache=True)
    _X_rcde_test = rcde.get_gram(X_test, use_cache=True, testing=True)

    X_rcde_train = _X_rcde_train @ _X_rcde_train.T
    X_rcde_test = _X_rcde_test @ _X_rcde_train.T

    print('Starting Linear SVC')
    svc.fit(X_rcde_train, y_train)
    train_score = svc.score(X_rcde_train, y_train)
    test_score = svc.score(X_rcde_test, y_test)

    print('='*50)
    print(X_rcde_train[:7,:7])
    print(X_rcde_test[:7,:7])
    print('='*50)

    print(svc.predict(X_rcde_train))
    print(y_train)
    print('='*50)
    print(svc.predict(X_rcde_test))
    print(y_test)
    print('='*50)

    print(f"Train score: {train_score:.4f}, Test score: {test_score:.4f}")

    print('Here')

    svc_grid = {
        'C': [0.1, 1.0, 10.0, 1000.0],
        'gamma': ['scale', 'auto'],
    }

    svc = KernelSVC()

    svc.fit_gridsearch(X_rcde_train, y_train, svc_grid)

    print(f"Best parameters found: {svc.best_params}")
    print(f"Best score: {svc.best_score:.4f}")
    svc.score(X_rcde_test, y_test)
    print(f"Test score with best parameters: {svc.test_score:.4f}")



def main2(X_train, X_test, y_train, y_test):

    config_cde = {
        'stdA' : 0.5, 
        'stdB' : 0.1, 
        'std0' : 1.5,
    }

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    rcde = RandomCDE(key=subkey,
        n_features=250,
        config=config_cde,
    )

    svc = KernelSVC(
        C=10000.0,
        gamma='scale'
    )

    print('Starting Random CDE')
    _X_rcde_train = rcde.get_features(X_train, use_cache=True)
    _X_rcde_test = rcde.get_features(X_test, use_cache=True)

    _X_rcde_train = _X_rcde_train / jnp.linalg.norm(_X_rcde_train, axis=1, keepdims=True + 1e-10)
    _X_rcde_test = _X_rcde_test / jnp.linalg.norm(_X_rcde_test, axis=1, keepdims=True + 1e-10)

    X_rcde_train = _X_rcde_train @ _X_rcde_train.T
    X_rcde_test = _X_rcde_test @ _X_rcde_train.T

    # _diag = jnp.sqrt(jnp.diag(svc_input) + EPS_)
    # svc_input = svc_input / (_diag[:, None] * _diag[None, :])

    print('Starting Linear SVC')
    svc.fit(X_rcde_train, y_train)
    train_score = svc.score(X_rcde_train, y_train)
    test_score = svc.score(X_rcde_test, y_test)

    print(f"Train score: {train_score:.4f}, Test score: {test_score:.4f}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Random Fourier Features and Random CDE with Linear SVC")
    parser.add_argument('--n_features', type=int, default=128, help='Number of features for RFF')
    parser.add_argument('--n_fourier_features', type=int, default=100, help='Number of Fourier features for RFF')
    parser.add_argument('--bandwidth', type=float, default=1.0, help='Bandwidth for RFF')

    args = parser.parse_args()

    # Load dataset
    X_train, y_train, X_test, y_test = load_dataset("Handwriting")

    print(X_train.shape)

    preprocess_params = {
        'normalize': True,
        'add_time': True,
        'lead_lag': False,
        'basepoint': True,
        'max_len': 200,
        'max_time': 100
    }

    # Apply the preprocessing_class
    preprocessing_class = Preprocessor(**preprocess_params)
    X_train = preprocessing_class.fit_transform(X_train)
    X_test = preprocessing_class.transform(X_test)

    # print('='*50)
    # print(X_train[:,:5])
    # print(X_test[:,:5])
    # print('='*50)
    # print(y_train)
    # print(y_test)
    # print('='*50)

    main(X_train, X_test, y_train, y_test)

    