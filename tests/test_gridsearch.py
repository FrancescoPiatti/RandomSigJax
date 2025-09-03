import argparse

from src.utils.load_ucr_uea_datasets import load_dataset
from src.svc.GridSearchSVC import GridSearchSVC

import warnings 
warnings.filterwarnings("ignore")


def main_cde():

    print("Loading dataset ...")
    X_tr, y_tr, X_te, y_te = load_dataset("Libras")

    X_tr = X_tr[::2]
    y_tr = y_tr[::2]
    X_te = X_te[::2]
    y_te = y_te[::2]

    # feature extraction
    param_grid = {
        'n_features': [16, 32],
        'n_fourier_features': [None, 4],
        'penalty':['l1','l2'],
        'gamma':['scale','auto'],
        'activation': ['id', 'relu', 'tanh'],
        'stdA': [1,0.1],
        'stdB': [0,], 
        'std0': 0.,
        'C': [2,3],
        'normalize': [True],
        'basepoint': [False, True],
        'lead_lag': [False, True],
        'max_time': 10,
        'normalize_feat':[False, True]
    }


    gs = GridSearchSVC('cde',
                       param_grid=param_grid,
                       verbose = True,
                       batch_size=100)
    
    gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft_lin_1d')

    print('Done')

    gs = GridSearchSVC('cde',
                       rff_type='2d',
                       param_grid=param_grid,
                       verbose=True,
                       batch_size=100)
    
    gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft_lin_2d')

    print('Done')

    gs = GridSearchSVC('cde',
                       linear_svc=False,
                       param_grid=param_grid,
                       batch_size=100)
    
    gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft_ker_1d')

    print('Done')


def main_rde():

    print("Loading dataset ...")
    X_tr, y_tr, X_te, y_te = load_dataset("Libras")

    X_tr = X_tr[::2]
    y_tr = y_tr[::2]
    X_te = X_te[::2]
    y_te = y_te[::2]

    # feature extraction
    param_grid = {
        'n_features': [4, 8],
        'n_fourier_features': [None],
        'penalty':['l1','l2'],
        'gamma':['scale','auto'],
        'activation': ['id', 'relu', 'tanh'],
        'stdA': [1,0.1],
        'stdB': [0,], 
        'std0': 0.,
        'C': [2,3],
        'order':[2,3],
        'normalize': [True],
        'basepoint': [False, True],
        'lead_lag': [False, True],
        'max_time': 10,
        'normalize_feat':[False, True]
    }


    gs = GridSearchSVC('rde',
                       param_grid=param_grid,
                       batch_size=100)
    
    gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft_lin_1d')

    print('Done')

    gs = GridSearchSVC('rde',
                       linear_svc=False,
                       param_grid=param_grid,
                       batch_size=100)
    
    gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft_ker_1d')

    print('Done')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test RandomDE")
    parser.add_argument('--rde', action='store_true', help='For Random RDE')
    args = parser.parse_args()

    if args.rde:
        main_rde()
    else:
        main_cde()