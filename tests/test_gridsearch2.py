import argparse

from experiments.dataloaders.load_datasets import load_dataset
from src.svc.GridSearchSVC import GridSearchSVC

from src.utils.logger import Logger
import warnings 
warnings.filterwarnings("ignore")


logger = Logger('tests/results/log1.log')

def main_cde():

    logger.log("Loading dataset ...")
    X_tr, y_tr, X_te, y_te = load_dataset("Libras")

    # X_tr = X_tr[::2]
    # y_tr = y_tr[::2]
    # X_te = X_te[::2]
    # y_te = y_te[::2]

    # feature extraction
    param_grid = {
        'n_features': [250],
        'n_fourier_features': [None, 64],
        # 'penalty':['l1','l2'],
        # 'gamma':['scale','auto'],
        'gamma':'scale',
        'bandwidth':[0.5,1,1.5],
        'activation': ['id'],
        'stdA': [0.25],
        'stdB': [0.1], 
        'std0': 0.,
        'C': [10, 100, 1000,10000],
        'add_time':True,
        'normalize': [True],
        'basepoint': [True],
        'lead_lag': [False, True],
        'max_time': 1,
        'normalize_feat':[False, True]
    }


    # gs = GridSearchSVC('cde',
    #                    param_grid=param_grid,
    #                    verbose=logger,
    #                    batch_size=100)
    
    # gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft_lin_1d')

    # logger.log('Done')

    gs = GridSearchSVC('cde',
                       rff_type='2d',
                       linear_svc=False,
                       param_grid=param_grid,
                       verbose=logger,
                       batch_size=100)
    
    gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft1')

    logger.log('Done')

    # gs = GridSearchSVC('cde',
    #                    linear_svc=False,
    #                    param_grid=param_grid,
    #                    verbose=logger,
    #                    batch_size=100)
    
    # gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft_ker_1d')

    # logger.log('Done')


def main_rde():

    logger.log("Loading dataset ...")
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
        'normalize_feat':[False, True],
        'bandwidth':[0.5,1,1.5]
    }


    gs = GridSearchSVC('rde',
                       param_grid=param_grid,
                       verbose=logger,
                       batch_size=100)
    
    gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft_lin_1d')

    logger.log('Done')

    gs = GridSearchSVC('rde',
                       linear_svc=False,
                       param_grid=param_grid,
                       verbose=logger,
                       batch_size=100)
    
    gs.fit(X_tr, y_tr, X_te, y_te, 'tests/results/draft_ker_1d')

    logger.log('Done')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test RandomDE")
    parser.add_argument('--rde', action='store_true', help='For Random RDE')
    args = parser.parse_args()

    if args.rde:
        main_rde()
    else:
        main_cde()