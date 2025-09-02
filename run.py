import jax

import argparse
import os

from src.svc.GridSearchSVC import GridSearchSVC
from src.utils.load_ucr_uea_datasets import load_dataset
from src.utils.hyperparams import suggest_bandwidth
from src.utils.hyperparams import suggest_stepsize
import warnings 

warnings.filterwarnings("ignore")

CONFIG_CDE_GS = {
    'n_features': [64, 256, 512, 1024],
    'n_fourier_features': [None, 32, 64, 128, 256, 512, 1024],
    'activation': ['id', 'tanh'],
    'stdA': [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
    'stdB': [0.0, 0.01, 0.1],
    'std0': [0.0, 0.1, 0.5, 1.0, 2.0],
    'cubic': [False],
    'normalize_feat': [False, True],
}

CONFIG_RDE_GS = {
    'n_features': [64, 128, 256, 512, 1024],
    'n_fourier_features': [None, 32, 64, 128, 512, 1024],
    'order': [2, 3, 4, 5],
    'activation': ['id','tanh'],
    'stdA': [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
    'stdB': [0.0, 0.01, 0.1],
    'std0': [0.0, 0.5, 1.0, 2.0],
    'normalize_feat': [False, True],
}

BANDWIDTH_RATIOS = [
    0.05,
    0.1,
    0.25,
    0.5,
    1.,
    2.5,
    5.,
    10.,
    50.,
]

# OK!
CONFIG_LIN_SVC_GS = {
    'C': [1e0, 1e1, 1e2, 1e3, 1e4],
    'penalty': ['l1', 'l2'],
}

# OK!
CONFIG_SVC_GS = {
    'C': [1e0, 1e1, 1e2, 1e3, 1e4],
    'gamma': ['scale', 'auto'],
}

# OK!
CONFIG_PRE_GS = {
    'add_time': True,
    'lead_lag': [False, True],
    'basepoint': [False, True],
    'normalize': [True],
    'max_time': [0., 1e0, 1e1, 1e2, 1e3],
    'max_len': [200, 400],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', required=True, help='Name of UEA dataset')
    parser.add_argument('--filename', type=str, default='results', help='Filename to save results')
    parser.add_argument('--type', type=str, default='cde', choices=['cde', 'rde'])
    parser.add_argument('--linsvc', action='store_true', help='Use linear SVC instead of SVC')
    parser.add_argument('--rff2d', action='store_true', help='Use RFF 2D instead of 1D')

    args = parser.parse_args()

    for dataset in args.dataset:

        _name = ''
        _name += 'lin' if args.linsvc else 'ker'
        _name += '2d' if args.rff2d else '1d'

        # 1. Create the directory name
        dir_name = f"results_{_name}"
        
        # 2. Make sure the directory exists
        os.makedirs(dir_name, exist_ok=True)
        
        # 3. Construct filename (inside that directory)
        filename = os.path.join(dir_name, f"{dataset}_{args.type}_")

        print(f"Loading dataset {dataset} ...")

        X_train, y_train, X_test, y_test = load_dataset(dataset)

        print(f"Dataset {dataset} loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"Shape of train labels: {y_train.shape}, Test labels: {y_test.shape}")

        if any(d.platform == "gpu" for d in jax.devices()):
            gpu = True 
        else:
            gpu = False

        if args.rff2d:
            rff_type = '2d'
        else:
            rff_type = '1d'

        if args.type == 'cde':
            param_grid = CONFIG_CDE_GS
        elif args.type == 'rde':
            param_grid = CONFIG_RDE_GS

        if args.linsvc:
            param_grid = {**param_grid, **CONFIG_PRE_GS, **CONFIG_LIN_SVC_GS}
        else:
            param_grid = {**param_grid, **CONFIG_PRE_GS, **CONFIG_SVC_GS}

        # Suggest bandwidth based on training data and add to param_grid
        suggested_bandwidth = suggest_bandwidth(X_train)
        param_grid['bandwidth'] = [suggested_bandwidth * br for br in BANDWIDTH_RATIOS]

        # Suggest step size based on data length
        param_grid['step'] = suggest_stepsize(X_train.shape[1])

        # Cap number of RFF if data dimension is small
        param_grid['n_fourier_features'] = [nff for nff in param_grid['n_fourier_features'] 
                                            if (nff is None) or (nff <= X_train.shape[2] * 50)]
        if (X_train.shape[2] * 50) - param_grid['n_fourier_features'][-1] > 100  and (X_train.shape[2] * 50) <= 1024:
            param_grid['n_fourier_features'].append(X_train.shape[2] * 50)


        gs_ = GridSearchSVC(args.type,
                            param_grid = param_grid,
                            linear_svc = args.linsvc,
                            rff_type = rff_type,
                            gpu = gpu,
                            batch_size = 128,
                            stratified = True,
                            n_splits = 4,
                            random_state = 42)
        

        gs_.fit(X_train, y_train, X_test, y_test, filename, save=True)

        print(f"Results saved to {filename} for dataset {dataset} with type {args.type}.")

