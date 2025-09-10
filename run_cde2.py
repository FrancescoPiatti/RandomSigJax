import jax

import argparse
import os

from src.svc.GridSearchSVC import GridSearchSVC
from src.utils.load_ucr_uea_datasets import load_dataset
from src.utils.logger import Logger

import warnings 

warnings.filterwarnings("ignore")

CONFIG_CDE_GS = {
    'n_features': [250],
    'n_fourier_features': [None, 32, 64, 128, 256, 512, 1024],
    'activation': ['id', 'tanh'],
    'stdA': [0.25, 0.5, 1.0, 1.5],
    'stdB': [0.0, 0.1],
    'std0': [0.0, 0.5, 1.0, 1.5],
    'normalize_feat': [False, True],
}


BANDWIDTH_RATIOS = [
    0.1,
    0.25,
    0.5,
    1.,
    2.5,
    5.,
    10.,
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
    'basepoint': [True],
    'normalize': [True],
    'max_time': [1e0, 1e1, 1e2, 1e3],
    'max_len': [200],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--dataset', type=str, nargs='+', required=True, help='Name of UEA dataset')
    parser.add_argument('--linsvc', action='store_true', help='Use linear SVC instead of SVC')
    parser.add_argument('--rff2d', action='store_true', help='Use RFF 2D instead of 1D')
    parser.add_argument('--no_logger', action='store_true', help='Don\'t use logger to log messages to file')
    parser.add_argument('-bs', '--batchsize', type=int, default=100, help='Batch size for GS and computing features')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    for dataset in args.dataset:

        _name = ''
        _name += 'lin' if args.linsvc else 'ker'
        _name += '2d' if args.rff2d else '1d'

        # 1. Create the directory name
        dir_name = f"results_{_name}"
        
        # 2. Make sure the directory exists
        os.makedirs(dir_name, exist_ok=True)
        
        # 3. Construct filename (inside that directory)
        filename = os.path.join(dir_name, f"{dataset}_cde_")

        if not args.no_logger:
            logger = Logger(os.path.join(dir_name, f"{dataset}_cde_log.txt"))
            logger.log(f"Loading dataset {dataset} ...")

        else:
            print(f"Loading dataset {dataset} ...")
            logger = False

        X_train, y_train, X_test, y_test = load_dataset(dataset)

        if not args.no_logger:
            logger.log(f"Dataset {dataset} loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            logger.log(f"Shape of train labels: {y_train.shape}, Test labels: {y_test.shape}")
        else:
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

        if args.linsvc:
            param_grid = {**CONFIG_CDE_GS, **CONFIG_PRE_GS, **CONFIG_LIN_SVC_GS}
        else:
            param_grid = {**CONFIG_CDE_GS, **CONFIG_PRE_GS, **CONFIG_SVC_GS}


        # ===============================================================================================

        # Suggest bandwidth based on training data and add to param_grid
        # suggested_bandwidth = suggest_bandwidth(X_train)
        # param_grid['bandwidth'] = [suggested_bandwidth * br for br in BANDWIDTH_RATIOS]

        param_grid['bandwidth'] = BANDWIDTH_RATIOS

        # Cap number of RFF if data dimension is small
        param_grid['n_fourier_features'] = [nff for nff in param_grid['n_fourier_features'] 
                                            if (nff is None) or (nff <= X_train.shape[2] * 50)]
        if (X_train.shape[2] * 50) - param_grid['n_fourier_features'][-1] > 100  and (X_train.shape[2] * 50) <= 1024:
            param_grid['n_fourier_features'].append(X_train.shape[2] * 50)

        # ===============================================================================================


        gs_ = GridSearchSVC('cde',
                            param_grid = param_grid,
                            linear_svc = args.linsvc,
                            rff_type = rff_type,
                            gpu = gpu,
                            verbose = logger,
                            batch_size = 128,
                            stratified = True,
                            n_splits = 4,
                            random_state = 42)
        

        gs_.fit(X_train, y_train, X_test, y_test, filename, save=True)

        if not args.no_logger:
            logger.log(f"Results saved to {filename} for dataset {dataset} with type {args.type}.")
        else:
            print(f"Results saved to {filename} for dataset {dataset} with type {args.type}.")

