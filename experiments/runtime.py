from src.features.RandomCDE import RandomCDE
from src.features.RandomRDE import RandomRDE
from src.features.RandomFourierFeatures import RandomFourierFeatures

from experiments.dataloaders.load_datasets import load_ucr_uea_dataset

try:
    import ksig 
    ksig_available = True

except Exception as e:
    ksig = None
    ksig_available = False
    print(f"Could not import ksig. {e}")

import jax.numpy as jnp
import jax.random as jr

import timeit
import argparse
import pandas as pd




def get_model(model : str,
              bm : jnp.ndarray,
              out_dim : int = 100,
              rff_dim : int = 100,
              order : int = 4, 
              step : int = 10):
    
    if model == 'rcde-cache':
        model = 'rcde'
        cache = True
        cache_logsigs = False
    elif model == 'rrde-cache-mat':
        model = 'rrde'
        cache = True
        cache_logsigs = False
    elif model == 'rrde-cache-all':
        model = 'rrde'
        cache = True
        cache_logsigs = True
    else:
        cache = False
        cache_logsigs = False


    if model == 'rcde':
    
        _cls1 = RandomFourierFeatures(key=42, method='1d', n_features=rff_dim)
        _cls2 = RandomCDE(key=42, n_features=out_dim)

        if cache:
            _bm_small = bm[:2]
            _rff_small = _cls1.get_features(_bm_small, use_cache=True)
            _ = _cls2.get_features(_rff_small, use_cache=True)

        def _run_fn():
            rff_feats = _cls1.get_features(bm, use_cache=cache)
            _ = _cls2.get_features(rff_feats, use_cache=cache)
            return None

        return _run_fn


    elif model == 'rrde':
    
        _cls = RandomRDE(key=42, n_features=out_dim, order=order, step=step)


        if cache:
            _bm_small = bm[:2]
            _ = _cls.get_features(_bm_small, use_cache=True, testing=False)

            if not cache_logsigs:
               _cls.clear_cache('logsigs')


        def _run_fn():
            _ = _cls.get_features(bm, use_cache=cache, testing=False)
            return None

        return _run_fn
    

    elif model == 'rfsf-dp' and ksig_available:

        rff = ksig.static.features.RandomFourierFeatures(bandwidth=1., n_components=rff_dim)
        dp = ksig.static.features.DiagonalProjection(n_components=rff_dim)
        
        def _run_fn():
            
            _cls = ksig.kernels.SignatureFeatures(n_levels=order, 
                                                  order=1, 
                                                  static_features=rff, 
                                                  projection=dp,
                                                  difference=True, 
                                                  normalize=True)
            
            _ = _cls.fit_transform(bm)
            return None
        

    elif model == 'rfsf-trp' and ksig_available:

        rff = ksig.static.features.RandomFourierFeatures(bandwidth=1., n_components=rff_dim)
        trp = ksig.static.features.TensorizedRandomProjection(n_components=rff_dim)
        
        def _run_fn():
            
            _cls = ksig.kernels.SignatureFeatures(n_levels=order, 
                                                  order=1, 
                                                  static_features=rff, 
                                                  projection=trp,
                                                  difference=True, 
                                                  normalize=True)
            
            _ = _cls.fit_transform(bm)
            return None
        
    else:
        raise ValueError(f"Model {model} not recognized.")
    

if __name__ == "__main__":

    # Constants
    BATCH_SIZE = 16
    TIMESTEPS = 200
    INP_DIM = 10
    OUT_DIM = 200
    RFF_DIM = 100
    ORDER = 4
    STEP = 10

    # Timer settings
    TIMES = 10
    REPEAT = 5

    # Model list
    MODEL_LIST_ALL = ['rcde', 'rrde', 'rcde-cache', 'rrde-cache-mat', 'rrde-cache-all']
    
    MODEL_LIST_RFF = []  
    MODEL_LIST_ORDER = []
    MODEL_LIST_STEP = []

    # Parameter lists
    RFF_LIST = [50, 100, 200]
    OUT_LIST = [50, 100, 200]
    ORDER_LIST = [2, 4, 6]
    STEP_LIST = [5, 10, 20]


    # Argument parser
    parser = argparse.ArgumentParser(description="Benchmark Random CDE/RDE models with RFF.")
    parser.add_argument('--rff', action='store_true', help="Analyze performance over different RFF dimensions.")
    parser.add_argument('--out', action='store_true', help="Analyze performance over different output dimensions.")
    args = parser.parse_args()


    def timer_func(func, times : int = 10, repeat : int = 5) -> float:
        timer = timeit.Timer(lambda : func())
        results = timer.repeat(number=times, repeat=repeat)
        elapsed_time = sum(results) / times / repeat
        return elapsed_time



    # Generate random Brownian motion data
    jr_key = jr.PRNGKey(0)
    bm = jr.normal(jr_key, (BATCH_SIZE, TIMESTEPS, INP_DIM))  # Example batch

    
    if args.rff:
        
        results_df = pd.DataFrame(index=RFF_LIST)

        for model in MODEL_LIST_RFF:

            elapsed_times = []

            for rff_dim in RFF_LIST:

                run_fn = get_model(model, bm, out_dim=OUT_DIM, rff_dim=rff_dim, order=ORDER, step=STEP)
                elapsed_time = timer_func(run_fn, times=TIMES, repeat=REPEAT)
                elapsed_times.append(elapsed_time)

                print(f"Model: {model}, RFF Dim: {rff_dim}, Time: {elapsed_time:.6f} seconds")

            results_df[model] = elapsed_times

        # Save results without converting to pandas DataFrame
        results_df.to_csv("time_rff_dim.csv")


    if args.out:
        
        results_df = pd.DataFrame(index=OUT_LIST)

        for model in MODEL_LIST_ALL:

            elapsed_times = []

            for out_dim in OUT_LIST:

                run_fn = get_model(model, bm, out_dim=out_dim, rff_dim=RFF_DIM, order=ORDER, step=STEP)
                elapsed_time = timer_func(run_fn, times=TIMES, repeat=REPEAT)
                elapsed_times.append(elapsed_time)

                print(f"Model: {model}, Out Dim: {out_dim}, Time: {elapsed_time:.6f} seconds")

            results_df[model] = elapsed_times

        # Save results without converting to pandas DataFrame
        results_df.to_csv("time_out_dim.csv")


    if args.order:

        results_df = pd.DataFrame(index=ORDER_LIST)

        for model in MODEL_LIST_ORDER:

            elapsed_times = []

            for order in ORDER_LIST:

                run_fn = get_model(model, bm, out_dim=OUT_DIM, rff_dim=RFF_DIM, order=order, step=STEP)
                elapsed_time = timer_func(run_fn, times=TIMES, repeat=REPEAT)
                elapsed_times.append(elapsed_time)

                print(f"Model: {model}, Order Dim: {order}, Time: {elapsed_time:.6f} seconds")

            results_df[model] = elapsed_times

        # Save results without converting to pandas DataFrame
        results_df.to_csv("time_order_dim.csv")


    if args.step:

        results_df = pd.DataFrame(index=STEP_LIST)

        for model in MODEL_LIST_STEP:

            elapsed_times = []

            for step in STEP_LIST:

                run_fn = get_model(model, bm, out_dim=OUT_DIM, rff_dim=RFF_DIM, order=ORDER, step=step)
                elapsed_time = timer_func(run_fn, times=TIMES, repeat=REPEAT)
                elapsed_times.append(elapsed_time)

                print(f"Model: {model}, Step Dim: {step}, Time: {elapsed_time:.6f} seconds")

            results_df[model] = elapsed_times

        # Save results without converting to pandas DataFrame
        results_df.to_csv("time_step_dim.csv")


    if args.uea:

        UCR_UEA_DATASETS = [
            'ArticularyWordRecognition',
            'AtrialFibrillation',
            'BasicMotions',
            'CharacterTrajectories',
            'Cricket',
            'ERing',
            'EigenWorms',
            'Epilepsy',
            'EthanolConcentration',
            # 'FaceDetection',
            'FingerMovements',
            'HandMovementDirection',
            'Handwriting',
            'Heartbeat',
            'JapaneseVowels',
            # 'LSST',
            'Libras',
            'MotorImagery',
            'NATOPS',
            'PEMS-SF',
            'PenDigits',
            'PhonemeSpectra',
            'RacketSports',
            'SelfRegulationSCP1',
            'SelfRegulationSCP2',
            'SpokenArabicDigits',
            'StandWalkJump',
            'UWaveGestureLibrary'
        ]

        results_df = pd.DataFrame(index=UCR_UEA_DATASETS)


        for model in MODEL_LIST_ALL:
                
            elapsed_times = []

            for d in UCR_UEA_DATASETS:

                x, _, _, _ = load_ucr_uea_dataset(d)

                run_fn = get_model(model, x, out_dim=OUT_DIM, rff_dim=RFF_DIM, order=ORDER, step=step)
                elapsed_time = timer_func(run_fn, times=TIMES, repeat=REPEAT)
                elapsed_times.append(elapsed_time)


            results_df[d]



                








    



    







        



