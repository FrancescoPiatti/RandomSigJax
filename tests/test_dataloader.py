
from experiments.dataloaders.load_datasets import load_weather_monash
from experiments.dataloaders.load_datasets import load_etth1
from experiments.dataloaders.load_datasets import load_ucr_uea_dataset


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

if __name__ == "__main__":

    # print('UCR/UEA Datasets:')
    # for dataset in UCR_UEA_DATASETS:

    #     xtr, ytr, xte, yte = load_ucr_uea_dataset(dataset)

    #     print(dataset, 'Train:', xtr.shape, 'Test:', xte.shape)

    # print('='*70)
    # print('ETTh1 Dataset:')

    # x, t = load_etth1()
    # print('Train:', x.shape, 'Times:', t.shape)


    print('='*70)
    print('Weather Dataset:')

    x, t, metadata = load_weather_monash()
    print('Train:', x.shape, 'Times:', t.shape)

    print('Metadata:', metadata)

