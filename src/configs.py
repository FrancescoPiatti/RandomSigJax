# ----------------------------- Configs for feature classes ----------------------------- 

DEFAULT_CONFIG_RCDE = {
    'stdA' : 1.0, 
    'stdB' : 0.0, 
    'std0' : 1.0
}

DEFAULT_CONFIG_RRDE = {
    'stdA' : 1.0, 
    'stdB' : 0.0, 
    'std0' : 1.0
}

# ----------------------------- Configs for GS ----------------------------- 

DEFAULT_CDE_GS = {
    'n_features': [32, 64, 128],
    'n_fourier_features': [None, 8, 16, 32, 64, 128],
    'bandwidth': [1., 0.5],
    'activation': ['id', 'relu', 'tanh', 'swish'],
    'stdA': [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 10.0,],
    'stdB': [0.0, 0.01, 0.1],
    'std0': [0.1, 0.5, 1.0, 2.0, 5.0,],
    'cubic': [False],
    'normalize_feat': [False, True],
}

DEFAULT_RDE_GS = {
    'n_features': [100, 200],
    'n_fourier_features': [50, 100],
    'step':[4,6],
    'bandwidth': [0.1, 0.5],
    'order': [1, 2],
    'activation': ['relu', 'tanh'],
    'stdA': [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 10.0,],
    'stdB': [0.0, 0.01, 0.1],
    'std0': [0.1, 0.5, 1.0, 2.0, 5.0,],
    'normalize_feat': [False, True],
}

DEFAULT_SVC_GS = {
    'C': [1e0, 1e1, 1e2, 1e3, 1e4],
    'gamma': ['scale', 'auto'],
}

DEFAULT_LIN_SVC_GS = {
    'C': [1e0, 1e1, 1e2, 1e3, 1e4],
    'penalty': ['l1','l2'],
}

DEFAULT_PRE_GS = {
    'add_time': [False],
    'lead_lag': [False],
    'basepoint': [False],
    'normalize': [False],
    'max_time': [1.0],
    'max_len': [None],
}