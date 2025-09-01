import jax.nn as nn

ACTIVATION_DICT = {
    'id': nn.identity,
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'tanh': nn.tanh,
    'softplus': nn.softplus,
    'softmax': nn.softmax,
    'elu': nn.elu,
    'leaky_relu': nn.leaky_relu,
    'selu': nn.selu,
    'gelu': nn.gelu,
    'swish': nn.swish,
    'mish': nn.mish,
    'logsigmoid': nn.log_sigmoid,
    'logsoftmax': nn.log_softmax,
}