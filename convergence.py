import jax
import jax.random as random
import jax.numpy as jnp

import argparse
import os

from src.features.RandomCDE import RandomCDE
from src.features.RandomFourierFeatures import RandomFourierFeatures

from polysigkernel import SigKernel 


key = random.PRNGKey(1)
key, subkey1, subkey2 = random.split(key, 3)

# 4-d brownian motion
x = 1e-1 * (random.normal(subkey1, shape=(3, 100, 4))).cumsum(axis=1)
y = 1e-1 * (random.normal(subkey2, shape=(3,100, 4))).cumsum(axis=1)

# Signature kernel 

sigkernel_cls = SigKernel(order=5, static_kernel='linear')
sigkernel = sigkernel_cls.kernel_matrix(x, y)

print("SigKernel shape:", sigkernel.shape)

sigrbfkernel_cls = SigKernel(order=5, static_kernel='rbf')
sigrbfkernel = sigrbfkernel_cls.kernel_matrix(x, y)

print("SigRBFKernel shape:", sigrbfkernel.shape)


print('='*50)

key, subkey = random.split(key)

randomff_cls = RandomFourierFeatures(subkey, method='1d', n_features=200)

x_fourier = randomff_cls.get_features(x, use_cache=True)
y_fourier = randomff_cls.get_features(y, use_cache=True)


n_features = 500
samples = 50
feats = []
feats_rbf = []

for _ in range(samples):

    key, subkey = random.split(key)
    
    rcde_class = RandomCDE(subkey, n_features=n_features)
    feat_kernel = rcde_class.get_gram(x, y, use_cache=True, return_interval=False) * n_features
    feat_rbf_kernel = rcde_class.get_gram(x_fourier, y_fourier, use_cache=True, return_interval=False) * n_features

    feats.append(feat_kernel[None, ...])
    feats_rbf.append(feat_rbf_kernel[None, ...])

feat_kernel = jnp.concatenate(feats, axis=0).mean(axis=0)
feat_rbf_kernel = jnp.concatenate(feats_rbf, axis=0).mean(axis=0)

print("feat_kernel shape:", feat_kernel.shape)
print("feat_rbf_kernel shape:", feat_rbf_kernel.shape)

print("="*50)

print('SIGKERNEL')

print(sigkernel)

print(feat_kernel)

print('-'*50)

print('SIGRBF')

print(sigrbfkernel)

print(feat_rbf_kernel)

print('='*50)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", type=int, default='0')
    parser.add_argument("--full", action='store_true', help='Evaluate the MAPE for a list of n_features')
    args = parser.parse_args()

    if any(d.platform == "gpu" for d in jax.devices()):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


    if args.full:   

        n_features_list = [100*i for i in range(1, 11)]
        samples = 50

        abs_error_linear = []
        abs_error_rbf = []

        for n_features in n_features_list:

            for _ in range(samples):

                key, subkey = random.split(key)
                
                rcde_class = RandomCDE(subkey, n_features=n_features)
                feat_kernel = rcde_class.get_gram(x, y, use_cache=True, return_interval=False) * n_features
                feat_rbf_kernel = rcde_class.get_gram(x_fourier, y_fourier, use_cache=True, return_interval=False) * n_features

                feats.append(feat_kernel[None, ...])
                feats_rbf.append(feat_rbf_kernel[None, ...])

            feat_kernel = jnp.concatenate(feats, axis=0).mean(axis=0)
            feat_rbf_kernel = jnp.concatenate(feats_rbf, axis=0).mean(axis=0)

            abs_error_linear.append(jnp.abs((feat_kernel - sigkernel) / sigkernel).mean())
            abs_error_rbf.append(jnp.abs((feat_rbf_kernel - sigrbfkernel) / sigrbfkernel).mean())

        print('Absolute Error (Linear):', abs_error_linear)
        print('Absolute Error (RBF):', abs_error_rbf)
