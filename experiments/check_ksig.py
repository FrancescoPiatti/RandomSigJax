import jax
import jax.numpy as jnp
import jax.random as jr

import ksig


if __name__ == '__main__':

    # Set up brownian motion
    key = jr.PRNGKey(42)
    bm = jnp.cumsum(jr.normal(key, (10, 20, 5)), axis=1)
    
    rff = ksig.static.features.RandomFourierFeatures(bandwidth=1., n_components=20)
    dp = ksig.static.features.DiagonalProjection(n_components=20)

    print('RFF and DP initialized.')

    _cls = ksig.kernels.SignatureFeatures(n_levels=3, static_features=rff, projection=dp)

    feats = _cls.fit_transform(bm)

    print("Features shape:", feats.shape)

