import jax
import jax.numpy as jnp
import jax.random as random

from src.features.RandomCDE import RandomCDE
from src.features.RandomFourierFeatures import RandomFourierFeatures
from src.features.RandomRDE import RandomRDE

from src.signatures import get_logsignatures
from src.configs import DEFAULT_CONFIG_RCDE
# from src.configs import DEFAULT_CONFIG_RRDE

key1 = jax.random.PRNGKey(0)
key2 = jax.random.PRNGKey(1)

def test_rcde():
    """
    Test the RandomCDE feature extraction.
    """
    
    # Generate random data
    X = random.normal(key1, (100, 20, 5))

    rcde = RandomCDE(key2,
                     n_features=10, 
                     activation='relu',
                     config=DEFAULT_CONFIG_RCDE,
                     cache=None)

    # Extract features
    feat1 = rcde.get_features(X)
    feat2 = rcde.get_features(X, batch_size=35)
    feat3 = rcde.get_features(X, return_interval=True)

    assert feat1.shape == (100, 10), "Feat1 shape mismatch"
    assert feat2.shape == (100, 10), "Feat2 shape mismatch"
    assert feat3.shape == (100, 20, 10), "Feat3 shape mismatch"
    assert not jnp.allclose(feat1, feat2), "Feat1 and Feat2 should not be close"
    
    feat4 = rcde.get_features(X, use_cache=True)
    feat5 = rcde.get_features(X, use_cache=True)

    assert jnp.array_equal(feat4, feat5), "Cached mismatch"


    print("RandomCDE test passed successfully!")


def test_rff():
    """
    Test the RandomFourier feature extraction.
    """
    
    # Generate random data
    X = random.normal(key1, (100, 20, 5))

    rff = RandomFourierFeatures(key2,
                                method='2D',
                                n_features=10,
                                bandwidth=1.0,
                                cache=None)

    feat1 = rff.get_features(X)
    feat2 = rff.get_features(X, use_cache=True)
    feat3 = rff.get_features(X, use_cache=True)

    assert feat1.shape == (100, 20, 10*2), "Feat1 shape mismatch"
    assert jnp.array_equal(feat2, feat3), "Feat2 and Feat3 not equal"

    rff = RandomFourierFeatures(key2,
                                method='1D',
                                n_features=10,
                                bandwidth=1.0,
                                cache=None)

    feat4 = rff.get_features(X)
    feat5 = rff.get_features(X, use_cache=True)
    feat6 = rff.get_features(X, use_cache=True)

    assert feat4.shape == (100, 20, 10), "Feat4 shape mismatch"
    assert jnp.array_equal(feat5, feat6), "Feat5 and Feat6 not equal"

    print("RandomFourierFeatures test passed successfully!")


def test_rrde():
    """
    Test the RandomCDE feature extraction.
    """
    
    # Generate random data
    X = random.normal(key1, (100, 20, 5))

    _, ls_t, ls_d = get_logsignatures(X, step=4, order=4, min_length=3).shape

    rrde = RandomRDE(key2,
                     n_features=10, 
                     activation='relu',
                     step=4,
                     config=DEFAULT_CONFIG_RCDE,
                     cache=None)

    # Extract features
    feat1 = rrde.get_features(X)
    feat2 = rrde.get_features(X, batch_size=35)
    feat3 = rrde.get_features(X, return_interval=True)

    assert feat1.shape == (100, 10), "Feat1 shape mismatch"
    assert feat2.shape == (100, 10), "Feat2 shape mismatch"
    assert feat3.shape == (100, ls_t+1, 10), "Feat3 shape mismatch"
    assert not jnp.allclose(feat1, feat2), "Feat1 and Feat2 should not be close"
    
    feat4 = rrde.get_features(X, use_cache=True)
    feat5 = rrde.get_features(X, use_cache=True)
    feat6 = rrde.get_features(X, use_cache=False)

    assert jnp.array_equal(feat4, feat5), "Cached mismatch"
    assert not jnp.array_equal(feat4, feat6), "Feat4 and Feat6 should not be equal"

    print("RandomRDE test passed successfully!")


if __name__ == "__main__":
    
    print('Starting tests for feature extraction classes...')

    # Run the test
    print('='*100)
    print('Testing RandomCDE...')
    test_rcde()

    print('='*100)
    print('Testing RandomFourierFeatures...')
    test_rff()

    print('='*100)
    print('Testing RandomRDE...')
    test_rrde()
    print('='*100)