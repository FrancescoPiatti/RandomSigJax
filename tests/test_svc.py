import jax
import jax.numpy as jnp
import argparse

from src_torch.svc.LinearSVC import LinearSVC

key = jax.random.PRNGKey(0)
X = jnp.random.randn(key, (200, 10))
y = jnp.random.randint(key, (200,), 0, 2)


svc_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1.0, 10.0],
    'max_iter': [100, 1000]
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    model = LinearSVC(gpu=args.gpu)
    model.fit(X, y)
    preds = model.predict(X)
    score = model.score(X, y)

    print(type(preds))
    print(preds.shape)
    print("Score:", score)

    model = LinearSVC(gpu=args.gpu)
    model.fit_gridsearch(X, y, svc_grid, cv=4, stratified=True)
    preds = model.predict(X)
    score = model.score(X, y)
    # best score

    print(type(preds))
    print(preds.shape)
    print("Score:", score)



