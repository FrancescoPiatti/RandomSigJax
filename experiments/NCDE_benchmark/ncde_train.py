import argparse
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax

from experiments.dataloaders.load_datasets import load_ucr_uea_dataset

from .ncde import NeuralCDE
from .ncde import count_parameters
from .ncde import evaluate_classification
from .ncde import train_step_classification

def train_neural_cde_classifier(
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray,
    *,
    hidden_size: int = 64,
    width_size: int = 128,
    depth: int = 2,
    lr: float = 1e-3,
    epochs: int = 200,
    seed: int = 0,
    batch_size: Optional[int] = None,
):
    """
    Train Neural CDE classifier on UCR data with optional mini-batching.

    Args:
        X_train: (N_train, L, C)
        y_train: (N_train,)
        X_test:  (N_test, L, C)
        y_test:  (N_test,)
        hidden_size, width_size, depth: architecture hyperparameters
        lr: learning rate
        steps: number of gradient steps
        seed: random seed
        batch_size: mini-batch size. If None, fall back to full-batch.

    Returns:
        model: trained NeuralCDEPathClassifier
        train_acc: final train accuracy
        test_acc: final test accuracy
    """
    _, _, C = X_train.shape
    n_classes = int(jnp.max(y_train)) + 1

    feature_dim = C

    key = jr.PRNGKey(seed)
    model = NeuralCDE(
        data_size=feature_dim,
        hidden_size=hidden_size,
        n_classes=n_classes,
        width_size=width_size,
        depth=depth,
        key=key,
    )

    n_params = count_parameters(model)
    print(f"[Neural CDE] feature_dim={feature_dim}, hidden_size={hidden_size}, "
          f"width_size={width_size}, depth={depth}, params={n_params}")

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    num_train = X_train.shape[0]
    if batch_size is None:
        batch_size = num_train
    batch_size = max(1, min(int(batch_size), num_train))

    for epoch in range(epochs):
        key, shuffle_key = jr.split(key)
        permutation = jr.permutation(shuffle_key, num_train)
        epoch_loss = 0.0
        epoch_acc = 0.0
        seen = 0

        for start in range(0, num_train, batch_size):
            batch_idx = permutation[start : start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            model, opt_state, loss, acc = train_step_classification(
                model, opt_state, X_batch, y_batch, optim
            )
            batch_count = int(X_batch.shape[0])
            epoch_loss += float(loss) * batch_count
            epoch_acc += float(acc) * batch_count
            seen += batch_count

        epoch_loss /= seen
        epoch_acc /= seen

        if epoch % 5 == 0 or epoch == epochs - 1:
            test_loss, test_acc = evaluate_classification(model, X_test, y_test)
            print(
                f"epoch {epoch:3d} | "
                f"train_loss {epoch_loss:.4f} | train_acc {epoch_acc:.4f} | "
                f"test_loss {test_loss:.4f} | test_acc {test_acc:.4f}"
            )

    # Final accuracies
    train_loss, train_acc = evaluate_classification(model, X_train, y_train)
    test_loss, test_acc = evaluate_classification(model, X_test, y_test)
    print(f"Final train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")

    return model, train_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description="Train a Neural CDE on a UCR dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ECG200",
        help="Name of the UCR/UEA dataset to load via aeon.",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--width-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print(f"Loading dataset {args.dataset}...")
    X_train, y_train, X_test, y_test = load_ucr_uea_dataset(args.dataset)
    print(
        f"Dataset stats -- train: {X_train.shape}, test: {X_test.shape}, "
        f"n_classes: {int(jnp.max(y_train)) + 1}"
    )

    train_neural_cde_classifier(
        X_train,
        y_train,
        X_test,
        y_test,
        hidden_size=args.hidden_size,
        width_size=args.width_size,
        depth=args.depth,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
