import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

import diffrax
import equinox as eqx


class NeuralCDE(eqx.Module):
    """
    Neural CDE classifier with dh_t = f(h_t) d x_t and linear readout.

    Equation (per sample):

        Given a path x : [0, 1] -> R^{d_x} on a uniform grid,

            h_0   = g_theta(x_0),
            d h_t = f_theta(h_t) d x_t,
            logits = W h_1 + b.

        Here:
          - g_theta: R^{d_x} -> R^{d_h} is an MLP (initial hidden state),
          - f_theta: R^{d_h} -> R^{d_h x d_x} is an MLP (vector field),
          - W, b define the linear readout to n_classes.
    """

    initial: eqx.nn.MLP
    f_mlp: eqx.nn.MLP
    readout: eqx.nn.Linear

    data_size: int      # d_x (feature dimension)
    hidden_size: int    # d_h

    def __init__(
        self,
        data_size: int,      # number of channels per time step (feature dim)
        hidden_size: int,    # hidden dimension
        n_classes: int,
        width_size: int,
        depth: int,
        *,
        key,
        **kwargs,
    ):
        
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size

        k_init, k_f, k_readout = jr.split(key, 3)

        # g_theta: x_0 -> h_0
        self.initial = eqx.nn.MLP(
            in_size=data_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=k_init,
        )

        # f_theta: h_t -> R^{hidden_size x data_size}
        self.f_mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            # small-ish final activation to keep vector field from exploding
            final_activation=jnn.tanh,
            key=k_f,
        )

        # Linear readout W h_1 + b
        self.readout = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=n_classes,
            key=k_readout,
        )

    def _vector_field(self, t, h, args):
        """
        f_theta(h_t) in R^{hidden_size x data_size}.
        """
        # h: (hidden_size,), ignore t and args
        out = self.f_mlp(h)  # (hidden_size * data_size,)
        return out.reshape(self.hidden_size, self.data_size)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for a single path.

        Args:
            x: (length, data_size) path values x_t,
               sampled on a uniform grid in [0, 1].

        Returns:
            logits: (n_classes,) array.
        """
        if x.ndim != 2:
            raise ValueError("x must be 2D of shape (length, data_size)")
        
        length, d = x.shape
        if d != self.data_size:
            raise ValueError(
                f"Expected x.shape[1] = {self.data_size}, got {d}"
            )

        # Uniform time grid in [0, 1]
        ts = jnp.linspace(0.0, 1.0, length)

        # Interpolation of the input path x_t
        control = diffrax.LinearInterpolation(ts, x)

        # Initial hidden state h_0 = g_theta(x_0)
        x0 = control.evaluate(ts[0])          # (data_size,)
        h0 = self.initial(x0)                 # (hidden_size,)

        # dh_t = f(h_t) d x_t  implemented via ControlTerm
        term = diffrax.ControlTerm(self._vector_field, control).to_ode()
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(t1=True)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=h0,
            saveat=saveat,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        )

        hT = sol.ys[-1]     # (hidden_size,)
        logits = self.readout(hT)
        return logits


def count_parameters(model: eqx.Module) -> int:
    """
    Approximate total number of trainable parameters in an Equinox model.
    """
    leaves = jax.tree_leaves(eqx.filter(model, eqx.is_array))
    return int(sum(jnp.size(p) for p in leaves))


# ================================================================
# Classification: cross-entropy + accuracy
# ================================================================

def loss_and_accuracy(model: NeuralCDE, X: jnp.ndarray, y: jnp.ndarray):
    """
    Compute cross-entropy loss and accuracy on a batch (classification).

    X: (N, L, C)  time series batch
    y: (N,)       integer labels in {0, ..., n_classes-1}
    """
    # vmap over the batch; each call integrates one CDE
    logits = jax.vmap(model)(X)          # (N, n_classes)
    log_probs = jnn.log_softmax(logits, axis=-1)
    nll = -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])

    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == y)
    return nll, acc


loss_and_accuracy_grad = eqx.filter_value_and_grad(loss_and_accuracy, has_aux=True)

@eqx.filter_jit
def train_step_classification(model, opt_state, X, y, optim):
    """
    One gradient step for classification (cross-entropy).
    """
    (loss, acc), grads = loss_and_accuracy_grad(model, X, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, acc


def evaluate_classification(model, X, y):
    """
    Evaluate classification loss and accuracy.
    """
    loss, acc = loss_and_accuracy(model, X, y)
    return float(loss), float(acc)


# ================================================================
# Regression / Forecasting: MSE
# ================================================================

def loss_mse(model: NeuralCDE, X: jnp.ndarray, y: jnp.ndarray):
    """
    Compute mean squared error loss on a batch (regression / forecasting).

    X: (N, L, C)  time series batch
    y: (N,) or (N, D)  regression targets

    Typically you'll set n_classes = 1 so model(X) returns shape (1,)
    and y has shape (N,) (broadcasting will work).
    """
    preds = jax.vmap(model)(X)  # (N, out_dim)

    # If out_dim == 1 and y is (N,), broadcasting handles it:
    diff = preds - y
    mse = jnp.mean(diff ** 2)
    return mse


loss_mse_grad = eqx.filter_value_and_grad(loss_mse, has_aux=False)

@eqx.filter_jit
def train_step_regression(model, opt_state, X, y, optim):
    """
    One gradient step for regression (MSE).
    """
    loss, grads = loss_mse_grad(model, X, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def evaluate_regression(model, X, y):
    """
    Evaluate MSE on regression / forecasting task.
    """
    loss = loss_mse(model, X, y)
    return float(loss)




# optim = optax.adam(1e-3)
# opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

# for step in range(steps):
#     model, opt_state, loss, acc = train_step_classification(model, opt_state, X_train, y_train, optim)

# train_loss, train_acc = evaluate_classification(model, X_train, y_train)
# test_loss, test_acc   = evaluate_classification(model, X_test,  y_test)


# # y_train, y_test can be (N,) or (N, 1) or (N, D)
# optim = optax.adam(1e-3)
# opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

# for step in range(steps):
#     model, opt_state, loss = train_step_regression(model, opt_state, X_train, y_train, optim)

# train_mse = evaluate_regression(model, X_train, y_train)
# test_mse  = evaluate_regression(model, X_test,  y_test)