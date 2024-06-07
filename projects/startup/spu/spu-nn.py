import sys

# !{sys.executable} -m pip install flax==0.6.0

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# ---------------------------
def breast_cancer(party_id=None, train: bool = True) -> (np.ndarray, np.ndarray):
    x, y = load_breast_cancer(return_X_y=True)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if train:
        if party_id:
            if party_id == 1:
                return x_train[:, :15], _
            else:
                return x_train[:, 15:], y_train
        else:
            return x_train, y_train
    else:
        return x_test, y_test
    


from typing import Sequence
import flax.linen as nn

FEATURES = [30, 15, 8, 1]

# ---------------------------
class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x
    

import jax.numpy as jnp


# ---------------------------
def predict(params, x):
    # TODO(junfeng): investigate why need to have a duplicated definition in notebook,
    # which is not the case in a normal python program.
    from typing import Sequence
    import flax.linen as nn

    FEATURES = [30, 15, 8, 1]

    class MLP(nn.Module):
        features: Sequence[int]

        @nn.compact
        def __call__(self, x):
            for feat in self.features[:-1]:
                x = nn.relu(nn.Dense(feat)(x))
            x = nn.Dense(self.features[-1])(x)
            return x

    return MLP(FEATURES).apply(params, x)

# ---------------------------
def loss_func(params, x, y):
    pred = predict(params, x)

    def mse(y, pred):
        def squared_error(y, y_pred):
            return jnp.multiply(y - y_pred, y - y_pred) / 2.0

        return jnp.mean(squared_error(y, pred))

    return mse(y, pred)

# ---------------------------
def train_auto_grad(x1, x2, y, params, n_batch=10, n_epochs=10, step_size=0.01):
    x = jnp.concatenate((x1, x2), axis=1)
    xs = jnp.array_split(x, len(x) / n_batch, axis=0)
    ys = jnp.array_split(y, len(y) / n_batch, axis=0)

    def body_fun(_, loop_carry):
        params = loop_carry
        for x, y in zip(xs, ys):
            _, grads = jax.value_and_grad(loss_func)(params, x, y)
            params = jax.tree_util.tree_map(
                lambda p, g: p - step_size * g, params, grads
            )
        return params

    params = jax.lax.fori_loop(0, n_epochs, body_fun, params)
    return params


# ---------------------------
def model_init(n_batch=10):
    model = MLP(FEATURES)
    return model.init(jax.random.PRNGKey(1), jnp.ones((n_batch, FEATURES[0])))

from sklearn.metrics import roc_auc_score

# ------------------------------------------------------
def validate_model(params, X_test, y_test):
    y_pred = predict(params, X_test)
    return roc_auc_score(y_test, y_pred)


# ------------------------------------------------------
import jax

# Load the data
x1, _ = breast_cancer(party_id=1, train=True)
x2, y = breast_cancer(party_id=2, train=True)


# Hyperparameter
n_batch = 10
n_epochs = 10
step_size = 0.01


# Train the model
init_params = model_init(n_batch)
params = train_auto_grad(x1, x2, y, init_params, n_batch, n_epochs, step_size)

# Test the model
X_test, y_test = breast_cancer(train=False)
auc = validate_model(params, X_test, y_test)
print(f'auc={auc}')



# ------------------------------------------------------
import secretflow as sf

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(['alice', 'bob'], address='local')

alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

x1, _ = alice(breast_cancer)(party_id=1, train=True)
x2, y = bob(breast_cancer)(party_id=2, train=True)
init_params = model_init(n_batch)


device = spu
x1_, x2_, y_ = x1.to(device), x2.to(device), y.to(device)
init_params_ = sf.to(alice, init_params).to(device)

params_spu = spu(train_auto_grad, static_argnames=['n_batch', 'n_epochs', 'step_size'])(
    x1_, x2_, y_, init_params_, n_batch=n_batch, n_epochs=n_epochs, step_size=step_size
)


params_spu = spu(train_auto_grad)(x1_, x2_, y_, init_params)
params = sf.reveal(params_spu)
print(params)


X_test, y_test = breast_cancer(train=False)
auc = validate_model(params, X_test, y_test)
print(f'auc={auc}')


