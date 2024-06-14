# Load the Dataset
# We are going to split the whole dataset into train and test subsets after normalization with breast_cancer. 
# * if train is True, returns train subsets. In order to simulate training with vertical dataset splitting, the party_id is provided. 
# * else, returns test subsets.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

#-----------------------------------------------------------
def breast_cancer(party_id=None, train: bool = True) -> (np.ndarray, np.ndarray):
    x, y = load_breast_cancer(return_X_y=True)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if train:
        if party_id:
            if party_id == 1:
                return x_train[:, :15], None
            else:
                return x_train[:, 15:], y_train
        else:
            return x_train, y_train
    else:
        return x_test, y_test
    
#---------------------------------------------------------------------
# First, let’s define the loss function, which is a negative log-likelihood in our case.
import jax.numpy as jnp

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

#---------------------------------------------------------------------
# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)

#-----------------------------------------------------------
# Training loss is the negative log-likelihood of the training examples.
def loss(W, b, inputs, targets):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    loss = -jnp.mean(jnp.log(label_probs))
    print(loss.primal)
    return loss

#---------------------------------------------------------------------
# Second, let’s define a single train step with SGD optimizer. 
# Just to remind you, x1 represents 15 features from one party while x2 represents the other 15 features from the other party.
from jax import grad

def train_step(W, b, x1, x2, y, learning_rate):
    x = jnp.concatenate([x1, x2], axis=1)
    Wb_grad = grad(loss, (0, 1))(W, b, x, y)
    W -= learning_rate * Wb_grad[0]
    b -= learning_rate * Wb_grad[1]
    return W, b

#---------------------------------------------------------------------
# Last, let’s build everything together as a fit method which returns the model and losses of each epoch.
def fit(W, b, x1, x2, y, epochs=1, learning_rate=1e-2):
    for _ in range(epochs):
        W, b = train_step(W, b, x1, x2, y, learning_rate=learning_rate)
    return W, b

#---------------------------------------------------------------------
# Validate the Model
# We could use the AUC to validate a binary classification model.
from sklearn.metrics import roc_auc_score

def validate_model(W, b, X_test, y_test):
    y_pred = predict(W, b, X_test)
    return roc_auc_score(y_test, y_pred)

#---------------------------------------------------------------------
# Let’s put everything we have together and train a LR model!
# %matplotlib inline

# Load the data
x1, _ = breast_cancer(party_id=1, train=True)
x2, y = breast_cancer(party_id=2, train=True)

# Hyperparameter
W = jnp.zeros((30,))
b = 0.0
epochs = 100
learning_rate = 1e-2

# Train the model
W, b = fit(W, b, x1, x2, y, epochs=epochs, learning_rate=learning_rate)

# Validate the model
X_test, y_test = breast_cancer(train=False)
auc = validate_model(W, b, X_test, y_test)
print(f'auc={auc}')

#===============================================================================================
# Train a Model with SPU
# Init the Environment

import secretflow as sf

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(['alice', 'bob'], address='local')

alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Load the Dataset
x1, _ = alice(breast_cancer)(party_id=1)
x2, y = bob(breast_cancer)(party_id=2)

# x1, x2, y

# Before training, we need to pass hyperparamters and all data to SPU device. 
# SecretFlow provides two methods: - secretflow.to: transfer a PythonObject or 
# DeviceObject to a specific device. - DeviceObject.to: transfer the DeviceObject to a specific device.

device = spu

W = jnp.zeros((30,))
b = 0.0

W_, b_, x1_, x2_, y_ = (
    sf.to(alice, W).to(device),
    sf.to(bob, b).to(device),
    x1.to(device),
    x2.to(device),
    y.to(device),
)

W_, b_ = device(
    fit,
    static_argnames=['epochs'],
    num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_USER,
    user_specified_num_returns=2,
)(W_, b_, x1_, x2_, y_, epochs=10, learning_rate=1e-2)

print(sf.reveal(W_))
print(sf.reveal(b_))

auc = validate_model(sf.reveal(W_), sf.reveal(b_), X_test, y_test)
print(f'auc={auc}')
