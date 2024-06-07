# Loading npz files
alice_path = "./alice_mnist.npz"
bob_path = "./bob_mnist.npz"

from secretflow.data.ndarray import load
from secretflow.data.split import train_test_split

fed_npz = load({'alice': alice_path, 'bob': bob_path}, allow_pickle=True)
print(fed_npz)
print(type(fed_npz["train_x"]))

alice_path = "./alice_mnist_train_x.npy"
bob_path = "./bob_mnist_train_x.npy"

fed_ndarray = load({alice: alice_path, bob: bob_path}, allow_pickle=True)

print(type(fed_ndarray))
