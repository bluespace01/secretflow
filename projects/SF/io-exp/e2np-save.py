import secretflow as sf

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address="local", log_to_driver=True)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

import numpy as np

all_data = np.load("./mnist.npz")

alice_train_x = all_data["x_train"][:30000]
alice_test_x = all_data["x_test"][:30000]
alice_train_y = all_data["y_train"][:30000]
alice_test_y = all_data["y_test"][:30000]

bob_train_x = all_data["x_train"][30000:]
bob_test_x = all_data["x_test"][30000:]
bob_train_y = all_data["y_train"][30000:]
bob_test_y = all_data["y_test"][30000:]

np.savez(
    "./alice_mnist.npz",
    train_x=alice_train_x,
    test_x=alice_test_x,
    train_y=alice_train_y,
    test_y=alice_test_y,
)
np.savez(
    "./bob_mnist.npz",
    train_x=bob_train_x,
    test_x=bob_test_x,
    train_y=bob_train_y,
    test_y=bob_test_y,
)

np.save("./alice_mnist_train_x.npy", alice_train_x)
np.save("./bob_mnist_train_x.npy", bob_train_x)

