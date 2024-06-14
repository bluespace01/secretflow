#------------------------------------------------------------------------------------------
# 数据同样使用mnist数据集，单方模型这里我们只是用了切分后的Alice方数据共20000个样本
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from secretflow.utils.simulation.datasets import dataset
from matplotlib import pyplot as plt

mnist = np.load(dataset('mnist'), allow_pickle=True)
image = mnist['x_train']
label = mnist['y_train']
print(mnist.files)

#----------------------------------------------------------------------
figure = plt.figure(figsize=(20, 4))
j = 0
for example in image[:40]:
    plt.subplot(4, 10, j + 1)
    plt.imshow(example, cmap='gray', aspect='equal')
    plt.axis('off')
    j += 1
plt.show()

figure = plt.figure(figsize=(20, 4))
j = 0
for example in image[10000:10040]:
    plt.subplot(4, 10, j + 1)
    plt.imshow(example, cmap='gray', aspect='equal')
    plt.axis('off')
    j += 1
plt.show()
#----------------------------------------------------------------------


def create_model():
    num_classes = 10
    input_shape = (28, 28, 1)
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    # Compile model
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
    )
    return model

single_model = create_model()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

alice_x = image[:10000]
alice_y = label[:10000]
alice_y = OneHotEncoder(sparse_output=False).fit_transform(alice_y.reshape(-1, 1))

random_seed = 1234
alice_X_train, alice_X_test, alice_y_train, alice_y_test = train_test_split(
    alice_x, alice_y, test_size=0.1, random_state=random_seed
)

single_model.fit(
    alice_X_train,
    alice_y_train,
    validation_data=(alice_X_test, alice_y_test),
    batch_size=128,
    epochs=10,
)

