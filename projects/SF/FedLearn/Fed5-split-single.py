from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split

import pandas as pd
from secretflow.utils.simulation.datasets import dataset

df = pd.read_csv(dataset('bank_marketing'), sep=';')
alice_data = df[["age", "job", "marital", "education", "y"]]


def create_model():
    model = keras.Sequential(
        [
            keras.Input(shape=4),
            layers.Dense(100, activation="relu"),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ]
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    return model


single_model = create_model()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
single_part_data = alice_data.copy()

single_part_data['job'] = encoder.fit_transform(alice_data['job'])
single_part_data['marital'] = encoder.fit_transform(alice_data['marital'])
single_part_data['education'] = encoder.fit_transform(alice_data['education'])
single_part_data['y'] = encoder.fit_transform(alice_data['y'])

y = single_part_data['y']
alice_data = single_part_data.drop(columns=['y'], inplace=False)

scaler = MinMaxScaler()
alice_data = scaler.fit_transform(alice_data)

random_state = 1234
train_data, test_data = train_test_split(
    alice_data, train_size=0.8, random_state=random_state
)
train_label, test_label = train_test_split(y, train_size=0.8, random_state=random_state)

single_model.fit(
    train_data,
    train_label,
    validation_data=(test_data, test_label),
    batch_size=128,
    epochs=10,
    shuffle=False,
)

single_model.evaluate(test_data, test_label, batch_size=128)
