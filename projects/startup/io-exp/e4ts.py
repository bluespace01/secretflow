import secretflow as sf
import math

path_to_flower_dataset='./datasets/flower_photos/'

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address="local", log_to_driver=False)

alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')


# Steps to use DataBuilder:

# Use the single-machine version engine (TensorFlow, PyTorch) to develop and get the Builder function of the Dataset.
# Wrap the Builder functions of each party to get create_dataset_builder function. Note: The dataset_builder needs to pass in the stage parameter.
# Build the data_builder_dict [PYU, dataset_builder].
# Pass the obtained data_builder_dict to the dataset_builder of the fit function. At the same time, the x parameter position is passed into the required input in dataset_builder (eg: the input passed in this example is the actual image path used).
# Using DataBuilder in FLModel requires a pre-defined data_builder_dict. Need to be able to return tf.dataset and steps_per_epoch. And the steps_per_epoch returned by all parties must be consistent.


# data_builder_dict = {
#             alice: create_alice_dataset_builder(
#                 batch_size=32,
#             ), # create_alice_dataset_builder must return (Dataset, steps_per_epoch)
#             bob: create_bob_dataset_builder(
#                 batch_size=32,
#             ), # create_bob_dataset_builder must return (Dataset, steps_per_epochstep_per_epochs)
#         }

#------------------------------------------------------------------------------
def create_dataset_builder(batch_size=32,):
    def dataset_builder(folder_path, stage="train"):
        import math
        import tensorflow as tf

        img_height = 180
        img_width = 180
        data_set = tf.keras.utils.image_dataset_from_directory(
            folder_path,
            validation_split=0.2,
            subset="both",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )
        if stage == "train":
            train_dataset = data_set[0]
            train_step_per_epoch = math.ceil(len(data_set[0].file_paths) / batch_size)
            return train_dataset, train_step_per_epoch
        elif stage == "eval":
            eval_dataset = data_set[1]
            eval_step_per_epoch = math.ceil(len(data_set[1].file_paths) / batch_size)
            return eval_dataset, eval_step_per_epoch

    return dataset_builder




import tensorflow as tf

img_height = 180
img_width = 180
batch_size = 32
# In this example, we use the TensorFlow interface for development.
data_set = tf.keras.utils.image_dataset_from_directory(
    path_to_flower_dataset,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

train_set = data_set[0]
test_set = data_set[1]

print(type(train_set), type(test_set))

x, y = next(iter(train_set))
print(f"x.shape = {x.shape}")
print(f"y.shape = {y.shape}")


data_builder_dict = {
    alice: create_dataset_builder(
        batch_size=32,
    ),
    bob: create_dataset_builder(
        batch_size=32,
    ),
}


#------------------------------------------------------------------------------
def create_conv_flower_model(input_shape, num_classes, name='model'):
    def create_model():
        from tensorflow import keras

        # Create model

        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(num_classes),
            ]
        )
        # Compile model
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=["accuracy"],
        )
        return model

    return create_model

#------------------------------------------------------------------------------
from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator


device_list = [alice, bob]
aggregator = SecureAggregator(charlie, [alice, bob])

# prepare model
num_classes = 5
input_shape = (180, 180, 3)

# keras model
model = create_conv_flower_model(input_shape, num_classes)


fed_model = FLModel(
    device_list=device_list,
    model=model,
    aggregator=aggregator,
    backend="tensorflow",
    strategy="fed_avg_w",
    random_seed=1234,
)


data = {
    alice: path_to_flower_dataset,
    bob: path_to_flower_dataset,
}


history = fed_model.fit(
    data,
    None,
    validation_data=data,
    epochs=5,
    batch_size=32,
    aggregate_freq=2,
    sampler_method="batch",
    random_seed=1234,
    dp_spent_step_freq=1,
    dataset_builder=data_builder_dict,
)

