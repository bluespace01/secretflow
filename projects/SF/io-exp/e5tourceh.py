import secretflow as sf
import math

path_to_flower_dataset='./datasets/flower_photos/'

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address="local", log_to_driver=False)

alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')


# 1. Develop DataBuilder using a single-machine engine.
# In the development of the DataBuilder, we are free to follow the logic of single-machine development. The objective is to construct a Dataloader object in Torch.

import math

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

path_to_flower_dataset='./datasets/flower_photos/'

# parameter
batch_size = 32
shuffle = True
random_seed = 1234
train_split = 0.8

# Define dataset
flower_transform = transforms.Compose(
    [
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
    ]
)
flower_dataset = datasets.ImageFolder(
    path_to_flower_dataset, transform=flower_transform
)
dataset_size = len(flower_dataset)
# Define sampler

indices = list(range(dataset_size))
if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
split = int(np.floor(train_split * dataset_size))
train_indices, val_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Define databuilder
train_loader = DataLoader(flower_dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(flower_dataset, batch_size=batch_size, sampler=valid_sampler)

x, y = next(iter(train_loader))
print(f"x.shape = {x.shape}")
print(f"y.shape = {y.shape}")


# 2. Wrap the developed DataBuilder.
# The DataBuilder we have developed needs to be distributed and executed on various computing machines during runtime. To facilitate serialization, we need to wrap them.

# It is essential to consider the following points:

# FLModel requires that the input to DataBuilder must include the stage parameter (stage=“train”).
# FLModel requires that the passed DataBuilder must return two results, namely, data_set and steps_per_epoch.

def create_dataset_builder(
    batch_size=32,
    train_split=0.8,
    shuffle=True,
    random_seed=1234,
):
    def dataset_builder(x, stage="train"):
        """ """
        import math

        import numpy as np
        from torch.utils.data import DataLoader
        from torch.utils.data.sampler import SubsetRandomSampler
        from torchvision import datasets, transforms

        # Define dataset
        flower_transform = transforms.Compose(
            [
                transforms.Resize((180, 180)),
                transforms.ToTensor(),
            ]
        )
        flower_dataset = datasets.ImageFolder(x, transform=flower_transform)
        dataset_size = len(flower_dataset)
        # Define sampler

        indices = list(range(dataset_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        split = int(np.floor(train_split * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # Define databuilder
        train_loader = DataLoader(
            flower_dataset, batch_size=batch_size, sampler=train_sampler
        )
        valid_loader = DataLoader(
            flower_dataset, batch_size=batch_size, sampler=valid_sampler
        )

        # Return
        if stage == "train":
            train_step_per_epoch = math.ceil(split / batch_size)

            return train_loader, train_step_per_epoch
        elif stage == "eval":
            eval_step_per_epoch = math.ceil((dataset_size - split) / batch_size)
            return valid_loader, eval_step_per_epoch

    return dataset_builder

# 3. Construct the dataset_builder_dict.

# prepare dataset dict
data_builder_dict = {
    alice: create_dataset_builder(
        batch_size=32,
        train_split=0.8,
        shuffle=False,
        random_seed=1234,
    ),
    bob: create_dataset_builder(
        batch_size=32,
        train_split=0.8,
        shuffle=False,
        random_seed=1234,
    ),
}


# 4. Once we obtain the dataset_builder_dict, we can proceed with federated training using it.
# Next, we define a FLModel with a Torch backend for training.

# Define the Model Architecture

from torch import nn

from secretflow.ml.nn.core.torch import BaseModule


class ConvRGBNet(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 45 * 45, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, xb):
        return self.network(xb)
    

from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SecureAggregator
from torch import nn, optim
from torchmetrics import Accuracy, Precision
from secretflow.ml.nn.core.torch import metric_wrapper, optim_wrapper, TorchModel


device_list = [alice, bob]
aggregator = SecureAggregator(charlie, [alice, bob])
# prepare model
num_classes = 5

input_shape = (180, 180, 3)
# torch model
loss_fn = nn.CrossEntropyLoss
optim_fn = optim_wrapper(optim.Adam, lr=1e-3)
model_def = TorchModel(
    model_fn=ConvRGBNet,
    loss_fn=loss_fn,
    optim_fn=optim_fn,
    metrics=[
        metric_wrapper(
            Accuracy, task="multiclass", num_classes=num_classes, average='micro'
        ),
        metric_wrapper(
            Precision, task="multiclass", num_classes=num_classes, average='micro'
        ),
    ],
)

fed_model = FLModel(
    device_list=device_list,
    model=model_def,
    aggregator=aggregator,
    backend="torch",
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

