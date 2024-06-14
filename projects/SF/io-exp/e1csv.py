import secretflow as sf

# Check the version of your SecretFlow
# print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'charlie'], address="local", log_to_driver=True)
alice, bob, charlie = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('charlie')

import pandas as pd

alldata_df = pd.read_csv("./iris.csv")

print(len(alldata_df))

h_alice_df = alldata_df.loc[:70]
h_bob_df = alldata_df.loc[70:]

# save the data to local file system
import tempfile

_, h_alice_path = tempfile.mkstemp()
_, h_bob_path = tempfile.mkstemp()
h_alice_df.to_csv(h_alice_path, index=False)
h_bob_df.to_csv(h_bob_path, index=False)

v_alice_df = alldata_df.loc[:, ['sepal_length', 'sepal_width']]
v_bob_df = alldata_df.loc[:, ['petal_length', 'petal_width', 'class']]

# save the data to local file system
_, v_alice_path = tempfile.mkstemp()
_, v_bob_path = tempfile.mkstemp()
v_alice_df.to_csv(v_alice_path, index=True, index_label="id")
v_bob_df.to_csv(v_bob_path, index=True, index_label="id")


# Loading CSV Data Example for Horizontal Scenario
from secretflow.data.horizontal import read_csv
from secretflow.security.aggregation.plain_aggregator import PlainAggregator
from secretflow.security.compare.plain_comparator import PlainComparator
from secretflow.data.split import train_test_split

path_dict = {alice: h_alice_path, bob: h_bob_path}

aggregator = PlainAggregator(charlie)
comparator = PlainComparator(charlie)

hdf = read_csv(filepath=path_dict, aggregator=aggregator, comparator=comparator)

print(hdf.columns)


label = hdf["class"]
data = hdf.drop(columns="class")
    
train_data, test_data = train_test_split(
    data, train_size=0.8, shuffle=True, random_state=1234
)

print(train_data.partition_shape(), test_data.partition_shape())


#Loading CSV Data Example for Vertical Scenario
from secretflow.data.vertical import read_csv

path_dict = {
    alice: v_alice_path,  # The path that alice can access
    bob: v_bob_path,  # The path that bob can access
}

# Prepare the SPU device
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

vdf = read_csv(path_dict, spu=spu, keys='id', drop_keys="id")

print(vdf.columns)

label = vdf["class"]
data = vdf.drop(columns="class")

train_data, test_data = train_test_split(
    data, train_size=0.8, shuffle=True, random_state=1234
)

print(train_data.partition_shape(), test_data.partition_shape())
