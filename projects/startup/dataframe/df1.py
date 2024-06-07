import secretflow as sf

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(['alice', 'bob'], address='local')
alice = sf.PYU('alice')
bob = sf.PYU('bob')



import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
data = pd.concat([iris.data, iris.target], axis=1)

# In order to facilitate the subsequent display,
# here we first set some data to None.
data.iloc[1, 1] = None
data.iloc[100, 1] = None

# Restore target to its original name.
data['target'] = data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print(data)




import tempfile
from secretflow.data.vertical import read_csv as v_read_csv

# Vertical partitioning.
v_alice, v_bob = data.iloc[:, :2], data.iloc[:, 2:]

# Save to temprary files.
_, alice_path = tempfile.mkstemp()
_, bob_path = tempfile.mkstemp()
v_alice.to_csv(alice_path, index=False)
v_bob.to_csv(bob_path, index=False)


df = v_read_csv({alice: alice_path, bob: bob_path})

# Before filling, the sepal width (cm) is missing in two positions.
df.count()['sepal width (cm)']

# Fill sepal width (cm) with 10.
df.fillna(value={'sepal width (cm)': 10}).count()['sepal width (cm)']


# Scaling features to a range
# Secretflow provides MinMaxScaler for scaling features to lie between a given minimum and maximum value. The input and output of MinMaxScaler are both DataFrame.
from secretflow.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_sepal_len = scaler.fit_transform(df['sepal length (cm)'])

print('Min: ', scaled_sepal_len.min())
print('Max: ', scaled_sepal_len.max())

# Variance scaling
# Secretflow provides StandardScaler for variance scaling. The input and output of StandardScaler are both DataFrames.
from secretflow.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_sepal_len = scaler.fit_transform(df['sepal length (cm)'])

print('Min: ', scaled_sepal_len.min())
print('Max: ', scaled_sepal_len.max())

# OneHot encoding
# Secretflow provides OneHotEncoder for OneHot encoding. The input and output of OneHotEncoder are DataFrame.
from secretflow.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder()
onehot_target = onehot_encoder.fit_transform(df['target'])

print('Columns: ', onehot_target.columns)
print('Min: \n', onehot_target.min())
print('Max: \n', onehot_target.max())

# Label encoding
# secretflow provides LabelEncoder for encoding target labels with value between 0 and n_classes-1. The input and output of LabelEncoder are DataFrame.
from secretflow.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_label = label_encoder.fit_transform(df['target'])

print('Columns: ', encoded_label.columns)
print('Min: \n', encoded_label.min())
print('Max: \n', encoded_label.max())


# Discretization
# SecretFlow provides KBinsDiscretizer for partitioning continuous features into discrete values. The input and output of KBinsDiscretizer are both DataFrame.
from secretflow.preprocessing import KBinsDiscretizer

estimator = KBinsDiscretizer(n_bins=5)
binned_petal_len = estimator.fit_transform(df['petal length (cm)'])

print('Min: \n', binned_petal_len.min())
print('Max: \n', binned_petal_len.max())


# WOE encoding
# secretflow provides VertWoeBinning to bin the features into buckets by quantile or chimerge method, and calculate the woe value and iv value in each bucket. And VertBinSubstitution can substitute the features with the woe value.

# woe binning use SPU or HEU device to protect label
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

# Only support binary classification label dataset for now.
# use linear dataset as example
from secretflow.utils.simulation.datasets import load_linear

vdf = load_linear(parts={alice: (1, 4), bob: (18, 22)})
print(f"orig ds in alice:\n {sf.reveal(vdf.partitions[alice].data)}")
print(f"orig ds in bob:\n {sf.reveal(vdf.partitions[bob].data)}")

from secretflow.preprocessing.binning.vert_woe_binning import VertWoeBinning

binning = VertWoeBinning(spu)
bin_rules = binning.binning(
    vdf,
    binning_method="quantile",
    bin_num=5,
    bin_names={alice: ["x1", "x2", "x3"], bob: ["x18", "x19", "x20"]},
    label_name="y",
)

print(f"bin_rules for alice:\n {sf.reveal(bin_rules[alice])}")
print(f"bin_rules for bob:\n {sf.reveal(bin_rules[bob])}")

from secretflow.preprocessing.binning.vert_bin_substitution import VertBinSubstitution

woe_sub = VertBinSubstitution()
sub_data = woe_sub.substitution(vdf, bin_rules)

print(f"substituted ds in alice:\n {sf.reveal(sub_data.partitions[alice].data)}")
print(f"substituted ds in bob:\n {sf.reveal(sub_data.partitions[bob].data)}")
