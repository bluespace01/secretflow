import secretflow as sf

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(['alice', 'bob', 'carol'], address='local')

import numpy as np
from sklearn.datasets import load_iris

data, target = load_iris(return_X_y=True, as_frame=True)
data['uid'] = np.arange(len(data)).astype('str')
data['month'] = ['Jan'] * 75 + ['Feb'] * 75

print(data)

import os

os.makedirs('.data', exist_ok=True)
da, db, dc = data.sample(frac=0.9), data.sample(frac=0.8), data.sample(frac=0.7)

da.to_csv('.data/alice.csv', index=False)
db.to_csv('.data/bob.csv', index=False)
dc.to_csv('.data/carol.csv', index=False)

alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

input_path = {alice: '.data/alice.csv', bob: '.data/bob.csv'}
output_path = {alice: '.data/alice_psi.csv', bob: '.data/bob_psi.csv'}
spu.psi_csv('uid', input_path, output_path, 'alice')

# verification
import pandas as pd

df = da.join(db.set_index('uid'), on='uid', how='inner', rsuffix='_bob', sort=True)
expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

da_psi = pd.read_csv('.data/alice_psi.csv')
db_psi = pd.read_csv('.data/bob_psi.csv')

pd.testing.assert_frame_equal(da_psi, expected)
pd.testing.assert_frame_equal(db_psi, expected)

print(da_psi)


#----------------------------------------------------------------------
spu.psi_csv(['uid', 'month'], input_path, output_path, 'alice')
df = da.join(
    db.set_index(['uid', 'month']),
    on=['uid', 'month'],
    how='inner',
    rsuffix='_bob',
    sort=True,
)
expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

da_psi = pd.read_csv('.data/alice_psi.csv')
db_psi = pd.read_csv('.data/bob_psi.csv')

pd.testing.assert_frame_equal(da_psi, expected)
pd.testing.assert_frame_equal(db_psi, expected)


#----------------------------------------------------------------------


carol = sf.PYU('carol')
spu_3pc = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

input_path = {alice: '.data/alice.csv', bob: '.data/bob.csv', carol: '.data/carol.csv'}
output_path = {
    alice: '.data/alice_psi.csv',
    bob: '.data/bob_psi.csv',
    carol: '.data/carol_psi.csv',
}
spu_3pc.psi_csv(
    ['uid', 'month'], input_path, output_path, 'alice', protocol='ECDH_PSI_3PC'
)

keys = ['uid', 'month']
df = da.join(db.set_index(keys), on=keys, how='inner', rsuffix='_bob', sort=True).join(
    dc.set_index(keys), on=keys, how='inner', rsuffix='_carol', sort=True
)
expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

da_psi = pd.read_csv('.data/alice_psi.csv')
db_psi = pd.read_csv('.data/bob_psi.csv')
dc_psi = pd.read_csv('.data/carol_psi.csv')

pd.testing.assert_frame_equal(da_psi, expected)
pd.testing.assert_frame_equal(db_psi, expected)
pd.testing.assert_frame_equal(dc_psi, expected)






