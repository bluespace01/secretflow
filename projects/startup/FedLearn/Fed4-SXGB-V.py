import spu
from sklearn.metrics import roc_auc_score

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ml.boost.sgb_v import (
    Sgb,
    get_classic_XGB_params,
    get_classic_lightGBM_params,
)
from secretflow.ml.boost.sgb_v.model import load_model
import pprint

pp = pprint.PrettyPrinter(depth=4)

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))


alice_ip = '127.0.0.1'
bob_ip = '127.0.0.1'
ip_party_map = {bob_ip: 'bob', alice_ip: 'alice'}

_system_config = {'lineage_pinning_enabled': False}
sf.shutdown()
# init cluster
sf.init(
    ['alice', 'bob'],
    address='local',
    _system_config=_system_config,
    object_store_memory=5 * 1024 * 1024 * 1024,
)

# SPU settings
cluster_def = {
    'nodes': [
        {'party': 'alice', 'id': 'local:0', 'address': alice_ip + ':12945'},
        {'party': 'bob', 'id': 'local:1', 'address': bob_ip + ':12946'},
        # {'party': 'carol', 'id': 'local:2', 'address': '127.0.0.1:12347'},
    ],
    'runtime_config': {
        # SEMI2K support 2/3 PC, ABY3 only support 3PC, CHEETAH only support 2PC.
        # pls pay attention to size of nodes above. nodes size need match to PC setting.
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
    },
}

# HEU settings
heu_config = {
    'sk_keeper': {'party': 'alice'},
    'evaluators': [{'party': 'bob'}],
    'mode': 'PHEU',
    'he_parameters': {
        # ou is a fast encryption schema that is as secure as paillier.
        'schema': 'ou',
        'key_pair': {
            'generate': {
                # bit size should be 2048 to provide sufficient security.
                'bit_size': 2048,
            },
        },
    },
    'encoding': {
        'cleartext_type': 'DT_I32',
        'encoder': "IntegerEncoder",
        'encoder_args': {"scale": 1},
    },
}


alice = sf.PYU('alice')
bob = sf.PYU('bob')
heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])

from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer()
x, y = ds['data'], ds['target']

alice_data = lambda: x[:, :15]
bob_data = lambda: x[:, 15:]


v_data = FedNdarray(
    {
        alice: (alice(alice_data)()),
        bob: (bob(bob_data)()),
    },
    partition_way=PartitionWay.VERTICAL,
)
label_data = FedNdarray(
    {alice: (alice(lambda: y)())},
    partition_way=PartitionWay.VERTICAL,
)

params = get_classic_XGB_params()
params['num_boost_round'] = 3
params['max_depth'] = 3
pp.pprint(params)

sgb = Sgb(heu)
model = sgb.train(params, v_data, label_data)

yhat = model.predict(v_data)
yhat = reveal(yhat)
print(f"auc: {roc_auc_score(y, yhat)}")

# each participant party needs a location to store
saving_path_dict = {
    # in production we may use remote oss, for example.
    device: "./" + device.party
    for device in v_data.partitions.keys()
}

r = model.save_model(saving_path_dict)
wait(r)

# alice is our label holder
model_loaded = load_model(saving_path_dict, alice)
fed_yhat_loaded = model_loaded.predict(v_data, alice)
yhat_loaded = reveal(fed_yhat_loaded.partitions[alice])

assert (
    yhat == yhat_loaded
).all(), "loaded model predictions should match original, yhat {} vs yhat_loaded {}".format(
    yhat, yhat_loaded
)



params = get_classic_lightGBM_params()
params['num_boost_round'] = 3
params['max_leaf'] = 2**3
pp.pprint(params)
model = sgb.train(params, v_data, label_data)


yhat = model.predict(v_data)
yhat = reveal(yhat)
print(f"auc: {roc_auc_score(y, yhat)}")

