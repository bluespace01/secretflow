# step 1: ray setup
#
# ray start --head --node-ip-address="192.168.10.111" --port="9001" --include-dashboard=False --disable-usage-stats

import secretflow as sf
import numpy as np
import spu

cluster_config ={
    'nodes': [
        {
            'party': 'alice',
            'address': '192.168.10.111:9002',
            'listen_addr': '0.0.0.0:9002'
        },
        {
            'party': 'bob',
            'address': '192.168.10.112:9002',
            'listen_addr': '0.0.0.0:9002'
        }
    ],
    'self_party': 'alice',
    'parties': {
        'alice': {
            'address': '192.168.10.111:9002',
            'listen_addr': '0.0.0.0:9002'
        },
        'bob': {
            'address': '192.168.10.112:9002',
            'listen_addr': '0.0.0.0:9002'
        }
    },
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
    }
}

sf.init(address='192.168.10.111:9001', cluster_config=cluster_config)

# 创建数据代理设备
alice = sf.PYU('alice')
bob = sf.PYU('bob')

# 在Alice和Bob上创建数据
data_alice = alice(lambda: np.array([5]))
data_bob = bob(lambda: np.array([7]))

result = alice(lambda x, y: x + y, data_alice, data_bob.to(alice))

print(alice(lambda x: x, result))