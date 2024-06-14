# step 1: ray setup
#
# ray start --head --node-ip-address="192.168.10.112" --port="9001" --include-dashboard=False --disable-usage-stats

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
    'self_party': 'bob',
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

sf.init(address='192.168.10.112:9001', cluster_config=cluster_config)

# 创建SPU对象
spu = sf.SPU(cluster_def=cluster_config)

# 接收Alice发送的加密数据
encrypted_data = spu.recv('alice')

# 解密数据
decrypted_data = spu.decrypt(encrypted_data)

print("Bob 收到并解密的数据:", decrypted_data)


