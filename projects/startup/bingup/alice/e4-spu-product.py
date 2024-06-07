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


# 创建SPU对象
spu = sf.SPU(cluster_def=cluster_config)

# 准备要计算的数据
data_alice = np.array([5])
data_bob = np.array([7])


# 发送数据给Bob
encrypted_data = spu.send('bob', data_alice)

# 接收Bob发送的数据
data_from_bob = spu.recv('bob')

# 计算加法结果
result = data_from_bob + data_bob

print("Alice 计算的结果:", result)