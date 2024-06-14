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

# 创建数据代理设备
alice = sf.PYU('alice')
bob = sf.PYU('bob')

# 创建Bob的数据，这里简单示例为数组中元素7
data_bob = bob(lambda: np.array([7]))

# 假设有来自Alice的数据请求处理（这部分需要与Alice节点的代码进行交互和同步）
# 注意：此处代码假设Alice已发送数据到Bob，Bob进行计算后返回结果。
# 实际应用中需要确保网络通信和数据同步的正确性。

# 临时代码，仅用于示例，没有实际操作
print("Bob's data ready for operations.")

# 此处可以添加具体的数学操作或数据处理任务



