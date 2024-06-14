
# 提示
# 请使用主节点的 node-ip-address 和 port 填充 sf.init 的 address 参数。
# alice 的 address 请填写可以被bob访通的地址，并且选择一个 未被占用的端口 ，注意不要和Ray端口冲突。
# alice 的 listen_addr 可以和alice address里的端口一样。
# bob 的 address 请填写可以被alice访通的地址，并且选择一个 未被占用的端口 ，注意不要和Ray端口冲突。
# bob 的 listen_addr 可以和bob address里的端口一样。

import secretflow as sf
import numpy as np
import spu

# Use ray head adress please.
# sf.init(parties=['alice', 'bob'], address='Ray head node address')
sf.init(parties=['alice', 'bob'], address='192.168.10.111:9001')

cluster_def={
    'nodes': [
        {
            'party': 'alice',
            # Please choose an unused port.
            # 'address': 'ip:port of alice',
            # 'listen_addr': '0.0.0.0:port'
            'address': '192.168.10.111:9002',
            'listen_addr': '0.0.0.0:9002'
        },
        {
            'party': 'bob',
            # Please choose an unused port.
            # 'address': 'ip:port of bob',
            # 'listen_addr': '0.0.0.0:port'
            'address': '192.168.10.112:9002',
            'listen_addr': '0.0.0.0:9002'
        },
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
    }
}

# 创建 SPU 实例
spu_instance  = sf.SPU(cluster_def=cluster_def)

# 定义 PYU
alice_pyu = sf.PYU('alice')
bob_pyu = sf.PYU('bob')

# 定义秘密共享的数据
alice_data = np.array([1, 2, 3], dtype=np.float32)
bob_data = np.array([4, 5, 6], dtype=np.float32)

# 将数据放入 PYU
alice_tensor_pyu = sf.to(alice_pyu, alice_data)
bob_tensor_pyu = sf.to(bob_pyu, bob_data)

# 将数据从 PYU 移动到 SPU
alice_tensor_spu = alice_tensor_pyu.to(spu_instance)
bob_tensor_spu = bob_tensor_pyu.to(spu_instance)


# 进行秘密共享计算
# 使用 SPU API 进行加法操作
result_spu = spu_instance(lambda x, y: x + y)(alice_tensor_spu, bob_tensor_spu)

# 将结果从 SPU 移动回 PYU
result_pyu = result_spu.to(alice_pyu)  # 假设 alice_pyu 是结果接收方

# 获取结果
result = sf.reveal(result_pyu)
print("------------------------------------------")
print("Result of the secure computation:", result)
print("------------------------------------------")