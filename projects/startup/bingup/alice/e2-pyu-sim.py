
# step 1
#   condition for ray master setup.
# 
# alice: 
#   ray start --head --node-ip-address="192.168.10.111" --port="9001" --resources='{"alice": 16}' --include-dashboard=False --disable-usage-stats
#   
# bob:
#   ray start --address="192.168.10.111:9001" --resources='{"bob": 16}' --disable-usage-stats

#
#
# 解释这个命令的各个部分：

# ray start --head：1
# 启动 Ray 集群的主节点。只有集群中的第一个节点需要使用这个参数，其他节点将连接到这个主节点。

# --node-ip-address="192.168.10.111"：
# 指定主节点的 IP 地址。这是 Ray 主节点在网络中的 IP 地址。

# --port="9001"：
# 指定 Ray 主节点的监听端口。这个端口将用于集群中其他节点与主节点通信。

# --resources='{"alice": 16}'：
# 为主节点分配自定义资源。这里将资源类型 alice 分配 16 个单位。

# --include-dashboard=False：
# 禁用 Ray 仪表板。如果你不需要使用 Ray 提供的 Web 仪表板，可以通过这个选项禁用它。

# --disable-usage-stats：
# 禁用 Ray 的使用统计报告，防止数据收集。



# step 2
import secretflow as sf
# Replace with the `node-ip-address` and `port` of head node.
sf.init(parties=['alice', 'bob'], address='192.168.10.111:9001')
alice = sf.PYU('alice')
bob = sf.PYU('bob')
alice(lambda x : x)(2)
# <secretflow.device.device.pyu.PYUObject object at 0x7fe932a1a640>
bob(lambda x : x)(2)
# <secretflow.device.device.pyu.PYUObject object at 0x7fe6fef03250>
