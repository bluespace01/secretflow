import spu

import secretflow as sf

# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(['alice', 'bob', 'carol', 'dave'], address='local')

cheetah_config = sf.utils.testing.cluster_def(
    parties=['alice', 'bob'],
    runtime_config={
        'protocol': spu.spu_pb2.CHEETAH,
        'field': spu.spu_pb2.FM64,
    },
)

spu_device2 = sf.SPU(cheetah_config)
print(spu_device2.cluster_def)

#-------------------------------------------------------
def get_carol_assets():
    return 1000000


def get_dave_assets():
    return 1000002


carol, dave = sf.PYU('carol'), sf.PYU('dave')

carol_assets = carol(get_carol_assets)()
dave_assets = dave(get_dave_assets)()


def get_winner(carol, dave):
    return carol > dave


winner = spu_device2(get_winner)(carol_assets, dave_assets)

print(sf.reveal(winner))
#-------------------------------------------------------

# Advanced Topic: Multiple Returns from SPU Computation

# Option 1: Treat All Returns as Single

def get_multiple_outputs(x, y):
    return x + y, x - y

single_output = spu_device2(get_multiple_outputs)(carol_assets, dave_assets)

print(sf.reveal(single_output))

# Option 2: Decide Return Nums on the Fly
from secretflow.device.device.spu import SPUCompilerNumReturnsPolicy

multiple_outputs = spu_device2(
    get_multiple_outputs, num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_COMPILER
)(carol_assets, dave_assets)

print(sf.reveal(multiple_outputs[0]))
print(sf.reveal(multiple_outputs[1]))

# Option 3: Decide Return Nums Manually

user_multiple_outputs = spu_device2(
    get_multiple_outputs,
    num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
    user_specified_num_returns=2,
)(carol_assets, dave_assets)

print(sf.reveal(multiple_outputs[0]))
print(sf.reveal(multiple_outputs[1]))


sf.shutdown()  # Shutdown the secretflow runtime.<|endoftext|>
