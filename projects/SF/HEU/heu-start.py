import secretflow as sf
import secretflow.device as ft
import numpy as np
import spu

sf.shutdown()
sf.init(['alice','bob'], num_cpus=16, log_to_driver=False)
cluster_def={
    'nodes':[
        {'party':'alice','id':'local:0','address':'127.0.0.1:12345'},
        {'party': 'bob', 'id':'local:1','address':'127.0.0.1:12346'},
    ],

    'runtime_config':{
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
    },
}
heu_config ={
    'sk_keeper': {'party': 'alice'},
    'evaluators': [{'party': 'bob'}],
    'mode': 'PHEU', # The HEU working mode, choose from PHEU / LHEU / FHEU_ROUGH / FHEU
    'he_parameters':{
        'schema':'zpaillier',
        'key_pair':{
            'generate':{
                'bit size': 2048,
            },
        },
    },
}
alice = sf.PYU('alice')
bob = sf.PYU('bob')
heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])
x=ft.with_device(alice)(np.random.rand)(3,4)
y=ft.with_device(bob)(p.random.rand)(3,4)
x_= x.to(heu)
y_ = y.to(heu)
add_=x_+ y_
add = add_.to(alice)
print(sf.reveal(add))

