import secretflow as sf
import numpy as np

def average(data):
    return np.average(data, axis=1)

alice = sf.PYU('alice')
bob = sf.PYU('bob')

from secretflow.device import TEEU

# mrenclave can be omitted in simulation mode.
teeu = TEEU('carol', mr_enclave='')

a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()

# Transfer data to teeu.
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

# TEEU runs average.
avg_val = teeu(average)([a_teeu, b_teeu])
print(sf.reveal(avg_val))

a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )

