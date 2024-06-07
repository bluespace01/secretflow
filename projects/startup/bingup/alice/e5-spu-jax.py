import spu.utils.distributed as ppd

import numpy as np

def make_rand():
    return np.random.rand()

def greater(x, y):
    return x > y

# initialized the distributed environment.
ppd.init(ppd.SAMPLE_NODES_DEF, ppd.SAMPLE_DEVICES_DEF)
ppd.current().nodes_def
ppd.current().devices
print(ppd.device('SPU').details())


# run make_rand on P1, the value is visible for P1 only.
x = ppd.device("P1")(make_rand)()

# run make_rand on P2, the value is visible for P2 only.
y = ppd.device("P2")(make_rand)()

# run greater on SPU, it automatically fetches x/y from P1/P2 (as ciphertext), and compute the result securely.
ans = ppd.device("SPU")(greater)(x, y)


x_revealed = ppd.get(x)
y_revealed = ppd.get(y)
x_revealed, y_revealed, np.greater(x_revealed, y_revealed)



