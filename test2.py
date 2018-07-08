a=[1,2,3]

import numpy as np

#np.savez_compressed('test',a=a)
k=np.load("test.npz")
print(k['a'])