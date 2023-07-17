"""We check here the code snippets from the README.md file."""


from teneva_bm import *
bm = BmQuboMaxcut().prep()
print(bm.info())


from teneva_bm import teneva_bm_demo
teneva_bm_demo('bm_qubo_knap_amba', with_info=True)


import numpy as np
from teneva_bm import *
np.random.seed(42)

bm = BmFuncAckley().prep()
print(bm.info())


# Get value at multi-index i:
i = np.ones(bm.d)
print(bm[i]) # you can use the alias "bm.get(i)"

# Get values for batch of multi-indices I:
I = np.array([i, i+1, i+2])
print(bm[I]) # you can use the alias "bm.get(I)"


# Get value at point x:
x = np.ones(bm.d) * 0.42
print(bm(x)) # you can use the alias "bm.get_poi(x)"

# Get values for batch of points X:
X = np.array([x, x*0.3, x*1.1])
print(bm(X)) # you can use the alias "bm.get_poi(X)"


# Generate random train dataset (from LHS):
# I_trn is array of [500, bm.d] and y_trn is [500]
I_trn, y_trn = bm.build_trn(500)

# Generate random test dataset (from random choice):
# I_tst is array of [100, bm.d] and y_tst of [100]
I_tst, y_tst = bm.build_tst(100)
