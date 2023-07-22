"""Basic benchmark optimization example.

We are looking for a global minimum for benchmark "BmQuboKnapDet" using a
gradient-free PROTES optimizer based on the tensor train (TT) decomposition
(see https://github.com/anabatsh/PROTES). The benchmark dimension ("d") and
budget ("m") are given as function "demo" arguments.

To run the code ("python demo_opti/demo_base.py"), you need to install the
PROTES optimizer: "pip install protes==0.3.2".

"""
import numpy as np
from protes import protes
from time import perf_counter as tpc


from teneva_bm import BmQuboKnapDet


def demo(d=100, m=50000):
    bm = BmQuboKnapDet(d)
    bm.set_cache(True)
    bm.set_log(True, cond='min', prefix='protes', with_max=False)
    bm.prep()
    print(bm.info())

    t = tpc()
    print(f'Optimization process:\n')
    i_opt, y_opt = protes(bm.get, bm.d, bm.n0, m)

    print(f'\nOptimization result:\n')
    print(f'Dimension    = {d:-11.0f}')
    print(f'Budget       = {m:-11.4e}')
    print(f'Time (sec)   = {tpc()-t:-11.4f}')
    print(f'Solution (y) = {y_opt:-11.4e} ')
    print(f'Solution (i) =', i_opt)
    print()

    print(bm.info_history())


if __name__ == '__main__':
    demo()
