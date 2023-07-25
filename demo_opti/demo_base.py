"""Basic benchmark optimization example.

We are looking for a global minimum for benchmark "BmQuboKnapDet" using a
gradient-free PROTES optimizer based on the tensor train (TT) decomposition
(see https://github.com/anabatsh/PROTES). The benchmark dimension ("d"; note
that it may be 10, 20, 50, 80 and 100 for the selected benchark) and budget
("m") are given as function "demo" arguments.

To run the code ("python demo_opti/demo_base.py"), you need to install the
PROTES optimizer: "pip install protes==0.3.3". The expected console otput is
expected to be like the following:
...
Optimization process:

protes > m 1.0e+02 [+ 0.0e+00] | t 1.7e+00 | min -1.024e+04 |
protes > m 2.0e+02 [+ 0.0e+00] | t 3.0e+00 | min -1.049e+04 |
protes > m 3.0e+02 [+ 0.0e+00] | t 3.0e+00 | min -1.115e+04 |
...
protes > m 1.5e+04 [+ 2.9e+02] | t 4.0e+00 | min -1.516e+04 |
protes > m 1.5e+04 [+ 4.7e+02] | t 4.1e+00 | min -1.517e+04 |
protes > m 1.9e+04 [+ 2.0e+04] | t 5.5e+00 | min -1.517e+04 | <<< DONE
...

"""
import numpy as np
from protes import protes
from time import perf_counter as tpc


from teneva_bm import BmQuboKnapDet


def demo(d=100, m=20000):
    bm = BmQuboKnapDet(d)
    bm.set_budget(m, m_cache=m)
    bm.set_cache(True)
    bm.set_log(True, cond='min', prefix='protes', with_max=False)
    bm.prep()
    print(bm.info())

    t = tpc()
    print(f'Optimization process:\n')
    i_opt, y_opt = protes(bm.get, bm.d, bm.n0)
    bm.log('<<< DONE', out=True)

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
