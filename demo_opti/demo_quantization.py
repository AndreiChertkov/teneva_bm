"""Benchmark optimization example with quantization.

We are looking for a global minimum for benchmark "BmHsFunc001" using a
gradient-free PROTES optimizer based on the tensor train (TT) decomposition
(see https://github.com/anabatsh/PROTES). Since PROTES works only for the
multidimensional case (d > 2), we perform quantization for the used 2D
benchmark. The mode size factor for the benchmark ("q"; "n = 2^q") and
budget ("m") are given as function "demo" arguments.

To run the code ("python demo_opti/demo_quantization.py"), you need to install
the PROTES optimizer: "pip install protes==0.3.2".

"""
import numpy as np
from protes import protes
from time import perf_counter as tpc


from teneva_bm import BmHsFunc001


def demo(q=15, m=500000):
    bm = BmHsFunc001(d=2, n=2**q)
    bm.set_cache(True)
    bm.set_quantization(True)
    bm.set_log(True, cond='min', prefix='protes', with_max=False)
    bm.prep()
    print(bm.info())

    t = tpc()
    print(f'Optimization process:\n')
    i_opt, y_opt = protes(bm.get, bm.d*q, 2, m, k=1000, k_top=10)
    bm.log('<<< DONE', out=True)

    print(f'\nOptimization result:\n')
    print(f'Dimension    = {bm.d:-11.0f}')
    print(f'Quantization = {q:-11.0f}')
    print(f'Budget       = {m:-11.4e}')
    print(f'Time (sec)   = {tpc()-t:-11.4f}')
    print(f'Solution (y) = {y_opt:-11.4e} ')
    print(f'Solution (x) =', bm.x_min)
    print()

    print(bm.info_history())


if __name__ == '__main__':
    demo()
