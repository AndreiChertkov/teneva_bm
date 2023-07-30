"""Basic benchmark optimization example.

We are looking for a global minimum for benchmark "BmQuboKnapDet" using a
gradient-free PROTES optimizer. The benchmark dimension ("d"; note that it may
be 10, 20, 50, 80 and 100 for the selected benchark) and budget ("m") are given
as function "demo" arguments. To run the code use the following command:
$ clear && python demo_opti/base.py

The console otput is expected to be like the following:
"
...
Optimization process:

protes > m 1.0e+02 [+ 0.0e+00] | t 1.6e+00 | min -1.024e+04 |
protes > m 2.0e+02 [+ 0.0e+00] | t 2.8e+00 | min -1.049e+04 |
protes > m 3.0e+02 [+ 0.0e+00] | t 2.8e+00 | min -1.115e+04 |
...
protes > m 1.5e+04 [+ 2.9e+02] | t 3.8e+00 | min -1.516e+04 |
protes > m 1.5e+04 [+ 4.7e+02] | t 3.9e+00 | min -1.517e+04 |
protes > m 1.9e+04 [+ 2.0e+04] | t 5.3e+00 | min -1.517e+04 | <<< DONE
...
"

"""
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
from protes import protes


from teneva_bm import BmQuboKnapDet


def demo(d=100, m=2.E+4):
    bm = BmQuboKnapDet(d)
    bm.set_budget(m, m_cache=m)
    bm.set_cache(True)
    bm.set_log(True, cond='min', prefix='protes', with_max=False)
    bm.prep()
    print(bm.info())

    protes(bm.get, bm.d, bm.n0)
    print(bm.info_history())

    # Just to show the useful bm's variables:
    print(f'\nOptimization result:\n')
    print(f'Dimension    = {d:-11.0f}')
    print(f'Budget       = {bm.m:-11.4e}')
    print(f'Time (sec)   = {bm.time_full:-11.4f}')
    print(f'Solution (y) = {bm.y_min:-11.4e} ')
    print(f'Solution (i) =', bm.i_min)


if __name__ == '__main__':
    demo()
