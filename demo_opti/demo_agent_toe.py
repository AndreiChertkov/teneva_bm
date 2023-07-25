"""Optimization example for agent_toe collection.

To run the code ("python demo_opti/demo_agent_toe.py"), you need to install the
PROTES optimizer: "pip install protes==0.3.3" and "mujoco" (see details in the
README.md file). The generated video "demo_agent_toe.mp.4" for the found
optimumal strategy will be in "_result" folder, and the console otput is
expected to be like the following:
...
Optimization process:

protes > m 1.0e+02 [+ 0.0e+00] | t 3.4e+00 | max  1.582e+01 |
protes > m 3.0e+02 [+ 0.0e+00] | t 8.0e+00 | max  1.985e+01 |
protes > m 4.0e+02 [+ 0.0e+00] | t 9.8e+00 | max  1.985e+01 |
...
protes > m 8.0e+03 [+ 0.0e+00] | t 1.4e+02 | max  4.363e+01 |
protes > m 9.0e+03 [+ 0.0e+00] | t 1.6e+02 | max  4.370e+01 |
protes > m 1.0e+04 [+ 0.0e+00] | t 1.8e+02 | max  4.370e+01 | <<< DONE
...

"""
import numpy as np
from protes import protes
from time import perf_counter as tpc


from teneva_bm import BmAgentToeSwimmer


def demo(steps=200, m=1.E+4):
    bm = BmAgentToeSwimmer(steps=steps)
    bm.set_budget(m, m_cache=m)
    bm.set_cache(True)
    bm.set_log(True, cond='max', prefix='protes', with_min=False)
    bm.prep()
    print(bm.info())

    t = tpc()
    print(f'Optimization process:\n')
    i_opt, y_opt = protes(bm.get, bm.d, bm.n0, is_max=True)
    bm.log('<<< DONE', out=True)

    print(f'\nOptimization result:\n')
    print(f'Budget       = {m:-11.4e}')
    print(f'Time (sec)   = {tpc()-t:-11.4f}')
    print(f'Solution (y) = {y_opt:-11.4e} ')
    print(f'Solution (i) =', i_opt)
    print()

    print(bm.info_history())

    bm.policy.set(i_opt)
    bm.render('_result/demo_agent_toe')


if __name__ == '__main__':
    demo()
