"""Optimization example for "agent" collection.

We optimize the agent "BmAgentSwimmer" using a gradient-free PROTES optimizer.

To run the code use the following command:
$ clear && python demo_opti/agent.py
The generated video for the found optimumal strategy will be in
"result/demo_opti_agent". The console otput is expected to be like:
"
...
Optimization process:

protes > m 1.0e+02 [+ 0.0e+00] | t 1.0e+01 | max  9.946e+01 |
protes > m 2.0e+02 [+ 0.0e+00] | t 2.0e+01 | max  1.084e+02 |
protes > m 3.0e+02 [+ 0.0e+00] | t 2.9e+01 | max  1.237e+02 |
...
protes > m 9.1e+03 [+ 0.0e+00] | t 8.1e+02 | max  2.467e+02 |
protes > m 9.6e+03 [+ 0.0e+00] | t 8.5e+02 | max  2.753e+02 |
protes > m 1.0e+04 [+ 0.0e+00] | t 8.9e+02 | max  2.753e+02 | <<< DONE
...
"

"""
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
from protes import protes


from teneva_bm import BmAgentSwimmer


def demo(steps=1000, m=1.E+4):
    bm = BmAgentSwimmer(steps=steps)
    bm.set_budget(m, m_cache=m)
    bm.set_cache(True)
    bm.set_log(True, cond='max', prefix='protes', with_min=False)
    bm.prep()
    print(bm.info())

    protes(bm.get, bm.d, bm.n0, is_max=True)
    print(bm.info_history())

    bm.render(f'result/demo_opti_agent/{bm.name}')


if __name__ == '__main__':
    demo()
