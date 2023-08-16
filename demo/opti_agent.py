"""Optimization example for "agent" collection.

We optimize the agent "BmAgentSwimmer" using a gradient-free PROTES optimizer.

To run the code use the following command:
$ clear && python demo/opti_agent.py
The generated video for the found optimumal strategy will be saved as a file
"result/demo/opti_agent.mp4". The console otput is expected to be like:
"
...
Optimization process:

protes > m 1.0e+02 [+ 0.0e+00] | t 1.0e+01 | max   9.946e+01 |
protes > m 2.0e+02 [+ 0.0e+00] | t 2.0e+01 | max   1.084e+02 |
protes > m 3.0e+02 [+ 0.0e+00] | t 2.9e+01 | max   1.237e+02 |
protes > m 5.0e+02 [+ 0.0e+00] | t 4.6e+01 | max   1.819e+02 |
protes > m 6.0e+02 [+ 0.0e+00] | t 5.5e+01 | max   1.913e+02 |
protes > m 8.0e+02 [+ 0.0e+00] | t 7.4e+01 | max   2.034e+02 |
protes > m 1.0e+03 [+ 0.0e+00] | t 9.2e+01 | max   2.034e+02 | <<< DONE
...
"

"""
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
from protes import protes


from teneva_bm import BmAgentSwimmer


def demo(steps=1000, m=1.E+3):
    bm = BmAgentSwimmer(steps=steps)
    bm.set_budget(m, m_cache=m)
    bm.set_cache(True)
    bm.set_log(True, cond='max', prefix='protes', with_min=False)
    bm.prep()
    print(bm.info())

    protes(bm.get, bm.d, bm.n0, is_max=True)
    print(bm.info_history())

    bm.render(f'result/demo/opti_agent')

    print('\n\n\n')


if __name__ == '__main__':
    demo()
