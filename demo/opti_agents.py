"""Optimization for all benchmarks from "agent" collection.

We are looking for a global maximum for benchmarks from "agent" collections
using a gradient-free PROTES optimizer. Note that we use a small budget during
optimization (so that the calculations happen quickly), hence the results are
far from optimal.

The generated video for the found optimumal strategy will be saved in the folder
"result/demo/opti_agents". To run the code use the following command:
$ clear && python demo/opti_agents.py

"""
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
from protes import protes


from teneva_bm import *


AGENTS = [
    BmAgentAnt,
    BmAgentCheetah,
    BmAgentHuman,
    BmAgentHumanStand,
    BmAgentLake,
    BmAgentLander,
    BmAgentPendInv,
    BmAgentPendInvDouble,
    BmAgentReacher,
    BmAgentSwimmer,
]


def demo(Agent, m=1.E+2):
    bm = Agent()
    bm.set_budget(m, m_cache=m)
    bm.set_cache(True)
    bm.set_log(True, cond='max', prefix='protes', with_min=False)
    bm.prep()

    protes(bm.get, bm.d, bm.n0, k=10, k_top=2, is_max=True)
    print(bm.info_current(bm.name))

    bm.render(f'result/demo/opti_agents/{bm.name}')

    print('\n\n\n')


if __name__ == '__main__':
    for Agent in AGENTS:
        demo(Agent)
