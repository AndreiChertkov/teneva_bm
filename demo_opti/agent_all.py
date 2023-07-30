"""Optimization for all benchmarks from "agent" collection (DRAFT)."""
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
from protes import protes


from teneva_bm import BmAgentAnt
from teneva_bm import BmAgentHuman
from teneva_bm import BmAgentHumanStand
from teneva_bm import BmAgentPendInv
from teneva_bm import BmAgentPendInvDouble
from teneva_bm import BmAgentSwimmer


AGENTS = [
    BmAgentAnt,
    BmAgentHuman,
    BmAgentHumanStand,
    BmAgentPendInv,
    BmAgentPendInvDouble,
    BmAgentSwimmer,
]


def demo(Agent, steps=1000, m=1.E+2):
    bm = Agent(steps=steps)
    bm.set_budget(m, m_cache=m)
    bm.set_cache(True)
    bm.set_log(False, cond='max', prefix='protes', with_min=False)
    bm.prep()

    protes(bm.get, bm.d, bm.n0, is_max=True)
    print(bm.info_current(bm.name))

    bm.render(f'result/demo_opti_agent_all/{bm.name}')


if __name__ == '__main__':
    for Agent in AGENTS:
        demo(Agent)
