"""Optimization for all benchmarks from "agent" collection (DRAFT)."""
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
from protes import protes


from teneva_bm import BmAgentHumanStand


def demo(Agent, steps=1000, m=1.E+4):
    bm = Agent(steps=steps)
    bm.set_budget(m, m_cache=m)
    bm.set_cache(True)
    bm.set_log(True, cond='max', prefix='protes', with_min=False)
    bm.prep()
    print(bm.info())

    protes(bm.get, bm.d, bm.n0, is_max=True)
    print(bm.info_history())

    bm.render(f'result/demo_opti_agent_all/{bm.name}')


if __name__ == '__main__':
    demo(BmAgentHumanStand)
