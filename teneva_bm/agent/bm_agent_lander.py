import numpy as np
from teneva_bm.agent.agent import Agent


class BmAgentLander(Agent):
    def __init__(self, d=None, n=3, seed=42, name=None,
                 steps=1000, policy='toeplitz'):
        super().__init__(d, n, seed, name, steps, policy)
        self.set_desc_agent('Lunar Lander (continuous)',
            'https://www.gymlibrary.dev/environments/box2d/lunar_lander')

    @property
    def ref(self):
        i = np.zeros(55, dtype=int)
        for k in [0, 6, 12, 20, 34, 44, 53]:
            i[k] = 1
        return np.array(i, dtype=int), -877.6547852646436

    def prep_bm(self):
        env = Agent.make('LunarLander-v2', continuous=True)
        return super().prep_bm(env)

    def render(self, fpath=None, i=None, best=True, fps=20., sz=(600, 600)):
        return super().render(fpath, i, best, fps, sz)
