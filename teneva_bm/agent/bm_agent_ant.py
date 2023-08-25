import numpy as np
from teneva_bm.agent.agent import Agent


class BmAgentAnt(Agent):
    def __init__(self, d=None, n=3, seed=42, name=None,
                 steps=1000, policy='toeplitz'):
        super().__init__(d, n, seed, name, steps, policy)
        self.set_desc_agent('Ant',
            'https://mgoulao.github.io/gym-docs/environments/mujoco/ant')

    @property
    def ref(self):
        i = np.zeros(80, dtype=int)
        for k in [0, 6, 12, 20, 34, 44, 53, 70, 78]:
            i[k] = 1
        return np.array(i, dtype=int), -2816.746884336937

    def prep_bm(self):
        env = Agent.make('Ant-v4')
        return super().prep_bm(env)
