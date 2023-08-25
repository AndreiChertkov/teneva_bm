import numpy as np
from teneva_bm.agent.agent import Agent


class BmAgentReacher(Agent):
    def __init__(self, d=None, n=3, seed=42, name=None,
                 steps=1000, policy='toeplitz'):
        super().__init__(d, n, seed, name, steps, policy)
        self.set_desc_agent('Reacher',
            'https://www.gymlibrary.dev/environments/mujoco/reacher')

    @property
    def ref(self):
        i = np.zeros(58, dtype=int)
        for k in [0, 6, 12, 20, 34, 44, 53]:
            i[k] = 1
        return np.array(i, dtype=int), -2155.341270305704

    def prep_bm(self):
        env = Agent.make('Reacher-v4')
        return super().prep_bm(env)
