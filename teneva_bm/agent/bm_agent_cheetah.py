import numpy as np
from teneva_bm.agent.agent import Agent


class BmAgentCheetah(Agent):
    def __init__(self, d=None, n=3, seed=42, name=None,
                 steps=1000, policy='toeplitz'):
        super().__init__(d, n, seed, name, steps, policy)
        self.set_desc_agent('Half Cheetah',
            'https://www.gymlibrary.dev/environments/mujoco/half_cheetah')

    @property
    def ref(self):
        i = np.zeros(68, dtype=int)
        for k in [0, 6, 12, 20, 34, 44, 53, 61, 67]:
            i[k] = 1
        return np.array(i, dtype=int), -634.5703872896258

    def prep_bm(self):
        env = Agent.make('HalfCheetah-v4')
        return super().prep_bm(env)
