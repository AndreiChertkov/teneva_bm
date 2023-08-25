import numpy as np
from teneva_bm.agent.agent import Agent


class BmAgentHuman(Agent):
    def __init__(self, d=None, n=3, seed=42, name=None,
                 steps=1000, policy='toeplitz'):
        super().__init__(d, n, seed, name, steps, policy)
        self.set_desc_agent('Humanoid',
            'https://mgoulao.github.io/gym-docs/environments/mujoco/Humanoid')

    @property
    def ref(self):
        i = np.zeros(438, dtype=int)
        for k in [0, 6, 12, 53, 122, 244, 311, 402]:
            i[k] = 1
        return np.array(i, dtype=int), 176.24318691880555

    def prep_bm(self):
        env = Agent.make('Humanoid-v4')
        return super().prep_bm(env)
