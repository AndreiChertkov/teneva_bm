import numpy as np
from teneva_bm.agent.agent import Agent


class BmAgentSwimmer(Agent):
    def __init__(self, d=None, n=3, seed=42, name=None,
                 steps=1000, policy='toeplitz'):
        super().__init__(d, n, seed, name, steps, policy)
        self.set_desc_agent('Swimmer',
            'https://mgoulao.github.io/gym-docs/environments/mujoco/swimmer')

    @property
    def ref(self):
        i = np.zeros(55, dtype=int)
        for k in [0, 6, 12, 20, 34, 44, 53]:
            i[k] = 1
        return np.array(i, dtype=int), 26.468193398280665

    def prep_bm(self):
        env = Agent.make('Swimmer-v4')
        return super().prep_bm(env)

    def _tmp_set_state(self, state, x=0., y=0.):
        # Draft
        qpos = np.array([x, y] + list(state[:3]))
        qvel = state[3:]
        self._env.set_state(qpos, qvel)


if __name__ == '__main__':
    # Service code just for test.
    bm = BmAgentSwimmer().prep()
    print(bm.d)
    print(bm[bm.ref[0]])
