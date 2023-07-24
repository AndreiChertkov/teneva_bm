from copy import deepcopy as copy
import numpy as np
import teneva


from teneva_bm import Bm
from teneva_bm.mujoco.agent import Agent
from teneva_bm.mujoco.policy import PolicyToeplitz


DESC = """
    Myjoco environment "Swimmer". For details, see
    https://mgoulao.github.io/gym-docs/environments/mujoco/swimmer
"""


class BmMujocoSwimmer(Bm):
    def __init__(self, d=None, n=3, name='MujocoSwimmer', desc=DESC):
        agent = Agent('Swimmer-v4')
        policy = PolicyToeplitz(agent.dim_state, agent.dim_actions)

        super().__init__(policy.d, n, name, desc)

        self.agent = agent
        self.policy = policy
        self.agent_actions = []

    @property
    def is_func(self):
        return False

    def render(self, fpath=None):
        if not fpath:
            fpath = f'_result/Bm{self.name}'
        self.agent.reset()
        self.agent.run(self.agent_actions, video=fpath)
        print(self.agent_actions)

    def set_opts(self, steps=20):
        """Setting options specific to this benchmark.

        Args:
            steps (int): number of environment steps.

        """
        self.opt_steps = steps

    def _f(self, i):
        print(i)
        self.policy.set_theta(i)
        self.agent.reset()
        for step in range(self.opt_steps):
            action = self.policy(self.agent.state)
            print(self.agent.state, action)
            self.agent.run([action])
        self.agent_actions = copy(self.agent.actions)
        return self.agent.reward


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmMujocoSwimmer().prep()
    print(bm.info())

    #I_trn, y_trn = bm.build_trn(1.E+1)
    #print(bm.info_history())

    text = 'Value at a random multi-index            :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)
    bm.render()
