import numpy as np


from teneva_bm.agent.agent import Agent



DESC = """
    Agent from myjoco environment "Swimmer". For details, see
    https://mgoulao.github.io/gym-docs/environments/mujoco/swimmer
"""


class BmAgentSwimmer(Agent):
    def __init__(self, d=None, n=32, name='AgentSwimmer', desc=DESC,
                 steps=1000):
        env = Agent.env_build('Swimmer', 'Swimmer-v4')
        super().__init__(d, n, name, desc, steps, env)

        if d is not None:
            self.set_err('Dimension number (d) should not be set manually')

    def set_state(self, state, x=0., y=0.):
        qpos = np.array([x, y] + list(state[:3]))
        qvel = state[3:]
        self.env.set_state(qpos, qvel)


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmAgentSwimmer().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+1)
    print(bm.info_history())

    text = 'Value at a random multi-index            :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Generate video for a random multi-index  :  '
    bm.render()
    text += 'see "_result" folder with mp4 file'
    print(text)
