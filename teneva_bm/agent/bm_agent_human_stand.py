import numpy as np


from teneva_bm.agent.agent import Agent



DESC = """
    Agent "HumanoidStandup" from myjoco environment. For details, see
    https://mgoulao.github.io/gym-docs/environments/mujoco/humanoid_standup/
    By default, the Toeplitz policy ("policy='toeplitz'") is used
    (https://github.com/jparkerholder/ASEBO/blob/master/asebo/policies.py).
    You can also set "direct" policy (direct optimization of agent's actions)
    or own policy class instance (see "agent/policy.py" with a description of
    the interface design details). The dimension is determined automatically
    according to the properties of the agent and the used policy; the default
    mode size is 3 and the number of agent's steps is 1000.
"""


class BmAgentHumanStand(Agent):
    def __init__(self, d=None, n=3, name='AgentHumanStand', desc=DESC,
                 steps=1000, policy='toeplitz'):
        super().__init__(d, n, name, desc, steps, policy)

    def prep_bm(self):
        env = Agent.make('HumanoidStandup-v4')
        return super().prep_bm(env)


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmAgentHumanStand().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+1)
    print(bm.info_history())

    text = 'Value at a random multi-index            :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Render for "direct" policy               :  '
    bm = BmAgentHumanStand(steps=250, policy='direct').prep()
    fpath = f'result/{bm.name}/render_direct'
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    bm.render(fpath)
    text += f' see {fpath}'
    print(text)

    text = 'Render for "toeplitz" policy             :  '
    bm = BmAgentHumanStand(steps=250, policy='toeplitz').prep()
    fpath = f'result/{bm.name}/render_toeplitz'
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    bm.render(fpath)
    text += f' see {fpath}'
    print(text)

    text = 'Generate image for a random multi-index  :  '
    fpath = f'result/{bm.name}/show'
    bm.show(fpath)
    text += f' see {fpath}'
    print(text)
