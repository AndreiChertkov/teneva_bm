import numpy as np


from teneva_bm.agent.agent import Agent



DESC = """
    Agent from myjoco environment "Humanoid Standup". For details, see
    https://mgoulao.github.io/gym-docs/environments/mujoco/humanoid_standup/

    By default, no policy is used ("policy_name" is 'none"). The Toeplitz
    discrete policy may be also used (if "policy_name" is 'toeplitz"), see
    https://github.com/jparkerholder/ASEBO/blob/master/asebo/policies.py
"""


class BmAgentHumanStand(Agent):
    def __init__(self, d=None, n=3, name='AgentHumanStand', desc=DESC,
                 steps=1000, policy_name='toeplitz'):
        super().__init__(d, n, name, desc, steps, policy_name)

    def prep_bm(self, policy=None):
        env = Agent.make('HumanoidStandup-v4')
        return super().prep_bm(env, policy)


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

    text = 'Generate video for a random multi-index  :  '
    bm = BmAgentHumanStand(steps=200).prep()
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    bm.render('result/BmAgentHumanStand_demo_none')
    text += 'see "result/...demo_none.mp4'
    print(text)

    text = 'Generate video for a random multi-index  :  '
    bm = BmAgentHumanStand(steps=200, policy_name='toeplitz').prep()
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    bm.render('result/BmAgentHumanStand_demo_toeplitz')
    text += 'see "result/...demo_toeplitz.mp4'
    print(text)
