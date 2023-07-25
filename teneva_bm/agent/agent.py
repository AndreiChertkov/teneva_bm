from copy import deepcopy as copy
import numpy as np
import os


try:
    import cv2
    with_cv2 = True
except Exception as e:
    with_cv2 = False


try:
    import gym
    with_gym = True
    np.bool8 = bool
except Exception as e:
    with_gym = False


from teneva_bm import Bm


class Agent(Bm):
    def env_build(name, name_full):
        if not with_gym:
            return None

        env = gym.make(name_full, render_mode='rgb_array')
        env.bm_name = name
        return env

    def __init__(self, d=None, n=32, name='Agent-abstract-class', desc='',
                 steps=1000, env=None, policy=None):
        d_ac = len(env.action_space.low) if env else 0
        super().__init__(d or d_ac*steps, n, name, desc)

        if not with_cv2:
            msg = 'Need "cv2" module. For installation please see README.md'
            self.set_err(msg)

        if not with_gym:
            msg = 'Need "gym" module. For installation please see README.md'
            self.set_err(msg)

        self.env = env
        self.opt_steps = steps

        if self.env:
            self.d_ac = d_ac
            self.a_ac = list(self.env.action_space.low)
            self.b_ac = list(self.env.action_space.high)

            self.d_st = len(self.env.observation_space.low)
            self.a_st = list(self.env.observation_space.low)
            self.b_st = list(self.env.observation_space.high)

            self.policy = policy or Policy(self.d_st, self.d_ac)

        if self.is_func:
            self.set_grid(self.a_ac*self.opt_steps, self.b_ac*self.opt_steps)

        self.reset()

    @property
    def is_func(self):
        return self.policy and self.policy.is_func

    @property
    def reward(self):
        return np.sum(np.array(self.rewards)) if len(self.rewards) else 0.

    @property
    def state(self):
        return self.states[-1] if len(self.states) else self.state0

    def gen_state0(self):
        return np.zeros(self.d_st)

    def info(self, footer=''):
        text = ''

        text += 'Dimension of state space for agent       : '
        v = self.d_st
        text += f'{v}\n'

        text += 'Dimension of action space for agent      : '
        v = self.d_ac
        text += f'{v}\n'

        return super().info(text+footer)

    def render(self, fpath=None):
        self.run(with_render=True)

        frames = self.frames
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

        if not fpath:
            fpath = f'_result/Bm{self.name}'
        if not fpath.endswith('.mp4'):
            fpath += '.mp4'

        fold = os.path.dirname(fpath)
        if fold:
            os.makedirs(fold, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fpath, fourcc, 20.0, (frames[0].shape[:2]))
        for frame in frames:
            out.write(frame)
        out.release()

    def reset(self, state0=None):
        self.state0 = self.gen_state0() if state0 is None else state0

        if self.env:
            self.env.reset(seed=self.seed)
            self.set_state(self.state0)

        self.actions = []
        self.frames = []
        self.rewards = []
        self.states = []

    def run(self, with_render=False):
        self.reset()

        for step in range(self.opt_steps):
            action = self.policy(self.state, step)
            state, reward = self.env.step(action)[:2]
            # TODO: add check for termination

            self.actions.append(action)
            self.rewards.append(reward)
            self.states.append(state)

            if with_render:
                try:
                    self.frames.append(self.env.render())
                except Exception as e:
                    self._wrn('Can not render agent for video generation')

    def set_state(self, state):
        raise NotImplementedError

    def _f(self, x):
        self.policy.set(x)
        self.run()
        return self.reward


class Policy:
    def __init__(self, d_st, d_ac, is_func=True):
        self.d_st = d_st
        self.d_ac = d_ac
        self.is_func = is_func

        self.set()

    def set(self, params=None):
        self.params = params

    def __call__(self, state, step):
        return self.act(state, step)

    def act(self, state, step):
        if self.params is None:
            raise ValueError('Policy parameters are not sest')

        return self.params[self.d_ac*step:self.d_ac*(step+1)]
