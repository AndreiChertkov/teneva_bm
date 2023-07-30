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
from teneva_bm.agent.policy import Policy
from teneva_bm.agent.policy import PolicyToeplitz


class Agent(Bm):
    def make(name, **args):
        if with_gym:
            return gym.make(name, render_mode='rgb_array', **args)

    def __init__(self, d=None, n=3, name='Agent-abstract-class', desc='',
                 steps=1000, policy_name='toeplitz'):
        super().__init__(None, None, name, desc)
        self._n_raw = n # We'll set it later in "prep_bm" method

        if not with_cv2:
            msg = 'Need "cv2" module. For installation please see README.md'
            self.set_err(msg)

        if not with_gym:
            msg = 'Need "gym" module. For installation please see README.md'
            self.set_err(msg)

        if d is not None:
            msg = 'Dimension number (d) should not be set manually'
            self.set_err(msg)

        self._steps = steps
        self._policy_name = policy_name

        self._with_render = False

    @property
    def identity(self):
        return ['_policy_name', 'n', '_steps']

    @property
    def is_func(self):
        return self._policy.is_func

    @property
    def is_opti_max(self):
        return True

    @property
    def _reward(self):
        return np.sum(np.array(self._rewards)) if len(self._rewards) else 0.

    @property
    def _state(self):
        return (self._states[-1] if len(self._states) else self._state0).copy()

    def get_config(self):
        conf = super().get_config()
        conf['_steps'] = self._steps
        conf['_policy_name'] = self._policy_name
        return conf

    def info(self, footer=''):
        text = ''

        text += 'Dimension of state space for agent       : '
        v = self._d_st
        text += f'{v}\n'

        text += 'Dimension of action space for agent      : '
        v = self._d_ac
        text += f'{v}\n'

        text += 'Number of agent steps                    : '
        v = self._steps
        text += f'{v}\n'

        text += 'Used policy                              : '
        v = self._policy_name
        text += f'{v}\n'

        return super().info(text+footer)

    def prep_bm(self, env, policy=None, d_st=None, d_ac=None):
        self._prep_env(env)
        self._prep_space(d_st, d_ac)
        self._prep_policy(policy)

        self.set_dimension(self._policy.d)
        self.set_size(self._n_raw)
        if self.is_func: # TODO !!!
            self.set_grid(self._a_ac*self._steps, self._b_ac*self._steps)

        self._reset()

    def render(self, fpath=None, i=None, best=True):
        self._with_render = True
        i, y = self.get_solution(i, best)
        self._with_render = False

        frames = self._frames
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        # frame = cv2.resize(frame,(512, 512))

        fpath = self.path_build(fpath, 'mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fpath, fourcc, 20.0, (frames[0].shape[:2]))
        for frame in frames:
            out.write(frame)
        out.release()

    def target(self, params):
        self._policy.set(params)
        self._run()
        return self._reward

    def _parse_action(self, action):
        return action
        if len(action) == 1:
            action = action[0]

    def _parse_reward(self, reward=0.):
        return reward

    def _parse_state(self, state):
        return state

    def _prep_env(self, env):
        if env is None:
            raise ValueError('Environment is not set')
        self._env = env

    def _prep_policy(self, policy=None):
        if policy is not None:
            self._policy = policy
        elif self._policy_name == 'none':
            self._policy = Policy(self._d_st, self._d_ac, self._steps)
        elif self._policy_name == 'toeplitz':
            self._policy = PolicyToeplitz(self._d_st, self._d_ac, self._steps)
        else:
            raise ValueError('Policy is not set')

    def _prep_space(self, d_st=None, d_ac=None):
        if d_st:
            self._d_st = d_st
            self._a_st = None
            self._b_st = None
        else:
            self._d_st = len(self._env.observation_space.low)
            self._a_st = list(self._env.observation_space.low)
            self._b_st = list(self._env.observation_space.high)

        if d_ac:
            self._d_ac = d_ac
            self._a_ac = None
            self._b_ac = None
        else:
            self._d_ac = len(self._env.action_space.low)
            self._a_ac = list(self._env.action_space.low)
            self._b_ac = list(self._env.action_space.high)

    def _reset(self):
        state, info = self._env.reset(seed=self.seed)

        self._step = 0
        self._state0 = self._parse_state(state)
        self._state_prev = None
        self._done = False

        self._actions = []
        self._frames = []
        self._rewards = []
        self._states = []

    def _run(self):
        self._reset()

        for step in range(self._steps):
            self._step = step

            action = self._policy(self._state, step)
            action = self._parse_action(action)

            state, reward, self._done = self._env.step(action)[:3]
            state = self._parse_state(state)
            reward = self._parse_reward(reward)

            self._state_prev = self._state.copy()
            self._actions.append(action)
            self._rewards.append(reward)
            self._states.append(state)

            if self._with_render:
                try:
                    self._frames.append(self._env.render())
                except Exception as e:
                    self._wrn('Can not render agent for video generation')

            if self._done:
                break
