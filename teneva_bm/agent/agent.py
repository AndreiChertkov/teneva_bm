import matplotlib.pyplot as plt
import numpy as np


try:
    import cv2
    with_cv2 = True
except Exception as e:
    with_cv2 = False


try:
    import gym
    gym.logger.set_level(40)
    np.bool8 = bool
    from gym.envs.toy_text import frozen_lake as gym_frozen_lake
    with_gym = True
except Exception as e:
    with_gym = False


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning,
    module=r"importlib\.")
warnings.filterwarnings('ignore', category=DeprecationWarning,
    module=r"ipkernel")


from teneva_bm import Bm
from teneva_bm.agent.policy import Policy
from teneva_bm.agent.policy import PolicyToeplitz


class Agent(Bm):
    def make(name, **args):
        if with_gym:
            return gym.make(name, render_mode='rgb_array', **args)

    frozen_lake = gym_frozen_lake if with_gym else None

    def __init__(self, d=None, n=3, name='Agent-abstract-class', desc='',
                 steps=1000, policy='toeplitz'):
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

        if isinstance(policy, str):
            self._policy_name = policy

            if self._policy_name == 'direct':
                self._policy = Policy()
            elif self._policy_name == 'toeplitz':
                self._policy = PolicyToeplitz()
            else:
                raise ValueError('Invalid policy name')

        else:
            self._policy_name = policy.name

            self._policy = policy

        self._with_render = False

    @property
    def identity(self):
        return ['_steps', '_policy_name', 'n']

    @property
    def is_func(self):
        return True

    @property
    def is_opti_max(self):
        return True

    @property
    def _d_ac(self):
        return len(self._env.action_space.low)

    @property
    def _d_st(self):
        return len(self._env.observation_space.low)

    @property
    def _a_ac(self):
        return np.array(self._env.action_space.low)

    @property
    def _a_st(self):
        return np.array(self._env.observation_space.low)

    @property
    def _b_ac(self):
        return np.array(self._env.action_space.high)

    @property
    def _b_st(self):
        return np.array(self._env.observation_space.high)

    @property
    def _n_ac(self):
        return None

    @property
    def _n_st(self):
        return None

    @property
    def _reward(self):
        return self._rewards[-1] if len(self._rewards) else 0.

    @property
    def _reward_total(self):
        return np.sum(np.array(self._rewards)) if len(self._rewards) else 0.

    @property
    def _state(self):
        return self._states[-1] if len(self._states) else self._state0

    def get_config(self):
        conf = super().get_config()
        conf['_steps'] = self._steps
        conf['_policy_name'] = self._policy_name
        conf['_d_st'] = self._d_st
        conf['_d_ac'] = self._d_ac
        return conf

    def info(self, footer=''):
        text = ''

        text += 'Number of agent steps                    : '
        v = self._steps
        text += f'{v}\n'

        text += 'Used policy                              : '
        v = self._policy_name
        text += f'{v}\n'

        text += 'Dimension of state space for agent       : '
        v = self._d_st
        text += f'{v}\n'

        text += 'Dimension of action space for agent      : '
        v = self._d_ac
        text += f'{v}\n'

        return super().info(text+footer)

    def prep_bm(self, env):
        if env is None:
            raise ValueError('Environment is not set')
        self._env = env

        self._policy.prep(self._steps,
            self._d_st, self._n_st, self._a_st, self._b_st,
            self._d_ac, self._n_ac, self._a_ac, self._b_ac)

        self.set_dimension(self._policy.d)
        self.set_size(self._n_raw)
        if self.is_func:
            self.set_grid(self._policy.a, self._policy.b)
        self._reset()

    def render(self, fpath=None, i=None, best=True, fps=20.):
        self._with_render = True
        i, y = self.get_solution(i, best)
        self._with_render = False

        frames = self._frames
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        # frame = cv2.resize(frame,(512, 512))

        fpath = self.path_build(fpath, 'mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fpath, fourcc, fps, (frames[0].shape[:2]))
        for frame in frames:
            out.write(frame)
        out.release()

    def show(self, fpath=None, i=None, best=True):
        i, y = self.get_solution(i, best)

        try:
            frame = self._env.render()
        except Exception as e:
            self.wrn('Can not render agent for image generation')

        plt.imshow(frame)
        plt.axis('off')

        fpath = self.path_build(fpath, 'png')
        plt.savefig(fpath, bbox_inches='tight') if fpath else plt.show()

    def target(self, params):
        self._policy.set(params)
        self._run()
        return self._reward_total

    def _parse_action(self, action):
        return action

    def _parse_action_gym(self, action):
        return action

    def _parse_reward(self, reward):
        return reward

    def _parse_state(self, state):
        return state

    def _parse_state_policy(self, state):
        return state

    def _reset(self):
        self._policy.reset()

        state, info = self._env.reset(seed=self.seed)

        self._step = 0
        self._state0 = self._parse_state(state)
        self._done = False

        self._actions = []
        self._frames = []
        self._rewards = []
        self._states = []

    def _run(self):
        self._reset()

        if self._with_render:
            try:
                self._frames.append(self._env.render())
            except Exception as e:
                self.wrn('Can not render agent for video generation')

        for step in range(self._steps):
            self._step = step

            state = self._parse_state_policy(self._state)
            action = self._policy(state)
            action = self._parse_action(action)
            self._actions.append(action)

            action_gym = self._parse_action_gym(action)
            state, reward, self._done = self._env.step(action_gym)[:3]

            state = self._parse_state(state)
            self._states.append(state)

            reward = self._parse_reward(reward)
            self._rewards.append(reward)

            if self._with_render:
                try:
                    self._frames.append(self._env.render())
                except Exception as e:
                    self.wrn('Can not render agent for video generation')

            if self._done:
                break
