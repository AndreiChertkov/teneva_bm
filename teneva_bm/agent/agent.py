import matplotlib.pyplot as plt
import numpy as np
from teneva_bm import Bm
from teneva_bm.agent.policy import Policy
from teneva_bm.agent.policy import PolicyToeplitz
import warnings


warnings.filterwarnings('ignore', category=DeprecationWarning,
    module=r'importlib\.')
warnings.filterwarnings('ignore', category=DeprecationWarning,
    module=r'ipkernel')


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


class Agent(Bm):
    frozen_lake = gym_frozen_lake if with_gym else None

    def make(name, **args):
        if with_gym:
            return gym.make(name, render_mode='rgb_array', **args)

    def __init__(self, d=None, n=3, seed=42, name=None,
                 steps=1000, policy='toeplitz'):
        super().__init__(None, None, seed, name)
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

        self.steps = steps

        if isinstance(policy, str):
            self.policy = policy

            if self.policy == 'direct':
                self._policy = Policy()
            elif self.policy == 'toeplitz':
                self._policy = PolicyToeplitz()
            else:
                msg = f'Policy "{policy}" is not supported'
                self.set_err(msg)

        else:
            self.policy = policy.name

            self._policy = policy

        self._with_render = False

    @property
    def args_info(self):
        return {**super().args_info,
            'steps': {
                'desc': 'Number of agent steps',
                'kind': 'int'
            },
            'policy': {
                'desc': 'Name of the used policy',
                'kind': 'str'
            }
        }

    @property
    def identity(self):
        return ['steps', 'policy', 'n', 'seed']

    @property
    def is_func(self):
        return True

    @property
    def is_opti_max(self):
        return True

    @property
    def prps_info(self):
        return {**super().prps_info,
            '_d_st': {
                'desc': 'Dimension of state space for agent',
                'kind': 'int'
            },
            '_d_ac': {
                'desc': 'Dimension of action space for agent',
                'kind': 'int'
            }
        }

    @property
    def with_render(self):
        return True

    @property
    def with_show(self):
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

    def prep_bm(self, env):
        if env is None:
            raise ValueError('Environment is not set')
        self._env = env

        self._policy.prep(self.steps,
            self._d_st, self._n_st, self._a_st, self._b_st,
            self._d_ac, self._n_ac, self._a_ac, self._b_ac)

        self.set_dimension(self._policy.d)
        self.set_size(self._n_raw)
        if self.is_func:
            self.set_grid(self._policy.a, self._policy.b)
        self._reset()

    def render(self, fpath, i=None, best=True, fps=20., sz=None):
        self._with_render = True
        i, y = self.get_solution(i, best)

        if not self._with_render:
            msg = 'Can not save rendered video for agent (render failed)'
            self.wrn(msg)
            return

        self._with_render = False

        if not len(self._frames):
            msg = 'Can not save rendered video for agent (empty frames)'
            self.wrn(msg)
            return

        frames = self._frames
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
        if sz is not None:
            frames = [cv2.resize(frame, sz) for frame in frames]

        fpath = self.path_build(fpath, 'mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fpath, fourcc, fps, (frames[0].shape[:2]))
        for frame in frames:
            out.write(frame)
        out.release()

    def set_desc_agent(self, name, link):
        self.set_desc("""
            Agent "#NAME" from myjoco environment. See
            #LINK
            By default, the Toeplitz policy ("policy='toeplitz'") is used
            (https://github.com/jparkerholder/ASEBO/blob/master/asebo/policies.py).
            You can also set "direct" policy (direct optimization of agent's
            actions) or own policy class instance (see "agent/policy.py" with a
            description of the interface design details). The dimension is
            determined automatically according to the properties of the agent
            and the used policy; the default mode size is 3 and the number of
            agent's steps is 1000.
        """.replace('#NAME', name).replace('#LINK', link))

    def show(self, fpath=None, i=None, best=True):
        i, y = self.get_solution(i, best)

        try:
            frame = self._env.render()
        except Exception as e:
            msg = 'Can not render agent for image generation'
            msg += f' [Error: {e}]'
            self.wrn(msg)
            self._with_render = False
            return

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

        def _render():
            try:
                self._frames.append(self._env.render())
            except Exception as e:
                msg = 'Can not render agent for video generation'
                msg += f' [Error: {e}]'
                self.wrn(msg)
                self._with_render = False

        if self._with_render:
            _render()

        for step in range(self.steps):
            self._step = step

            state = self._parse_state_policy(self._state)
            action = self._policy(state)
            action = self._parse_action(action)
            self._actions.append(action)

            action = self._parse_action_gym(action)
            state, reward, self._done = self._env.step(action)[:3]

            state = self._parse_state(state)
            self._states.append(state)

            reward = self._parse_reward(reward)
            self._rewards.append(reward)

            if self._with_render:
                _render()

            if self._done:
                break
