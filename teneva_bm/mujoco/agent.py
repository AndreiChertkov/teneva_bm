import cv2
import gym
import numpy as np
import os


class Agent:
    def __init__(self, name, seed=42):
        self.name = name
        self.seed = seed

        self.env = gym.make(self.name, render_mode='rgb_array')

        self.dim_actions = len(self.env.action_space.low)
        self.lim_actions_a = self.env.action_space.low[0]
        self.lim_actions_b = self.env.action_space.high[0]

        self.dim_state = len(self.env.observation_space.low)
        self.lim_state_a = self.env.observation_space.low[0]
        self.lim_state_b = self.env.observation_space.high[0]

        self.reset()

    @property
    def reward(self):
        return np.sum(np.array(self.rewards)) if len(self.rewards) else 0.

    @property
    def state(self):
        return self.states[-1] if len(self.states) else self.state0

    def copy(self):
        return Agent(self.name, self.seed)

    def info(self):
        text = '\n' + '=' * 20

        text += f'\nAgent              | '
        text += f'{self.name}'

        text += f'\nDim (actions)      : '
        text += f'{self.dim_actions}'

        text += f'\nDim (state)        : '
        text += f'{self.dim_state}'

        text += f'\nLim (actions)      : '
        text += f'[{self.lim_actions_a}, {self.lim_actions_b}]'

        text += f'\nLim (state)        : '
        text += f'[{self.lim_state_a}, {self.lim_state_b}]'

        text += '\n' + '-' * 20

        print(text + '\n')

    def reset(self, state0=None):
        self.env.reset(seed=self.seed)

        if state0 is None:
            if self.name == 'Swimmer-v4':
                state0 = np.zeros(8)
            else:
                raise NotImplementedError()
        self.state0 = state0
        self.set_state(self.state0)

        self.actions = []
        self.rewards = []
        self.states = []

    def run(self, actions=None, steps=None, state=None, video=None):
        """Run several steps for agent.

        Args:
            actions (list): list of actions for all steps. If it is not set,
                then random actions will be performed.
            steps (int): optional number of steps.
            state (np.ndarray): optional initial state for the agent. If set,
                then the agent's current state will be overwritten.
            video (str): optional path to video file. If set, then video will be
                generated.

        Returns:
            float: The total accumulated reward.

        Note:
            All actions, rewards and states during the steps are available in
            the corresponding class variables.

        """
        if steps is None:
            if actions is None:
                steps = self.env.spec.max_episode_steps
            else:
                steps = len(actions)

        if state is not None:
            self.set_state(state)

        frames = []

        for step in range(steps):
            if actions is not None:
                if len(actions) < step+1:
                    raise ValueError('Actions have invalid length')
                action = actions[step]
            else:
                action = self.env.action_space.sample()

            state, reward  = self.env.step(action)[:2]

            if video:
                frames.append(self.env.render())

            self.actions.append(action)
            self.rewards.append(reward)
            self.states.append(state)

        if video:
            self.save_video(frames, video)

        return self.reward

    def save_video(self, frames, fpath):
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

        fold = os.path.dirname(fpath)
        if fold:
            os.makedirs(fold, exist_ok=True)

        if not fpath.endswith('.mp4'):
            fpath += '.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fpath, fourcc, 20.0, (frames[0].shape[:2]))
        for frame in frames:
            out.write(frame)
        out.release()

    def set_state(self, state, x=0., y=0.):
        if self.name == 'Swimmer-v4':
            qpos = np.array([x, y] + list(state[:3]))
            qvel = state[3:]
            self.env.set_state(qpos, qvel)
        else:
            raise NotImplementedError()
