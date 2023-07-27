import matplotlib.pyplot as plt
import numpy as np


try:
    from gym.envs.toy_text import frozen_lake
    with_gym = True
    np.bool8 = bool
except Exception as e:
    with_gym = False


from teneva_bm.agent.agent import Agent


DESC = """
    Agent from myjoco environment "FrozenLake". For details, see
    https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
    Note that agent actions mean: 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP

    By default ("policy_name" is 'none"), no policy is used. The Toeplitz
    discrete policy may be also used (if "policy_name" is 'toeplitz"), see
    https://github.com/jparkerholder/ASEBO/blob/master/asebo/policies.py
"""


class BmAgentLake(Agent):
    INIT = 'S'
    FROZ = 'F'
    HOLE = 'H'
    GOAL = 'G'

    def map(size, rand, holes_min, holes_max, p=0.8):
        if not with_gym:
            return [], []

        map = []
        while len(map) == 0 or not frozen_lake.is_valid(map, size):
            p = min(1., p)
            map = rand.choice([BmAgentLake.FROZ, BmAgentLake.HOLE],
                (size, size), p=[p, 1-p])
            map[0, 0] = BmAgentLake.INIT
            map[-1, -1] = BmAgentLake.GOAL

            holes = []
            for i in range(size):
                for j in range(size):
                    if map[i, j] == BmAgentLake.HOLE:
                        holes.append(np.array([i, j], dtype=int))

            if len(holes) < holes_min or len(holes) > holes_max:
                map = []

        return [''.join(x) for x in map], holes

    def __init__(self, d=None, n=4, name='AgentLake', desc=DESC,
                 steps=100, size=5, holes_min=10, holes_max=20,
                 policy_name='none', with_state_ext=False):
        super().__init__(d, n, name, desc, steps, policy_name)

        self._size = size
        self._holes_min = holes_min
        self._holes_max = holes_max
        self._with_state_ext = with_state_ext

    @property
    def is_func(self):
        return False

    @property
    def _is_state_done(self):
        return self._done

    @property
    def _is_state_fail(self):
        return self._is_state_done and not self._is_state_goal

    @property
    def _is_state_goal(self):
        if self._state[0] != self._state_goal[0]:
            return False
        if self._state[1] != self._state_goal[1]:
            return False
        return True

    @property
    def _is_state_hole(self):
        for hole in self.holes:
            if hole[0] == self._state[0] and hole[1] == self._state[1]:
                return True
        return False

    @property
    def _is_state_same(self):
        if self.step == 0:
            return False
        if self._state[0] != self._state_prev[0]:
            return False
        if self._state[1] != self._state_prev[1]:
            return False
        return True

    @property
    def _state_out(self):
        """The output state for the policy."""
        if not self._with_state_ext:
            return self._state

        state = []
        for hole in self.holes:
            state.extend(list(hole))
        state.extend(list(self._state_goal))
        state.extend(list(self._state))

        return np.array(state, dtype=int)

    def prep_bm(self, policy=None):
        map, self._holes = BmAgentLake.map(self._size, self.rand,
            self._holes_min, self._holes_max)

        env = Agent.env_build('FrozenLake-v1', desc=map, is_slippery=False)

        self._state_goal = self._discretize(self._size*self._size-1)

        if self._with_state_ext:
            # State includes positions of the holes, goal and agent
            d_st = len(self._holes) * 2 + 2 + 2
        else:
            # State includes only current position of the agent
            d_st = 2

        d_ac = 1
        self._n_st = np.array([env.observation_space.n] * d_st, dtype=int)
        self._n_ac = np.array([env.action_space.n] * d_ac, dtype=int)

        return super().prep_bm(env, policy, d_st, d_ac)

    def render000(self, actions, fpath=None):
        """Generate the video for agent movements with given actions."""
        self._reset()

        if not fpath:
            fpath = f'AgentLake_size-{self.size}_seed-{self.seed}.avi'
        if not fpath.endswith('.avi'):
            fpath += '.avi'
        if os.path.dirname(fpath):
            os.makedirs(os.path.dirname(fpath), exist_ok=True)


        frame = self._env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(512, 512))

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(fpath, fourcc, 5.0, (512, 512))

        ret = out.write(frame)

        for a in actions:
            state, reward, done, truncated, info = self._env.step(int(a))

            frame = self._env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(512, 512))

            ret = out.write(frame)

            if done:
                break

        out.release()

    def show(self, fpath=None, i=None, best=True):
        frame = self._env.render()
        plt.imshow(frame)
        plt.axis('off')

        fpath = self.path_build(fpath, 'png')
        plt.savefig(fpath, bbox_inches='tight') if fpath else plt.show()

    def _discretize(self, state):
        x, y = (state // self._size, state % self._size)
        return np.array([x, y], dtype=int)

    def _gen_state0(self):
        return np.zeros(self._d_st, dtype=int)

    def _parse_state(self, state):
        return self._discretize(state)

    def _parse_reward(self, reward=0., eps=1.E-4):
        dist = np.linalg.norm(self._state_goal - self._state)
        reward += 1. / (dist + eps)
        # reward += 1000 if self._is_state_goal else 0
        # reward -= 1000 if self._is_state_hole else 0
        # reward -= 10 if self._is_state_same else 0
        # reward -= self.step
        return reward


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmAgentLake().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+1)
    print(bm.info_history())

    text = 'Value at a random multi-index            :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Generate image for start state           :  '
    bm = BmAgentLake().prep()
    bm.show('result/BmAgentLake_demo_start')
    text += 'see "result/...demo_start.png'
    print(text)

    text = 'Generate image for final state           :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    bm.show('result/BmAgentLake_demo_final')
    text += 'see "result/...demo_final.png'
    print(text)

    text = 'Generate video for a random multi-index  :  '
    bm.render('result/BmAgentLake_demo_none', i)
    text += 'see "result/...demo_none.mp4'
    print(text)
