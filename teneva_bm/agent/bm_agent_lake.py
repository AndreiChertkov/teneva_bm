import matplotlib.pyplot as plt
import numpy as np
from teneva_bm.agent.agent import Agent


class BmAgentLake(Agent):
    INIT = 'S'
    FROZ = 'F'
    HOLE = 'H'
    GOAL = 'G'

    def holes(map):
        size = len(map)
        holes = []
        for i in range(size):
            for j in range(size):
                if map[i][j] == BmAgentLake.HOLE:
                    holes.append(np.array([i, j], dtype=int))
        return holes

    def map(size=10, holes=10, seed=42, iters=1.E+4, prob_hole=0.05):
        map = []
        rand = np.random.default_rng(seed) if isinstance(seed, int) else seed
        iter = 0
        while len(map) == 0 or not Agent.frozen_lake.is_valid(map, size):
            iter += 1
            if iter > iters:
                if prob_hole > 0.9:
                    msg = 'Can not build the map for selected size / holes'
                    raise ValueError(msg)
                return BmAgentLake.map(size, holes, seed, iters, prob_hole+0.1)

            map = rand.choice([BmAgentLake.FROZ, BmAgentLake.HOLE],
                (size, size), p=[1-prob_hole, prob_hole])
            map[0, 0] = BmAgentLake.INIT
            map[-1, -1] = BmAgentLake.GOAL

            if len(BmAgentLake.holes(map)) != holes:
                map = []

        return [''.join(x) for x in map]

    def __init__(self, d=None, n=4, seed=42, name=None,
                 policy='direct', size=10, holes=10, with_state_ext=False):
        super().__init__(d, n, seed, name, size*size, policy)

        self.set_desc("""
            Agent "FrozenLake" from gym environment. For details, see
            https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
            By default, the direct optimization of agent's actions is performed
            (i.e., "policy='direct'"). You can also set own policy class
            instance (see "agent/policy.py" with a description of the interface
            design details). The dimension is determined automatically
            according to the properties of the agent and the used policy; the
            mode size should be 4. Note that agent actions mean: 0: LEFT, 1:
            DOWN, 2: RIGHT, 3: UP.
        """)

        self.size = size
        self.holes = holes
        self.with_state_ext = with_state_ext

        if isinstance(policy, str):
            if policy == 'direct':
                if n != 4:
                    msg = f'Mode size should be 4 for "direct" policy'
                    self.set_err(msg)
            else:
                msg = f'Policy "{policy}" is not supported'
                self.set_err(msg)

    @property
    def args_info(self):
        return {**super().args_info,
            'size': {
                'desc': 'Size of the lake',
                'kind': 'int'
            },
            'holes': {
                'desc': 'Number of the holes',
                'kind': 'int'
            },
            'with_state_ext': {
                'desc': 'State is extended',
                'kind': 'bool'
            },
        }

    @property
    def identity(self):
        return ['size', 'holes', 'policy', 'n', 'seed', 'with_state_ext']

    @property
    def is_func(self):
        return False if self.policy == 'direct' else True

    @property
    def ref(self):
        i = np.zeros(100, dtype=int)
        for k in [0, 6, 12, 20, 34, 44, 53, 88, 91]:
            i[k] = 1
        return np.array(i, dtype=int), 0.08304479020429478

    @property
    def _a_ac(self):
        return None

    @property
    def _a_st(self):
        return None

    @property
    def _b_ac(self):
        return None

    @property
    def _b_st(self):
        return None

    @property
    def _d_ac(self):
        return 1

    @property
    def _d_st(self):
        if self.with_state_ext:
            # State includes positions of the holes, goal and agent
            return len(self._holes) * 2 + 2 + 2
        else:
            # State includes only current position of the agent
            return 2

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
        for hole in self._holes:
            if hole[0] == self._state[0] and hole[1] == self._state[1]:
                return True
        return False

    @property
    def _is_state_same(self):
        if self._step_prev is None:
            return False
        if self._state[0] != self._step_prev[0]:
            return False
        if self._state[1] != self._step_prev[1]:
            return False
        return True

    @property
    def _n_ac(self):
        return np.array([self._env.action_space.n] * self._d_ac, dtype=int)

    @property
    def _n_st(self):
        return np.array([self._env.observation_space.n] * self._d_st, dtype=int)

    @property
    def _step_prev(self):
        if len(self._states) == 0:
            return None
        elif len(self._states) == 1:
            return self._state0
        else:
            return self._states[-2]

    def prep_bm(self, policy=None):
        self._map = BmAgentLake.map(self.size, self.holes, self.rand)
        self._holes = BmAgentLake.holes(self._map)

        env = Agent.make('FrozenLake-v1', desc=self._map, is_slippery=False)

        self._state_goal = self._discretize(self.size*self.size-1)

        return super().prep_bm(env)

    def render(self, fpath=None, i=None, best=True, fps=5., sz=None):
        return super().render(fpath, i, best, fps, sz)

    def _discretize(self, state):
        x, y = (state // self.size, state % self.size)
        return np.array([x, y], dtype=int)

    def _parse_action_gym(self, action):
        return action[0]

    def _parse_state(self, state):
        return self._discretize(state)

    def _parse_state_policy(self, state):
        if not self.with_state_ext:
            return state

        state_ext = []
        for hole in self._holes:
            state_ext.extend(list(hole))
        state_ext.extend(list(self._state_goal))
        state_ext.extend(list(state))
        return state_ext

    def _parse_reward(self, reward, eps=1.E-4):
        dist = np.linalg.norm(self._state - self._state_goal)
        reward += 1. / (dist + eps)
        # reward += 1000 if self._is_state_goal else 0
        # reward -= 1000 if self._is_state_hole else 0
        # reward -= 10 if self._is_state_same(state) else 0
        # reward -= self._step
        return reward
