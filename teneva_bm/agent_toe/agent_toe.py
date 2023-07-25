import numpy as np
from scipy.linalg import toeplitz


from teneva_bm.agent.agent import Agent


class AgentToe(Agent):
    def __init__(self, d=None, n=3, name='Agent-toe-abstract-class', desc='',
                 steps=1000, env=None):
        d_st = len(env.observation_space.low) if env else 0
        d_ac = len(env.action_space.low) if env else 0
        policy = PolicyToeplitz(d_st, d_ac)
        super().__init__(policy.d, n, name, desc, steps, env, policy)

        if d is not None:
            self.set_err('Dimension number (d) should not be set manually')


class PolicyToeplitz:
    def __init__(self, d_st, d_ac, num_hidden=8):
        self.d_st = d_st
        self.d_ac = d_ac
        self.d_in = num_hidden
        self.is_func = False

        self.all_weight_init()
        self.set()

    def __call__(self, state, step):
        return self.act(state, step)

    def act(self, state, step):
        h1 = np.dot(self.W1, state) + self.b1
        z1 = self._activation(h1)

        h2 = np.dot(self.W2, z1) + self.b2
        z2 = self._activation(h2)

        out = np.dot(self.W3, z2)
        out = self._activation(out)

        return out

    def all_weight_init(self):
        self.w1 = self.weight_init(self.d_st + self.d_in -1)
        self.w2 = self.weight_init(self.d_in * 2 - 1)
        self.w3 = self.weight_init(self.d_ac + self.d_in - 1)

        self.W1 = self._build_layer(self.d_in, self.d_st, self.w1)
        self.W2 = self._build_layer(self.d_in, self.d_in, self.w2)
        self.W3 = self._build_layer(self.d_ac, self.d_in, self.w3)

        self.b1 = self.weight_init(self.d_in)
        self.b2 = self.weight_init(self.d_in)

        self.params = np.concatenate(
            [self.w1, self.b1, self.w2, self.b2, self.w3])

        self.d = len(self.params)

    def set(self, params=None):
        self.params = params

        if self.params is None:
            return

        self.all_weight_init()
        self.update(params)
        self.params = np.concatenate(
            [self.w1, self.b1, self.w2, self.b2, self.w3])

    def update(self, vec):
        self.w1 += vec[:len(self.w1)]
        vec = vec[len(self.w1):]

        self.b1 += vec[:len(self.b1)]
        vec = vec[len(self.b1):]

        self.w2 += vec[:len(self.w2)]
        vec = vec[len(self.w2):]

        self.b2 += vec[:len(self.b2)]
        vec = vec[len(self.b2):]

        self.w3 += vec

        self.W1 = self._build_layer(self.d_in, self.d_st, self.w1)
        self.W2 = self._build_layer(self.d_in, self.d_in, self.w2)
        self.W3 = self._build_layer(self.d_ac, self.d_in, self.w3)

    def weight_init(self, d, zeros=True):
        # return self.rand.random(self.d) / np.sqrt(d)
        if zeros: # TODO: check
            return np.zeros(d)

    def _activation(self, inputs):
        return np.tanh(inputs)

    def _build_layer(self, d1, d2, v):
        col = v[:d1]
        row = v[(d1-1):]
        return toeplitz(col, row)
