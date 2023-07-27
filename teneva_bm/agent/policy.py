import numpy as np
from scipy.linalg import toeplitz


class Policy:
    def __init__(self, d_st, d_ac, steps, is_func=True):
        self.name = 'none'

        self.d_st = d_st
        self.d_ac = d_ac
        self.steps = steps
        self.is_func = is_func

    @property
    def d(self):
        return self.steps * self.d_ac

    def __call__(self, state, step):
        return self.act(state, step)

    def act(self, state, step):
        return self.actions[self.d_ac*step:self.d_ac*(step+1)]

    def set(self, params):
        self.actions = params.copy()


class PolicyToeplitz(Policy):
    def __init__(self, d_st, d_ac, steps, num_hidden=8):
        super().__init__(d_st, d_ac, steps, is_func=False)

        self.name = 'toeplitz'

        self.d_in = num_hidden

        self.d_w1 = self.d_st + self.d_in - 1
        self.d_b1 = self.d_in
        self.d_w2 = self.d_in * 2 - 1
        self.d_b2 = self.d_in
        self.d_w3 = self.d_ac + self.d_in - 1

    @property
    def d(self):
        return self.d_w1 + self.d_b1 + self.d_w2 + self.d_b2 + self.d_w3

    def act(self, state, step):
        h1 = np.dot(self.W1, state) + self.b1
        z1 = self._activation(h1)

        h2 = np.dot(self.W2, z1) + self.b2
        z2 = self._activation(h2)

        out = np.dot(self.W3, z2)
        out = self._activation(out)

        return out

    def set(self, params):
        params = params.copy()

        self.w1 = params[:self.d_w1]
        self.W1 = self._build_layer(self.d_in, self.d_st, self.w1)
        params = params[self.d_w1:]

        self.b1 = params[:self.d_b1]
        params = params[self.d_b1:]

        self.w2 = params[:self.d_w2]
        self.W2 = self._build_layer(self.d_in, self.d_in, self.w2)
        params = params[self.d_w2:]

        self.b2 = params[:self.d_b2]
        params = params[self.d_b2:]

        self.w3 = params.copy()
        self.W3 = self._build_layer(self.d_ac, self.d_in, self.w3)

    def _activation(self, inputs):
        return np.tanh(inputs)

    def _build_layer(self, d1, d2, params):
        return toeplitz(params[:d1], params[(d1-1):])
