import numpy as np
from scipy.linalg import toeplitz


class Policy:
    def __init__(self, name='direct'):
        self.name = name

        self.reset()

    @property
    def a(self):
        return np.array(list(self.a_ac) * self.steps)

    @property
    def b(self):
        return np.array(list(self.b_ac) * self.steps)

    @property
    def d(self):
        return self.d_ac * self.steps

    def __call__(self, state):
        return self.act(state)

    def act(self, state):
        action = self.actions[self.d_ac*self.step:self.d_ac*(self.step+1)]

        self.step += 1
        return action

    def prep(self, steps=0, d_st=0, n_st=None, a_st=None, b_st=None,
             d_ac=0, n_ac=None, a_ac=None, b_ac=None):
        self.steps = steps

        self.d_st = d_st
        self.n_st = self._prep_opt(n_st, self.d_st, int)
        self.a_st = self._prep_opt(a_st, self.d_st, float)
        self.b_st = self._prep_opt(b_st, self.d_st, float)

        self.d_ac = d_ac
        self.n_ac = self._prep_opt(n_ac, self.d_ac, int)
        self.a_ac = self._prep_opt(a_ac, self.d_ac, float)
        self.b_ac = self._prep_opt(b_ac, self.d_ac, float)

    def reset(self):
        self.step = 0

    def set(self, params):
        self.actions = params.copy()

    def _prep_opt(self, opt, d=None, kind=float):
        if opt is None:
            return None
        if isinstance(opt, (int, float)):
            if d is None or d <= 0:
                raise ValueError('Invalid grid option')
            opt = np.ones(d, dtype=kind) * kind(opt)
        return np.asanyarray(opt, dtype=kind)


class PolicyToeplitz(Policy):
    def __init__(self, name='toeplitz', num_hidden=8):
        super().__init__(name)

        self.d_in = num_hidden

    @property
    def a(self):
        return np.array([-1.] * self.d)

    @property
    def b(self):
        return np.array([+1.] * self.d)

    @property
    def d(self):
        return self.d_w1 + self.d_b1 + self.d_w2 + self.d_b2 + self.d_w3

    def act(self, state):
        h1 = np.dot(self.W1, state) + self.b1
        z1 = self._activation(h1)

        h2 = np.dot(self.W2, z1) + self.b2
        z2 = self._activation(h2)

        out = np.dot(self.W3, z2)
        out = self._activation(out)

        self.step += 1
        return out

    def prep(self, *args, **kwargs):
        super().prep(*args, **kwargs)

        self.d_w1 = self.d_st + self.d_in - 1
        self.d_b1 = self.d_in
        self.d_w2 = self.d_in * 2 - 1
        self.d_b2 = self.d_in
        self.d_w3 = self.d_ac + self.d_in - 1

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
