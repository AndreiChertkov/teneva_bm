import numpy as np
from teneva_bm import Bm


class BmHsFunc095(Bm):
    def __init__(self, d=6, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 095 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                b = [4.97, -1.88, -29.08, -78.02]
            x - continuous control
                x[0] | >= 0 | <= 0.31
                x[1] | >= 0 | <= 0.046
                x[2] | >= 0 | <= 0.068
                x[3] | >= 0 | <= 0.042
                x[4] | >= 0 | <= 0.028
                x[5] | >= 0 | <= 0.0134
            F - objective function
                4.3 * x[0] + 31.8 * x[1] + 63.3 * x[2] + 15.8 * x[3] + 68.5 * x[4] + 4.7 * x[5]
            C - constraint function
                17.1 * x[0] + 38.2 * x[1] + 204.2 * x[2] + 212.3 * x[3] + \
                623.4 * x[4] + 1495.5 * x[5] - 169 * x[0] * x[2] - 3580 * x[2] * x[4] - \
                3810 * x[3] * x[4] - 18500 * x[3] * x[5] - 24300 * x[4] * x[5] - b[0] >= 0
                --------------------------------------------------------------------------
                17.9 * x[0] + 36.8 * x[1] + 113.9 * x[2] + 169.7 * x[3] + \
                337.8 * x[4] + 1385.2 * x[5] - 139 * x[0] * x[2] - 2450 * x[3] * x[4] - \
                16600 * x[3] * x[5] - 17200 * x[4] * x[5] - b[1] >= 0
                --------------------------------------------------------------------------
                (-273) * x[1] - 70 * x[3] - 819 * x[4] + 26000 * x[3] * x[4] - b[2] >= 0
                159.9 * x[0] - 311 * x[1] + 587 * x[3] + 391 * x[4] + 2198 * x[5] - 14000 * x[0] * x[5] - b[3] >= 0
            The exact global minimum is approx. known:
                y ~= 0.016
                x[0] ~= 0
                x[1] ~= 0
                x[2] ~= 0
                x[3] ~= 0
                x[4] ~= 0
                x[5] ~= 0.003
            Hyperparameters: 
                * The dimension d should be 6
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0, 0, 0, 0], [0.31, 0.046, 0.068, 0.042, 0.028, 0.0134])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 6}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    @property
    def with_constr(self):
        return True

    def set_parameters(self):
        self.parameters = {'b': [4.97, -1.88, -29.08, -78.02]}

    def _constr_batch(self, X):
        c_1 = -1 * (
            17.1 * X[:, 0] + 38.2 * X[:, 1] + 204.2 * X[:, 2] + 212.3 * X[:, 3] + \
            623.4 * X[:, 4] + 1495.5 * X[:, 5] - 169 * X[:, 0] * X[:, 2] - 3580 * X[:, 2] * X[:, 4] - \
            3810 * X[:, 3] * X[:, 4] - 18500 * X[:, 3] * X[:, 5] - \
            24300 * X[:, 4] * X[:, 5] - self.parameters['b'][0]
        )
        c_2 = -1 * (
            17.9 * X[:, 0] + 36.8 * X[:, 1] + 113.9 * X[:, 2] + 169.7 * X[:, 3] + \
            337.8 * X[:, 4] + 1385.2 * X[:, 5] - 139 * X[:, 0] * X[:, 2] - 2450 * X[:, 3] * X[:, 4] - \
            16600 * X[:, 3] * X[:, 5] - 17200 * X[:, 4] * X[:, 5] - self.parameters['b'][1]
        )
        c_3 = -1 * (
            (-273) * X[:, 1] - 70 * X[:, 3] - 819 * X[:, 4] + \
            26000 * X[:, 3] * X[:, 4] - self.parameters['b'][2]
        )
        c_4 = -1 * (
            159.9 * X[:, 0] - 311 * X[:, 1] + 587 * X[:, 3] + \
            391 * X[:, 4] + 2198 * X[:, 5] - \
            14000 * X[:, 0] * X[:, 5] - self.parameters['b'][3]
        )
        return np.array([c_1, c_2, c_3, c_4])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 4.3 * X[:, 0] + 31.8 * X[:, 1] + 63.3 * X[:, 2] + 15.8 * X[:, 3] + 68.5 * X[:, 4] + 4.7 * X[:, 5]