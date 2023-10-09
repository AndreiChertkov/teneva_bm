import numpy as np
from teneva_bm import Bm


class BmHsFunc112(Bm):
    def __init__(self, d=10, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 112 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                c = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179]
            x - continuous control
                x[0] | >= 1.0e-6
                x[1] | >= 1.0e-6
                x[2] | >= 1.0e-6
                x[3] | >= 1.0e-6
                x[4] | >= 1.0e-6
                x[5] | >= 1.0e-6
                x[6] | >= 1.0e-6
                x[7] | >= 1.0e-6
                x[8] | >= 1.0e-6
                x[9] | >= 1.0e-6
            F - objective function
                sum(x * (c + log(x / sum(x))))
            C - constraint function
                x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2 = 0
                x[3] + 2 * x[4] + x[5] + x[6] - 1 = 0
                x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1 = 0
            The exact global minimum is approx. known:
                y ~= -47.761
                x[0] ~= 0.041
                x[1] ~= 0.148
                x[2] ~= 0.783
                x[3] ~= 0.001
                x[4] ~= 0.485
                x[5] ~= 0.001
                x[6] ~= 0.027
                x[7] ~= 0.018
                x[8] ~= 0.037
                x[9] ~= 0.097
            Hyperparameters: 
                * The dimension d should be 10
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06], 
            [+10, +10, +10, +10, +10, +10, +10, +10, +10, +10]
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 10}

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
        self.parameters = {
            'c': np.array([-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179])
        }

    def _constr_batch(self, X):
        c_1 = np.abs(X[:, 0] + 2 * X[:, 1] + 2 * X[:, 2] + X[:, 5] + X[:, 9] - 2)
        c_2 = np.abs(X[:, 3] + 2 * X[:, 4] + X[:, 5] + X[:, 6] - 1)
        c_3 = np.abs(X[:, 2] + X[:, 6] + X[:, 7] + 2 * X[:, 8] + X[:, 9] - 1)
        return np.array([c_1, c_2, c_3])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (X * (self.parameters['c'][None, :] + np.log(X / X.sum(-1)[:, None]))).sum(-1)