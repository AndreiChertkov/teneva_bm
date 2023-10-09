import numpy as np
from teneva_bm import Bm


class BmHsFunc111(Bm):
    def __init__(self, d=10, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 111 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                c = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179]
            x - continuous control
                x[0] | >= -100 | <= 100
                x[1] | >= -100 | <= 100
                x[2] | >= -100 | <= 100
                x[3] | >= -100 | <= 100
                x[4] | >= -100 | <= 100
                x[5] | >= -100 | <= 100
                x[6] | >= -100 | <= 100
                x[7] | >= -100 | <= 100
                x[8] | >= -100 | <= 100
                x[9] | >= -100 | <= 100
            F - objective function
                sum(exp(x) * (c + x - log(sum(x))))
            C - constraint function
                exp(x[0]) + 2 * exp(x[1]) + 2 * exp(x[2]) + exp(x[5]) + exp(x[9]) - 2 = 0
                exp(x[3]) + 2 * exp(x[4]) + exp(x[5]) + exp(x[6]) - 1 = 0
                exp(x[2]) + exp(x[6]) + exp(x[7]) + 2 * exp(x[8]) + exp(x[9]) - 1 = 0
            The exact global minimum is approx. known:
                y ~= -47.761
                x[0] ~= -3.202
                x[1] ~= -1.912
                x[2] ~= -0.244
                x[3] ~= -6.561
                x[4] ~= -0.723
                x[5] ~= -7.274
                x[6] ~= -3.597
                x[7] ~= -4.020
                x[8] ~= -3.288
                x[9] ~= -2.334
            Hyperparameters: 
                * The dimension d should be 10
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100], 
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
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
        c_1 = np.abs(np.exp(X[:, 0]) + 2 * np.exp(X[:, 1]) + 2 * np.exp(X[:, 2]) + np.exp(X[:, 5]) + np.exp(X[:, 9]) - 2)
        c_2 = np.abs(np.exp(X[:, 3]) + 2 * np.exp(X[:, 4]) + np.exp(X[:, 5]) + np.exp(X[:, 6]) - 1)
        c_3 = np.abs(np.exp(X[:, 2]) + np.exp(X[:, 6]) + np.exp(X[:, 7]) + 2 * np.exp(X[:, 8]) + np.exp(X[:, 9]) - 1)
        return np.array([c_1, c_2, c_3])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (np.exp(X) * (self.parameters['c'][None, :] + X - np.log(X.sum(-1))[:, None])).sum(-1)