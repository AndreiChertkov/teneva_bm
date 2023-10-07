import numpy as np
from teneva_bm import Bm


class BmHsFunc072(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 072 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                a11 = 4
                a12 = 2.25
                a13 = 1
                a14 = 0.25
                a21 = 0.16
                a22 = 0.36
                a23 = 0.64
                a24 = 0.64
                b1 = 0.0401
                b2 = 0.010085
            x - continuous control
                x[0] | >= 0.001 | <= 400000
                x[1] | >= 0.001 | <= 300000
                x[2] | >= 0.001 | <= 200000
                x[3] | >= 0.001 | <= 100000
            F - objective function
                1 + x[0] + x[1] + x[2] + x[3]
            C - constraint function
                b1 - a11 / x[0] - a12 / x[1] - a13 / x[2] - a14 / x[3] >= 0
                b2 - a21 / x[0] - a22 / x[1] - a23 / x[2] - a24 / x[3] >= 0
            The exact global minimum is approx. known:
                y ~= 727.679
                x[0] ~= 193.407
                x[1] ~= 179.547
                x[2] ~= 185.018
                x[3] ~= 168.707
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0.001, 0.001, 0.001, 0.001], [400000, 300000, 200000, 100000])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 4}

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
            'a11': 4,
            'a12': 2.25,
            'a13': 1,
            'a14': 0.25,
            'a21': 0.16,
            'a22': 0.36,
            'a23': 0.64,
            'a24': 0.64,
            'b1': 0.0401,
            'b2': 0.010085,
        }

    def _constr_batch(self, X):
        c_1 = -1 * (
            self.parameters['b1'] - 
            self.parameters['a11'] / X[:, 0] - 
            self.parameters['a12'] / X[:, 1] - 
            self.parameters['a13'] / X[:, 2] - 
            self.parameters['a14'] / X[:, 3]
        )
        c_2 = -1 * (
            self.parameters['b2'] - 
            self.parameters['a21'] / X[:, 0] - 
            self.parameters['a22'] / X[:, 1] - 
            self.parameters['a23'] / X[:, 2] - 
            self.parameters['a24'] / X[:, 3]
        )
        return np.array([c_1, c_2])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 1 + X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3]