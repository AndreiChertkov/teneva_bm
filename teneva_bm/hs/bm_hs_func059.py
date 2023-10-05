import numpy as np
from teneva_bm import Bm


class BmHsFunc059(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 059 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                a = -0.12694
                b = -3.4054e-4
            x - continuous control
                x[0] | >= 0 | <= 75
                x[1] | >= 0 | <= 65
            F - objective function
                (-75.196) + 3.8112 * x[0] + a * x[0] ** 2 + 0.0020567 * x[0] ** 3 - 
                1.0345e-5 * x[0] ** 4 + 6.8306 * x[1] - 0.030234 * x[0] * x[1] + 
                1.28134e-3 * x[1] * x[0] ** 2 + 2.266e-7 * x[0] ** 4 * x[1] - 
                0.25645 * x[1] ** 2 + 0.0034604 * x[1] ** 3 - 1.3514e-5 * x[1] ** 4 + 
                28.106 / (x[1] + 1) + 5.2375e-6 * x[0] ** 2 * x[1] ** 2 + 
                6.3e-8 * x[0] ** 3 * x[1] ** 2 - 7e-10 * x[0] ** 3 * x[1] ** 3 + 
                b * x[0] * x[1] ** 2 + 1.6638e-6 * x[0] * x[1] ** 3 + 
                2.8673 * exp(0.0005 * x[0] * x[1]) - 3.5256e-5 * x[0] ** 3 * x[1]
            C - constraint function
                x[0] * x[1] - 700 >= 0
                x[1] - x[0] ** 2 / 125 >= 0
                (x[1] - 50) ** 2 - 5 * (x[0] - 55) >= 0
            The exact global minimum is approx. known:
                y ~= -7.804
                x[0] ~= 13.550
                x[1] ~= 51.660
            Hyperparameters: 
                * The dimension d should be 2
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0], [75, 65])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 2}

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
            'a': -0.12694,
            'b': -3.4054e-4
        }
        
    def _constr_batch(self, X):
        c_1 = -1 * (X[:, 0] * X[:, 1] - 700)
        c_2 = -1 * (X[:, 1] - X[:, 0] ** 2 / 125)
        c_3 = -1 * ((X[:, 1] - 50) ** 2 - 5 * (X[:, 0] - 55))
        return np.array([c_1, c_2, c_3])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (-75.196) + 3.8112 * X[:, 0] + self.parameters['a'] * X[:, 0] ** 2 + \
               0.0020567 * X[:, 0] ** 3 - 1.0345e-5 * X[:, 0] ** 4 + 6.8306 * X[:, 1] - \
               0.030234 * X[:, 0] * X[:, 1] + 1.28134e-3 * X[:, 1] * X[:, 0] ** 2 + \
               2.266e-7 * X[:, 0] ** 4 * X[:, 1] - 0.25645 * X[:, 1] ** 2 + 0.0034604 * X[:, 1] ** 3 - \
               1.3514e-5 * X[:, 1] ** 4 + 28.106 / (X[:, 1] + 1) + 5.2375e-6 * X[:, 0] ** 2 * X[:, 1] ** 2 + \
               6.3e-8 * X[:, 0] ** 3 * X[:, 1] ** 2 - 7e-10 * X[:, 0] ** 3 * X[:, 1] ** 3 + \
               self.parameters['b'] * X[:, 0] * X[:, 1] ** 2 + 1.6638e-6 * X[:, 0] * X[:, 1] ** 3 + \
               2.8673 * np.exp(0.0005 * X[:, 0] * X[:, 1]) - 3.5256e-5 * X[:, 0] ** 3 * X[:, 1]