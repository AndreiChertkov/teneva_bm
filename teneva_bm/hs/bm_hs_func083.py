import numpy as np
from teneva_bm import Bm


class BmHsFunc083(Bm):
    def __init__(self, d=5, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 083 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .-------------------------------------.
            | F(x | p, t) -> min s.t. C(x) = True |
            .-------------------------------------.
            p - parameters
                а = [85.334407, 0.0056858, 0.0006262, 0.0022053, 80.51249, 0.0071317, 
                     0.0029955, 0.0021813, 9.300961, 0.0047026, 0.0012547, 0.0019085]
            x - continuous control
                x[0] | >= 78 | <= 102
                x[1] | >= 33 | <= 45
                x[2] | >= 27 | <= 45
                x[3] | >= 27 | <= 45
                x[4] | >= 27 | <= 45
            t - intermediates
                t[0] = a[0] + a[1] * x[1] * x[4] + \
                       a[2] * x[0] * x[3] - a[3] * x[2] * x[4]
                t[1] = a[4] + a[5] * x[1] * x[4] + \
                       a[6] * x[0] * x[1] + a[7] * x[2] ** 2 - 90
                t[2] = a[8] + a[9] * x[2] * x[4] + \
                       a[10] * x[0] * x[2] + a[11] * x[2] * x[3] - 20
            F - objective function
                5.3578547 * x[2] ** 2 + 0.8356891 * x[0] * x[4] + 37.293239 * x[0] - 40792.141
            C - constraint function
                t[0] >= 0
                92 - t[0] >= 0
                t[1] >= 0
                20 - t[1] >= 0
                t[2] >= 0
                5 - t[2] >= 0
            The exact global minimum is approx. known:
                y ~= -30665.539
                x[0] ~= 78
                x[1] ~= 33
                x[2] ~= 29.995
                x[3] ~= 45
                x[4] ~= 36.776
            Hyperparameters: 
                * The dimension d should be 5
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([78, 33, 27, 27, 27], [102, 45, 45, 45, 45])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 5}

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
            'a': [
                85.334407, 0.0056858, 0.0006262, 0.0022053, 80.51249, 0.0071317, 
                0.0029955, 0.0021813, 9.300961, 0.0047026, 0.0012547, 0.0019085
            ]
        }

    def intermediates(self, X):
        t = [
            self.parameters['a'][0] + self.parameters['a'][1] * X[:, 1] * X[:, 4] + \
            self.parameters['a'][2] * X[:, 0] * X[:, 3] - self.parameters['a'][3] * X[:, 2] * X[:, 4],
            self.parameters['a'][4] + self.parameters['a'][5] * X[:, 1] * X[:, 4] + \
            self.parameters['a'][6] * X[:, 0] * X[:, 1] + self.parameters['a'][7] * X[:, 2] ** 2 - 90,
            self.parameters['a'][8] + self.parameters['a'][9] * X[:, 2] * X[:, 4] + \
            self.parameters['a'][10] * X[:, 0] * X[:, 2] + self.parameters['a'][11] * X[:, 2] * X[:, 3] - 20
        ]
        return t

    def _constr_batch(self, X):
        t = self.intermediates(X)
        c_1 = -1 * (t[0])
        c_2 = -1 * (92 - t[0])
        c_3 = -1 * (t[1])
        c_4 = -1 * (20 - t[1])
        c_5 = -1 * (t[2])
        c_6 = -1 * (5 - t[2])
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return 5.3578547 * X[:, 2] ** 2 + 0.8356891 * X[:, 0] * X[:, 4] + 37.293239 * X[:, 0] - 40792.141