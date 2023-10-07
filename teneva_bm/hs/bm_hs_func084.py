import numpy as np
from teneva_bm import Bm


class BmHsFunc084(Bm):
    def __init__(self, d=5, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 084 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .-------------------------------------.
            | F(x | p, t) -> min s.t. C(x) = True |
            .-------------------------------------.
            p - parameters
                a = [-24345, -8720288.849, 150512.5253, -156.6950325, 476470.3222, 
                     729482.8271, -145421.402, 2931.1506, -40.427932, 5106.192, 15711.36, 
                     -155011.1084, 4360.53352, 12.9492344, 10236.884, 13176.786, 
                     -326669.5104, 7390.68412, -27.8986976, 16643.076, 30988.146]
            x - continuous control
                x[0] | >= 0   | <= 1000
                x[1] | >= 1.2 | <= 2.4
                x[2] | >= 20  | <= 60
                x[3] | >= 9   | <= 9.3
                x[4] | >= 6.5 | <= 7
            t - intermediates
                t[0] = a[6] * x[0] + a[7] * x[0] * x[1] + a[8] * x[0] * x[2] + a[9] * x[0] * x[3] + a[10] * x[0] * x[4]
                t[1] = a[11] * x[0] + a[12] * x[0] * x[1] + a[13] * x[0] * x[2] + a[14] * x[0] * x[3] + a[15] * x[0] * x[4]
                t[2] = a[16] * x[0] + a[17] * x[0] * x[1] + a[18] * x[0] * x[2] + a[19] * x[0] * x[3] + a[20] * x[0] * x[4]
            F - objective function
                (-1) * a[0] - a[1] * x[0] - a[2] * x[0] * x[1] - a[3] * x[0] * x[2] - a[4] * x[0] * x[3] - a[5] * x[0] * x[4]
            C - constraint function
                294000 - t[0] >= 0
                t[0] >= 0
                294000 - t[1] >= 0
                t[1] >= 0
                277200 - t[2] >= 0
                t[2] >= 0
            The exact global minimum is approx. known:
                y ~= -5280335.133
                x[0] ~= 4.537
                x[1] ~= 2.400
                x[2] ~= 60
                x[3] ~= 9.300
                x[4] ~= 7
            Hyperparameters: 
                * The dimension d should be 5
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 1.2, 20, 9, 6.5], [1000, 2.4, 60, 9.3, 7])
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
                -24345, -8720288.849, 150512.5253, -156.6950325, 476470.3222, 
                729482.8271, -145421.402, 2931.1506, -40.427932, 5106.192, 15711.36, 
                -155011.1084, 4360.53352, 12.9492344, 10236.884, 13176.786, 
                -326669.5104, 7390.68412, -27.8986976, 16643.076, 30988.146
            ]
        }

    def intermediates(self, X):
        t = [
            self.parameters['a'][6] * X[:, 0] + \
            self.parameters['a'][7] * X[:, 0] * X[:, 1] + self.parameters['a'][8] * X[:, 0] * X[:, 2] + \
            self.parameters['a'][9] * X[:, 0] * X[:, 3] + self.parameters['a'][10] * X[:, 0] * X[:, 4],
            self.parameters['a'][11] * X[:, 0] + \
            self.parameters['a'][12] * X[:, 0] * X[:, 1] + self.parameters['a'][13] * X[:, 0] * X[:, 2] + \
            self.parameters['a'][14] * X[:, 0] * X[:, 3] + self.parameters['a'][15] * X[:, 0] * X[:, 4],
            self.parameters['a'][16] * X[:, 0] + \
            self.parameters['a'][17] * X[:, 0] * X[:, 1] + self.parameters['a'][18] * X[:, 0] * X[:, 2] + \
            self.parameters['a'][19] * X[:, 0] * X[:, 3] + self.parameters['a'][20] * X[:, 0] * X[:, 4]
        ]
        return t

    def _constr_batch(self, X):
        t = self.intermediates(X)
        c_1 = -1 * (294000 - t[0])
        c_2 = -1 * (t[0])
        c_3 = -1 * (294000 - t[1])
        c_4 = -1 * (t[1])
        c_5 = -1 * (277200 - t[2])
        c_6 = -1 * (t[2])
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return - self.parameters['a'][0] \
               - self.parameters['a'][1] * X[:, 0] \
               - self.parameters['a'][2] * X[:, 0] * X[:, 1] \
               - self.parameters['a'][3] * X[:, 0] * X[:, 2] \
               - self.parameters['a'][4] * X[:, 0] * X[:, 3] \
               - self.parameters['a'][5] * X[:, 0] * X[:, 4]