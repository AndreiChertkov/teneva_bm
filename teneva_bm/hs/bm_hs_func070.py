import numpy as np
from teneva_bm import Bm


class BmHsFunc070(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 070 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .-------------------------------------.
            | F(x | p, t) -> min s.t. C(x) = True |
            .-------------------------------------.
            p - parameters
                e = -1
                d = 7.658
                yobs = [0.00189, 0.1038, 0.268, 0.506, 0.577, 0.604, 0.725, 0.898, 0.947, 0.845, 
                        0.702, 0.528, 0.385, 0.257, 0.159, 0.0869, 0.0453, 0.01509, 0.00189]
                c = [0.1, 1, 2, 3, ..., 18]
            x - continuous control
                x[0] | >= 0.00001 | <= 100
                x[1] | >= 0.00001 | <= 100
                x[2] | >= 0.00001 | <= 1
                x[3] | >= 0.00001 | <= 100
            t - intermediates
                b = x[2] + (1 - x[2]) * x[3]
                ycal =  (1 + 1 / (12 * x[1])) ** e * \
                        x[2] * b ** x[1] * \
                        (x[1] / 6.2832) ** (1 / 2) * \
                        (c / d) ** (x[1] - 1) * \
                        exp(x[1] - b * c * x[1] / 7.658) + \
                        (1 + 1 / (12 * x[0])) ** e * \
                        (1 - x[2]) * (b / x[3]) ** x[0] * \
                        (x[0] / 6.2832) ** (1 / 2) * \
                        (c / 7.658) ** (x[0] - 1) * \
                        exp(x[0] - b * c * x[0] / (7.658 * x[3]))
            F - objective function
                sum((yobs - ycal) ** 2)
            C - constraint function
                b >= 0
            The exact global minimum is approx. known:
                y ~= 0.007
                x[0] ~= 12.277
                x[1] ~= 4.632
                x[2] ~= 0.313
                x[3] ~= 2.029
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([1e-05, 1e-05, 1e-05, 1e-05], [100, 100, 1, 100])
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
            'e': -1,
            'd': 7.658,
            'yobs': np.array([
                0.00189, 0.1038, 0.268, 0.506, 0.577, 0.604, 0.725, 0.898, 0.947, 0.845, 
                0.702, 0.528, 0.385, 0.257, 0.159, 0.0869, 0.0453, 0.01509, 0.00189
            ]),
            'c': np.array([0.1] + list(range(1, 19)))
        }

    def intermediates(self, X):
        X = X[:, :, None]
        c = self.parameters['c'][None, :]
        b = X[:, 2] + (1 - X[:, 2]) * X[:, 3]
        ycal_1 = (1 + 1 / (12 * X[:, 1])) ** self.parameters['e']
        ycal_2 = X[:, 2] * b ** X[:, 1] * (X[:, 1] / 6.2832) ** (1 / 2)
        ycal_3 = (c / self.parameters['d']) ** (X[:, 1] - 1)
        ycal_4 = np.exp(X[:, 1] - b * c * X[:, 1] / 7.658)
        ycal_5 = (1 + 1 / (12 * X[:, 0])) ** self.parameters['e']
        ycal_6 = (1 - X[:, 2]) * (b / X[:, 3]) ** X[:, 0]
        ycal_7 = (X[:, 0] / 6.2832) ** (1 / 2) * (c / 7.658) ** (X[:, 0] - 1)
        ycal_8 = np.exp(X[:, 0] - b * c * X[:, 0] / (7.658 * X[:, 3]))
        ycal = ycal_1 * ycal_2 * ycal_3 * ycal_4 * ycal_5 * ycal_6 * ycal_7 * ycal_8
        b = np.squeeze(b, -1)
        return b, ycal
                
    def constr_batch(self, X):
        b, ycal = self.intermediates(X)
        return -1 * (b)

    def target_batch(self, X):
        b, ycal = self.intermediates(X)
        obj = ((self.parameters['yobs'][None, :] - ycal) ** 2).sum(-1)
        return obj