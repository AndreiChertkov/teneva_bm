import numpy as np
from teneva_bm import Bm


class BmHsFunc057(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 057 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .-------------------------------------.
            | F(x | p, t) -> min s.t. C(x) = True |
            .-------------------------------------.
            p - parameters
                a = [8, 8, 10, 10, 10, 10, 12, 12, 12, 12, 
                     14, 14, 14, 16, 16, 16, 18, 18, 20, 20, 
                     20, 22, 22, 22, 24, 24, 24, 26, 26, 26, 
                     28, 28, 30, 30, 30, 32, 32, 34, 36, 36, 
                     38, 38, 40, 42]
                b = [0.49, 0.49, 0.48, 0.47, 0.48, 0.47, 0.46, 0.46, 0.45, 0.43, 
                     0.45, 0.43, 0.43, 0.44, 0.43, 0.43, 0.46, 0.45, 0.42, 0.42, 
                     0.43, 0.41, 0.41, 0.40, 0.42, 0.40, 0.40, 0.41, 0.40, 0.41, 
                     0.41, 0.40, 0.40, 0.40, 0.38, 0.41, 0.40, 0.40, 0.41, 0.38, 
                     0.40, 0.40, 0.39, 0.39]
            x - continuous control
                x[0] | >= 0.4
                x[1] | >= -4
            t - intermediates
                t = b - x[0] - (0.49 - x[0]) * exp(-x[1] * (a - 8))
            F - objective function
                (t ** 2).sum()
            C - constraint function
                0.49 * x[1] - x[0] * x[1] - 0.09 >= 0
            The exact global minimum is approx. known:
                y ~= 0.028
                x[0] ~= 0.420
                x[1] ~= 1.285
            Hyperparameters: 
                * The dimension d should be 2
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0.4, -4], [+10, +10])
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
            'a': np.array([
                8, 8, 10, 10, 10, 10, 12, 12, 12, 12, 
                14, 14, 14, 16, 16, 16, 18, 18, 20, 20, 
                20, 22, 22, 22, 24, 24, 24, 26, 26, 26, 
                28, 28, 30, 30, 30, 32, 32, 34, 36, 36, 
                38, 38, 40, 42
            ]),
            'b': np.array([
                0.49, 0.49, 0.48, 0.47, 0.48, 0.47, 0.46, 0.46, 0.45, 0.43, 
                0.45, 0.43, 0.43, 0.44, 0.43, 0.43, 0.46, 0.45, 0.42, 0.42, 
                0.43, 0.41, 0.41, 0.40, 0.42, 0.40, 0.40, 0.41, 0.40, 0.41, 
                0.41, 0.40, 0.40, 0.40, 0.38, 0.41, 0.40, 0.40, 0.41, 0.38, 
                0.40, 0.40, 0.39, 0.39
            ])
        }

    def intermediates(self, X):        
        t = self.parameters['b'][None, :] - X[:, 0][:, None] - \
            0.49 - X[:, 0][:, None] * np.exp(-X[:, 1][:, None] * (self.parameters['a'][None, :] - 8))
        return t

    def constr_batch(self, X):
        return -1 * (0.49 * X[:, 1] - X[:, 0] * X[:, 1] - 0.09)

    def target_batch(self, X):
        t = self.intermediates(X)
        return (t ** 2).sum(-1)