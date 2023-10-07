import numpy as np
from teneva_bm import Bm


class BmHsFunc086(Bm):
    def __init__(self, d=5, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 086 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                a = [[-1  ,  2,  0,  1  ,  0  ], 
                     [ 0  , -2,  0,  0.4,  2  ], 
                     [-3.5,  0,  2,  0  ,  0  ], 
                     [ 0  , -2,  0, -4  , -1  ], 
                     [ 0  , -9, -2,  1  , -2.8], 
                     [ 2  ,  0, -4,  0  ,  0  ], 
                     [-1  , -1, -1, -1  , -1  ], 
                     [-1  , -2, -3, -2  , -1  ], 
                     [ 1  ,  2,  3,  4  ,  5  ], 
                     [ 1  ,  1,  1,  1  ,  1  ]]
                b = [-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]
                c = [[ 30, -20, -10,  32, -10], 
                     [-20,  39, -6 , -31,  32], 
                     [-10, -6 ,  10, -6 , -10], 
                     [ 32, -31, -6 ,  39, -20], 
                     [-10,  32, -10, -20,  30]]
                d = [-15, -27, -36, -18, -12]
                e = [-15, -27, -36, -18, -12]
            x - continuous control
                x[0] | >= 0
                x[1] | >= 0
                x[2] | >= 0
                x[3] | >= 0
                x[4] | >= 0
            F - objective function
                e @ x + (c @ x) @ x + d @ x ** 3
            C - constraint function
                a @ x - b >= 0
            The exact global minimum is approx. known:
                y ~= -32.349
                x[0] ~= 0.300
                x[1] ~= 0.333
                x[2] ~= 0.400
                x[3] ~= 0.428
                x[4] ~= 0.224
            Hyperparameters: 
                * The dimension d should be 5
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0, 0, 0, 0, 0], [+10, +10, +10, +10, +10])
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
            'a': np.array([
                [-16, 2, 0, 1, 0], 
                [ 0, -2, 0, 0.4, 2], 
                [-3.5, 0, 2, 0, 0], 
                [ 0, -2, 0, -4, -1], 
                [ 0, -9, -2, 1, -2.8], 
                [ 2, 0, -4, 0, 0], 
                [-1, -1, -1, -1, -1], 
                [-1, -2, -3, -2, -1], 
                [ 1, 2, 3, 4, 5], 
                [ 1, 1, 1, 1, 1]
            ]),
            'b': np.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]),
            'c': np.array([
                [ 30, -20, -10,  32, -10], 
                [-20,  39, -6 , -31,  32], 
                [-10, -6 ,  10, -6 , -10], 
                [ 32, -31, -6 ,  39, -20], 
                [-10,  32, -10, -20,  30]
            ]),
            'd': np.array([-15, -27, -36, -18, -12]),
            'e': np.array([-15, -27, -36, -18, -12]),
        }

    def _constr_batch(self, X):
        return -1 * (X @ self.parameters['a'].T - self.parameters['b']).T

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return X @ self.parameters['e'] + ((X @ self.parameters['c'].T) * X).sum(-1) + X @ self.parameters['d']