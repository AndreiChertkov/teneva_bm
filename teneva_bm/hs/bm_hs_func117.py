import numpy as np
from teneva_bm import Bm


class BmHsFunc117(Bm):
    def __init__(self, d=15, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 117 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                a = [[-16 ,  2 ,  0 ,  1 ,  0  ], 
                     [ 0  , -2 ,  0 , 0.4,  2  ], 
                     [-3.5,  0 ,  2 ,  0 ,  0  ], 
                     [ 0  , -2 ,  0 , -4 , -1  ], 
                     [ 0  , -9 , -2 ,  1 , -2.8], 
                     [ 2  ,  0 , -4 ,  0 ,  0  ],
                     [-1  , -1 , -1 , -1 , -1  ],
                     [-1  , -2 , -3 , -2 , -1  ],
                     [ 1  ,  2 ,  3 ,  4 ,  5  ],
                     [ 1  ,  1 ,  1 ,  1 ,  1  ]]
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
                x[5] | >= 0
                x[6] | >= 0
                x[7] | >= 0
                x[8] | >= 0
                x[9] | >= 0
                x[10] | >= 0
                x[11] | >= 0
                x[12] | >= 0
                x[13] | >= 0
                x[14] | >= 0
            F - objective function
                (-1) * b @ x[:10] + (c @ x[10:]) @ x[10:] + 2 * d @ x[10:] ** 3
            C - constraint function
                2 * c @ x[10:] + 3 * d * x[10:] ** 2 - a @ x[10:] + e >= 0
            The exact global minimum is approx. known:
                y ~= 32.349
                x[0] ~= 0
                x[1] ~= 0
                x[2] ~= 5.174
                x[3] ~= 0
                x[4] ~= 3.061
                x[5] ~= 11.840
                x[6] ~= 0
                x[7] ~= 0
                x[8] ~= 0.104
                x[9] ~= 0
                x[10] ~= 0.300
                x[11] ~= 0.333
                x[12] ~= 0.400
                x[13] ~= 0.428
                x[14] ~= 0.224
            Hyperparameters: 
                * The dimension d should be 15
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [+10, +10, +10, +10, +10, +10, +10, +10, +10, +10, +10, +10, +10, +10, +10]
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 15}

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
            'a': np.array([[-16 ,  2 ,  0 ,  1 ,  0  ],
                           [ 0  , -2 ,  0 , 0.4,  2  ], 
                           [-3.5,  0 ,  2 ,  0 ,  0  ], 
                           [ 0  , -2 ,  0 , -4 , -1  ],
                           [ 0  , -9 , -2 ,  1 , -2.8], 
                           [ 2  ,  0 , -4 ,  0 ,  0  ],
                           [-1  , -1 , -1 , -1 , -1  ],
                           [-1  , -2 , -3 , -2 , -1  ],
                           [ 1  ,  2 ,  3 ,  4 ,  5  ],
                           [ 1  ,  1 ,  1 ,  1 ,  1  ]]),
            'b': np.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]),
            'c': np.array([[ 30, -20, -10,  32, -10], 
                           [-20,  39, -6 , -31,  32], 
                           [-10, -6 ,  10, -6 , -10], 
                           [ 32, -31, -6 ,  39, -20], 
                           [-10,  32, -10, -20,  30]]),
            'd': np.array([-15, -27, -36, -18, -12]),
            'e': np.array([-15, -27, -36, -18, -12])
        }

    def _constr_batch(self, X):
        c_1 = 2 * X[:, 10:] @ self.parameters['c']
        c_2 = 3 * X[:, 10:] ** 2 @ self.parameters['d']
        c_3 = 1 * X[:, :10] @ self.parameters['a']
        c_4 = self.parameters['e']
        c = -1 * (c_1 + c_2[:, None] - c_3 + c_4[None, :]).T
        return -1 * c
    
    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return (-1) * X[:, :10] @ self.parameters['b'] + \
               ((X[:, 10:] @ self.parameters['c']) * X[:, 10:]).sum(-1) + \
               2 * (X[:, 10:] ** 3) @ self.parameters['d']