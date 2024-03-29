import numpy as np
from teneva_bm import Bm


class BmHsFunc119(Bm):
    def __init__(self, d=16, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 119 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .------------------------------.
            | F(x) -> min s.t. C(x) = True |
            .------------------------------.
            x - continuous control
                x[0] | >= 0 | <= 5
                x[1] | >= 0 | <= 5
                x[2] | >= 0 | <= 5
                x[3] | >= 0 | <= 5
                x[4] | >= 0 | <= 5
                x[5] | >= 0 | <= 5
                x[6] | >= 0 | <= 5
                x[7] | >= 0 | <= 5
                x[8] | >= 0 | <= 5
                x[9] | >= 0 | <= 5
                x[10] | >= 0 | <= 5
                x[11] | >= 0 | <= 5
                x[12] | >= 0 | <= 5
                x[13] | >= 0 | <= 5
                x[14] | >= 0 | <= 5
                x[15] | >= 0 | <= 5
            F - objective function
                (x[0] ** 2 + x[0] + 1) * (x[0] ** 2 + x[0] + 1) + 
                (x[0] ** 2 + x[0] + 1) * (x[3] ** 2 + x[3] + 1) + 
                (x[0] ** 2 + x[0] + 1) * (x[6] ** 2 + x[6] + 1) + 
                (x[0] ** 2 + x[0] + 1) * (x[7] ** 2 + x[7] + 1) + 
                (x[0] ** 2 + x[0] + 1) * (x[15] ** 2 + x[15] + 1) + 
                (x[1] ** 2 + x[1] + 1) * (x[1] ** 2 + x[1] + 1) + 
                (x[1] ** 2 + x[1] + 1) * (x[2] ** 2 + x[2] + 1) + 
                (x[1] ** 2 + x[1] + 1) * (x[6] ** 2 + x[6] + 1) + 
                (x[1] ** 2 + x[1] + 1) * (x[9] ** 2 + x[9] + 1) + 
                (x[2] ** 2 + x[2] + 1) * (x[2] ** 2 + x[2] + 1) + 
                (x[2] ** 2 + x[2] + 1) * (x[6] ** 2 + x[6] + 1) + 
                (x[2] ** 2 + x[2] + 1) * (x[8] ** 2 + x[8] + 1) + 
                (x[2] ** 2 + x[2] + 1) * (x[9] ** 2 + x[9] + 1) + 
                (x[2] ** 2 + x[2] + 1) * (x[13] ** 2 + x[13] + 1) + 
                (x[3] ** 2 + x[3] + 1) * (x[3] ** 2 + x[3] + 1) + 
                (x[3] ** 2 + x[3] + 1) * (x[6] ** 2 + x[6] + 1) + 
                (x[3] ** 2 + x[3] + 1) * (x[10] ** 2 + x[10] + 1) + 
                (x[3] ** 2 + x[3] + 1) * (x[14] ** 2 + x[14] + 1) + 
                (x[4] ** 2 + x[4] + 1) * (x[4] ** 2 + x[4] + 1) + 
                (x[4] ** 2 + x[4] + 1) * (x[5] ** 2 + x[5] + 1) + 
                (x[4] ** 2 + x[4] + 1) * (x[9] ** 2 + x[9] + 1) + 
                (x[4] ** 2 + x[4] + 1) * (x[11] ** 2 + x[11] + 1) + 
                (x[4] ** 2 + x[4] + 1) * (x[15] ** 2 + x[15] + 1) + 
                (x[5] ** 2 + x[5] + 1) * (x[5] ** 2 + x[5] + 1) + 
                (x[5] ** 2 + x[5] + 1) * (x[7] ** 2 + x[7] + 1) + 
                (x[5] ** 2 + x[5] + 1) * (x[14] ** 2 + x[14] + 1) + 
                (x[6] ** 2 + x[6] + 1) * (x[6] ** 2 + x[6] + 1) + 
                (x[6] ** 2 + x[6] + 1) * (x[10] ** 2 + x[10] + 1) + 
                (x[6] ** 2 + x[6] + 1) * (x[12] ** 2 + x[12] + 1) + 
                (x[7] ** 2 + x[7] + 1) * (x[7] ** 2 + x[7] + 1) + 
                (x[7] ** 2 + x[7] + 1) * (x[9] ** 2 + x[9] + 1) + 
                (x[7] ** 2 + x[7] + 1) * (x[14] ** 2 + x[14] + 1) + 
                (x[8] ** 2 + x[8] + 1) * (x[8] ** 2 + x[8] + 1) + 
                (x[8] ** 2 + x[8] + 1) * (x[11] ** 2 + x[11] + 1) + 
                (x[8] ** 2 + x[8] + 1) * (x[15] ** 2 + x[15] + 1) + 
                (x[9] ** 2 + x[9] + 1) * (x[9] ** 2 + x[9] + 1) + 
                (x[9] ** 2 + x[9] + 1) * (x[13] ** 2 + x[13] + 1) + 
                (x[10] ** 2 + x[10] + 1) * (x[10] ** 2 + x[10] + 1) + 
                (x[10] ** 2 + x[10] + 1) * (x[12] ** 2 + x[12] + 1) + 
                (x[11] ** 2 + x[11] + 1) * (x[11] ** 2 + x[11] + 1) + 
                (x[11] ** 2 + x[11] + 1) * (x[13] ** 2 + x[13] + 1) + 
                (x[12] ** 2 + x[12] + 1) * (x[12] ** 2 + x[12] + 1) + 
                (x[12] ** 2 + x[12] + 1) * (x[13] ** 2 + x[13] + 1) + 
                (x[13] ** 2 + x[13] + 1) * (x[13] ** 2 + x[13] + 1) + 
                (x[14] ** 2 + x[14] + 1) * (x[14] ** 2 + x[14] + 1) + 
                (x[15] ** 2 + x[15] + 1) * (x[15] ** 2 + x[15] + 1)
            C - constraint function
                0.22 * x[0] + 0.2 * x[1] + 0.19 * x[2] + 0.25 * x[3] + 0.15 * x[4] + 
                0.11 * x[5] + 0.12 * x[6] + 0.13 * x[7] + x[8] - 2.5 = 0
                (-1.46) * x[0] - 1.3 * x[2] + 1.82 * x[3] - 1.15 * x[4] + 0.8 * x[6] + x[9] - 1.1 = 0
                1.29 * x[0] - 0.89 * x[1] - 1.16 * x[4] - 0.96 * x[5] - 0.49 * x[7] + x[10] + 3.1 = 0
                (-1.1) * x[0] - 1.06 * x[1] + 0.95 * x[2] - 0.54 * x[3] - 1.78 * x[5] - 0.41 * x[6] + x[11] + 3.5 = 0
                (-1.43) * x[3] + 1.51 * x[4] + 0.59 * x[5] - 0.33 * x[6] - 0.43 * x[7] + x[12] - 1.3 = 0
                (-1.72) * x[1] - 0.33 * x[2] + 1.62 * x[4] + 1.24 * x[5] + 0.21 * x[6] - 0.26 * x[7] + x[13] - 2.1 = 0
                1.12 * x[0] + 0.31 * x[3] + 1.12 * x[6] - 0.36 * x[8] + x[14] - 2.3 = 0
                0.45 * x[1] + 0.26 * x[2] - 1.1 * x[3] + 0.58 * x[4] - 1.03 * x[6] + 0.1 * x[7] + x[15] + 1.5 = 0
            The exact global minimum is approx. known:
                y ~= 244.900
                x[0] ~= 0.040
                x[1] ~= 0.792
                x[2] ~= 0.203
                x[3] ~= 0.844
                x[4] ~= 1.270
                x[5] ~= 0.935
                x[6] ~= 1.682
                x[7] ~= 0.155
                x[8] ~= 1.568
                x[9] ~= 0
                x[10] ~= 0
                x[11] ~= 0
                x[12] ~= 0.660
                x[13] ~= 0
                x[14] ~= 0.674
                x[15] ~= 0
            Hyperparameters: 
                * The dimension d should be 16
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        )
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)

    @property
    def args_constr(self):
        return {'d': 16}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    @property
    def with_constr(self):
        return True

    def _constr_batch(self, X):
        c_1 = np.abs(
            0.22 * X[:, 0] + 0.2 * X[:, 1] + 0.19 * X[:, 2] + 0.25 * X[:, 3] + \
            0.15 * X[:, 4] + 0.11 * X[:, 5] + 0.12 * X[:, 6] + 0.13 * X[:, 7] + X[:, 8] - 2.5
        )
        c_2 = np.abs((-1.46) * X[:, 0] - 1.3 * X[:, 2] + 1.82 * X[:, 3] - 1.15 * X[:, 4] + 0.8 * X[:, 6] + X[:, 9] - 1.1)
        c_3 = np.abs(1.29 * X[:, 0] - 0.89 * X[:, 1] - 1.16 * X[:, 4] - 0.96 * X[:, 5] - 0.49 * X[:, 7] + X[:, 10] + 3.1)
        c_4 = np.abs(
            (-1.1) * X[:, 0] - 1.06 * X[:, 1] + 0.95 * X[:, 2] - 0.54 * X[:, 3] - \
            1.78 * X[:, 5] - 0.41 * X[:, 6] + X[:, 11] + 3.5
        )
        c_5 = np.abs((-1.43) * X[:, 3] + 1.51 * X[:, 4] + 0.59 * X[:, 5] - 0.33 * X[:, 6] - 0.43 * X[:, 7] + X[:, 12] - 1.3)
        c_6 = np.abs(
            (-1.72) * X[:, 1] - 0.33 * X[:, 2] + 1.62 * X[:, 4] + 
            1.24 * X[:, 5] + 0.21 * X[:, 6] - 0.26 * X[:, 7] + X[:, 13] - 2.1
        )
        c_7 = np.abs(1.12 * X[:, 0] + 0.31 * X[:, 3] + 1.12 * X[:, 6] - 0.36 * X[:, 8] + X[:, 14] - 2.3)
        c_8 = np.abs(
            0.45 * X[:, 1] + 0.26 * X[:, 2] - 1.1 * X[:, 3] + 0.58 * X[:, 4] - 
            1.03 * X[:, 6] + 0.1 * X[:, 7] + X[:, 15] + 1.5
        )
        return np.array([c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8])

    def constr_batch(self, X):
        c = self._constr_batch(X)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c

    def target_batch(self, X):
        return  (X[:, 0] ** 2 + X[:, 0] + 1) * (X[:, 0] ** 2 + X[:, 0] + 1) + \
                (X[:, 0] ** 2 + X[:, 0] + 1) * (X[:, 3] ** 2 + X[:, 3] + 1) + \
                (X[:, 0] ** 2 + X[:, 0] + 1) * (X[:, 6] ** 2 + X[:, 6] + 1) + \
                (X[:, 0] ** 2 + X[:, 0] + 1) * (X[:, 7] ** 2 + X[:, 7] + 1) + \
                (X[:, 0] ** 2 + X[:, 0] + 1) * (X[:, 15] ** 2 + X[:, 15] + 1) + \
                (X[:, 1] ** 2 + X[:, 1] + 1) * (X[:, 1] ** 2 + X[:, 1] + 1) + \
                (X[:, 1] ** 2 + X[:, 1] + 1) * (X[:, 2] ** 2 + X[:, 2] + 1) + \
                (X[:, 1] ** 2 + X[:, 1] + 1) * (X[:, 6] ** 2 + X[:, 6] + 1) + \
                (X[:, 1] ** 2 + X[:, 1] + 1) * (X[:, 9] ** 2 + X[:, 9] + 1) + \
                (X[:, 2] ** 2 + X[:, 2] + 1) * (X[:, 2] ** 2 + X[:, 2] + 1) + \
                (X[:, 2] ** 2 + X[:, 2] + 1) * (X[:, 6] ** 2 + X[:, 6] + 1) + \
                (X[:, 2] ** 2 + X[:, 2] + 1) * (X[:, 8] ** 2 + X[:, 8] + 1) + \
                (X[:, 2] ** 2 + X[:, 2] + 1) * (X[:, 9] ** 2 + X[:, 9] + 1) + \
                (X[:, 2] ** 2 + X[:, 2] + 1) * (X[:, 13] ** 2 + X[:, 13] + 1) + \
                (X[:, 3] ** 2 + X[:, 3] + 1) * (X[:, 3] ** 2 + X[:, 3] + 1) + \
                (X[:, 3] ** 2 + X[:, 3] + 1) * (X[:, 6] ** 2 + X[:, 6] + 1) + \
                (X[:, 3] ** 2 + X[:, 3] + 1) * (X[:, 10] ** 2 + X[:, 10] + 1) + \
                (X[:, 3] ** 2 + X[:, 3] + 1) * (X[:, 14] ** 2 + X[:, 14] + 1) + \
                (X[:, 4] ** 2 + X[:, 4] + 1) * (X[:, 4] ** 2 + X[:, 4] + 1) + \
                (X[:, 4] ** 2 + X[:, 4] + 1) * (X[:, 5] ** 2 + X[:, 5] + 1) + \
                (X[:, 4] ** 2 + X[:, 4] + 1) * (X[:, 9] ** 2 + X[:, 9] + 1) + \
                (X[:, 4] ** 2 + X[:, 4] + 1) * (X[:, 11] ** 2 + X[:, 11] + 1) + \
                (X[:, 4] ** 2 + X[:, 4] + 1) * (X[:, 15] ** 2 + X[:, 15] + 1) + \
                (X[:, 5] ** 2 + X[:, 5] + 1) * (X[:, 5] ** 2 + X[:, 5] + 1) + \
                (X[:, 5] ** 2 + X[:, 5] + 1) * (X[:, 7] ** 2 + X[:, 7] + 1) + \
                (X[:, 5] ** 2 + X[:, 5] + 1) * (X[:, 14] ** 2 + X[:, 14] + 1) + \
                (X[:, 6] ** 2 + X[:, 6] + 1) * (X[:, 6] ** 2 + X[:, 6] + 1) + \
                (X[:, 6] ** 2 + X[:, 6] + 1) * (X[:, 10] ** 2 + X[:, 10] + 1) + \
                (X[:, 6] ** 2 + X[:, 6] + 1) * (X[:, 12] ** 2 + X[:, 12] + 1) + \
                (X[:, 7] ** 2 + X[:, 7] + 1) * (X[:, 7] ** 2 + X[:, 7] + 1) + \
                (X[:, 7] ** 2 + X[:, 7] + 1) * (X[:, 9] ** 2 + X[:, 9] + 1) + \
                (X[:, 7] ** 2 + X[:, 7] + 1) * (X[:, 14] ** 2 + X[:, 14] + 1) + \
                (X[:, 8] ** 2 + X[:, 8] + 1) * (X[:, 8] ** 2 + X[:, 8] + 1) + \
                (X[:, 8] ** 2 + X[:, 8] + 1) * (X[:, 11] ** 2 + X[:, 11] + 1) + \
                (X[:, 8] ** 2 + X[:, 8] + 1) * (X[:, 15] ** 2 + X[:, 15] + 1) + \
                (X[:, 9] ** 2 + X[:, 9] + 1) * (X[:, 9] ** 2 + X[:, 9] + 1) + \
                (X[:, 9] ** 2 + X[:, 9] + 1) * (X[:, 13] ** 2 + X[:, 13] + 1) + \
                (X[:, 10] ** 2 + X[:, 10] + 1) * (X[:, 10] ** 2 + X[:, 10] + 1) + \
                (X[:, 10] ** 2 + X[:, 10] + 1) * (X[:, 12] ** 2 + X[:, 12] + 1) + \
                (X[:, 11] ** 2 + X[:, 11] + 1) * (X[:, 11] ** 2 + X[:, 11] + 1) + \
                (X[:, 11] ** 2 + X[:, 11] + 1) * (X[:, 13] ** 2 + X[:, 13] + 1) + \
                (X[:, 12] ** 2 + X[:, 12] + 1) * (X[:, 12] ** 2 + X[:, 12] + 1) + \
                (X[:, 12] ** 2 + X[:, 12] + 1) * (X[:, 13] ** 2 + X[:, 13] + 1) + \
                (X[:, 13] ** 2 + X[:, 13] + 1) * (X[:, 13] ** 2 + X[:, 13] + 1) + \
                (X[:, 14] ** 2 + X[:, 14] + 1) * (X[:, 14] ** 2 + X[:, 14] + 1) + \
                (X[:, 15] ** 2 + X[:, 15] + 1) * (X[:, 15] ** 2 + X[:, 15] + 1)