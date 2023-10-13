import numpy as np
from teneva_bm import Bm


class BmHsFunc025(Bm):
    def __init__(self, d=3, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 025 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem without constraints:
            .--------------------.
            | F(x | p, t) -> min |
            .--------------------.
            p - parameters
                ai = range(1, 99)
                u = 25 + (-50 * log(ai / 100)) ** (2 / 3)
            x - continuous control
                x[0] | >= 0.1 | <= 100
                x[1] | >= 0   | <= 25.6
                x[2] | >= 0   | <= 5
            t - intermediates
                t = (-1 / 100) * ai + np.exp(-1 / x[0]) * (u - x[1]) ** x[2]
            F - objective function
                (t ** 2).sum()
            The exact global minimum is known:
                y = 0
                x[0] = 50
                x[1] = 25
                x[2] = 1.5
            Hyperparameters:
                * The dimension d should be 3
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([0.1, 0, 0], [100, 25.6, 5])
        # self.set_min(x=[50, 25, 1.5], y=0)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 3}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    def set_parameters(self):
        ai = np.arange(1, 99)
        u = 25 + (-50 * np.log(ai / 100)) ** (2 / 3)
        self.parameters = {'ai': ai, 'u': u}

    def intermediates(self, X):
        t = (-1 / 100) * self.parameters['ai'] + np.exp(-1 / X[:, 0])[:, None] * \
            (self.parameters['u'][None, :] - X[:, 1][:, None]) ** X[:, 2][:, None]
        return t

    def target_batch(self, X):
        t = self.intermediates(X)
        return (t ** 2).sum(-1)


if __name__ == '__main__':
    bm = BmHsFunc025()
    bm.prep()
    y = bm(bm.x_min_real)
    print(y, bm.y_min_real)
