import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Rosenbrock function (continuous).
    The dimension and mode size may be any (default are d=50, n=15).
    Default grid limits are [-2.048, 2.048]; the exact global minimum
    is known: x = [1, ..., 1], y = 0.
    See https://www.sfu.ca/~ssurjano/rosen.html for details.
    See also the work Momin Jamil, Xin-She Yang. "A literature survey of
    benchmark functions for global optimization problems". Journal of
    Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
    ("105. Rosenbrock Function"; Continuous, Differentiable,
    Non-Separable, Scalable, Unimodal).
    Note that the method "build_cores" for construction of the function
    in the TT-format on the discrete grid is available.
"""


class BmFuncRosenbrock(Bm):
    def __init__(self, d=50, n=15, name='FuncRosenbrock', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(-2.048, +2.048)

        self.set_min(x=[1.]*self.d, y=0.)

        self.with_cores = True

    @property
    def is_func(self):
        return True

    def _cores(self, X):
        Y = []
        for i, x in enumerate(X.T):
            x2 = x*x
            if i == 0:
                G = np.zeros([1, len(x), 3])
                G[0, :, 0] = 1
                G[0, :, 1] = x2
                G[0, :, 2] = 100*(x2**2) + (1-x)**2
            elif i == self.d-1:
                G = np.zeros([3, len(x), 1])
                G[2, :, 0] = 1
                G[1, :, 0] = -200*x
                G[0, :, 0] = 100*x2
            else:
                G = np.zeros([3, len(x), 3])
                G[0, :, 0] = 1.
                G[2, :, 2] = 1.
                G[0, :, 1] = x2
                G[0, :, 2] = 100*x2 + 100*(x2**2) + (1-x)**2
                G[1, :, 2] = -200*x
            Y.append(G)
        return Y

    def _f_batch(self, X):
        y1 = 100. * (X[:, 1:] - X[:, :-1]**2)**2
        y2 = (X[:, :-1] - 1.)**2
        return np.sum(y1 + y2, axis=1)

    def _f_pt(self, x):
        """Draft."""
        y1 = 100. * (x[1:] - x[:-1]**2)**2
        y2 = (x[:-1] - 1.)**2
        return torch.sum(y1 + y2)


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmFuncRosenbrock().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+4)
    print(bm.info_history())

    text = 'Value at a random multi-index            :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at 3 random multi-indices          :  '
    i1 = [np.random.choice(k) for k in bm.n]
    i2 = [np.random.choice(k) for k in bm.n]
    i3 = [np.random.choice(k) for k in bm.n]
    I = [i1, i2, i3]
    y = bm[I]
    text += '; '.join([f'{y_cur:-10.3e}' for y_cur in y])
    print(text)

    text = 'TT-cores accuracy on train data          :  '
    Y = bm.build_cores()
    e = teneva.accuracy_on_data(Y, I_trn, y_trn)
    text += f'{e:-10.1e}'
    print(text)

    text = 'Value at the minimum (real vs calc)      :  '
    y_real = bm.y_min_real
    y_calc = bm(bm.x_min_real)
    text += f'{y_real:-10.3e}       /      {y_calc:-10.3e}'
    print(text)
