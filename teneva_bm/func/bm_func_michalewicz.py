import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Michalewicz function (continuous).
    The dimension and mode size may be any (default are d=50, n=15).
    Default grid limits are [0, pi]; the exact global minimum is known
    only for the case of dimensions 2, 5, and 10; in this cases, only the
    corresponding value of the function is known, but not the argument.
    See https://www.sfu.ca/~ssurjano/michal.html for details.
    See also Charlie Vanaret, Jean-Baptiste Gotteland, Nicolas Durand,
    Jean-Marc Alliot. "Certified global minima for a benchmark of difficult
    optimization problems". arXiv preprint arXiv:2003.09867 2020
    (the function has d! local minima, and it is multimodal).
    Note that the method "build_cores" for construction of the function
    in the TT-format on the discrete grid is available.
"""


class BmFuncMichalewicz(Bm):
    def __init__(self, d=50, n=15, name='FuncMichalewicz', desc=DESC):
        super().__init__(d, n, name, desc)

        self.set_grid(0., np.pi)

        if self.d == 2:
            self.set_min(x=[2.20, 1.57], y=-1.8013)
        if self.d == 5:
            self.set_min(y=-4.687658)
        if self.d == 10:
            self.set_min(y=-9.66015)

        self.with_cores = True

    @property
    def is_func(self):
        return True

    def set_opts(self, m=10.):
        """Setting options specific to this benchmark.

        Args:
            m (float): parameter of the function.

        """
        self.opt_m = m

    def _cores(self, X):
        Y = self._cores_add(
            [np.sin(x) * np.sin(i*x**2/np.pi)**(2*self.opt_m)
                for i, x in enumerate(X.T, 1)])
        Y[-1] *= -1.
        return Y

    def _f_batch(self, X):
        y1 = np.sin(((np.arange(self.d) + 1) * X**2 / np.pi))
        y = -np.sum(np.sin(X) * y1**(2 * self.opt_m), axis=1)
        return y

    def _f_pt(self, x):
        """Draft."""
        d = torch.tensor(self.d)
        par_m = torch.tensor(self.opt_m)
        pi = torch.tensor(np.pi)
        y1 = torch.sin(((torch.arange(d) + 1) * x**2 / pi))
        y = -torch.sum(torch.sin(x) * y1**(2 * par_m))
        return y


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmFuncMichalewicz().prep()
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
