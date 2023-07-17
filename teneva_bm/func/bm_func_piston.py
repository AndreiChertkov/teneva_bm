import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Analytical Piston function (continuous).
    The dimension is 7 and the mode size may be any (default is n=15),
    each mode has its own (substantially different) limits.
    See Vitaly Zankin, Gleb Ryzhakov, Ivan Oseledets. "Gradient descent
    based D-optimal design for the least-squares polynomial approximation".
    arXiv preprint arXiv:1806.06631 2018 for details.
"""


class BmFuncPiston(Bm):
    def __init__(self, d=7, n=15, name='FuncPiston', desc=DESC):
        super().__init__(d, n, name, desc)

        if self.d != 7:
            self.set_err('Dimension should be "7"')

        self.set_grid(
            [30., 0.005, 0.002, 1000,  90000, 290, 340],
            [60., 0.020, 0.010, 5000, 110000, 296, 360])

    @property
    def is_func(self):
        return True

    def _f_batch(self, X):
        _M  = X[:, 0]
        _S  = X[:, 1]
        _V0 = X[:, 2]
        _k  = X[:, 3]
        _P0 = X[:, 4]
        _Ta = X[:, 5]
        _T0 = X[:, 6]

        _A = _P0 * _S + 19.62 * _M - _k * _V0 / _S
        _Q = _P0 * _V0 / _T0
        _V = _S / 2 / _k * (np.sqrt(_A**2 + 4 * _k * _Q * _Ta) - _A)
        _C = 2 * np.pi * np.sqrt(_M / (_k + _S**2 * _Q * _Ta / _V**2))

        return _C

    def _f_pt(self, x):
        """Draft."""
        pi = torch.tensor(np.pi)

        _M  = x[0]
        _S  = x[1]
        _V0 = x[2]
        _k  = x[3]
        _P0 = x[4]
        _Ta = x[5]
        _T0 = x[6]

        _A = _P0 * _S + 19.62 * _M - _k * _V0 / _S
        _Q = _P0 * _V0 / _T0
        _V = _S / 2 / _k * (torch.sqrt(_A**2 + 4 * _k * _Q * _Ta) - _A)
        _C = 2 * pi * torch.sqrt(_M / (_k + _S**2 * _Q * _Ta / _V**2))

        return _C


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmFuncPiston().prep()
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
