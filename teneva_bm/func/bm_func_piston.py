import numpy as np
import teneva
from teneva_bm.func.func import Func


class BmFuncPiston(Func):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Piston function (continuous).
            The dimension is 7 and the mode size may be any (default is n=16),
            each mode has its own different limits (with small random shift).
            See Vitaly Zankin, Gleb Ryzhakov, Ivan Oseledets. "Gradient descent
            based D-optimal design for the least-squares polynomial
            approximation". arXiv:1806.06631 2018 for details.
        """)

        self.set_grid(
            [30., 0.005, 0.002, 1000,  90000, 290, 340],
            [60., 0.020, 0.010, 5000, 110000, 296, 360], sh=True)

        if self.d != 7:
            self.set_err('Dimension should be "7"')

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 0.3320429515579626

    def target_batch(self, X):
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

    def _target_pt(self, x):
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
