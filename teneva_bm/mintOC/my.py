import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmMy(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""_""")

    @property
    def args_constr(self):
        return {'n': 2}

    @property
    def identity(self):
        return ['d']

    @property
    def is_tens(self):
        return True

    @property
    def with_constr(self):
        return True
    
    def _constr(self, i):
        i_pad = np.zeros(i.shape[0] + 3, dtype=i.dtype)
        i_pad[3:] = i
        c_1 = -1 * (i_pad[3:] - (i_pad[2:-1] - i_pad[1:-2]))
        c_2 = -1 * (i_pad[3:] - (i_pad[2:-1] - i_pad[0:-3]))
        return np.hstack([c_1, c_2])
    
    def constr(self, i):
        c = self._constr(i)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c
    
    def target(self, i):
        t = np.arange(self.d) * 0.05
        i_interpolate = interp1d(t, i, kind='nearest', fill_value='extrapolate')
        def f(t, x): 
            dx = x ** 3 - i_interpolate(t)
            return dx
        sol = solve_ivp(f, [t[0], t[-1]], [0.8], t_eval=t)
        if sol.success:
            x = sol.y[0]
            y = sum(0.5 * (x - 0.7) ** 2)
        else:
            y = 1e+10
        return y