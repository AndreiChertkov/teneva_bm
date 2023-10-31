import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmBatchReactor(Bm):
    def __init__(self, d=100, n=101, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc(""" Batch Reactor """)
        self.set_grid(298, 398)
        self.set_grid_kind('uni')

    @property
    def args_constr(self):
        return {'n': 101}

    @property
    def identity(self):
        return ['d']

    # @property
    # def is_tens(self):
    #     return True
    
    @property
    def is_func(self):
        return True
    
    def target(self, T):
        t = np.linspace(0, 1, self.d)
        T_interpolate = interp1d(t, T, kind='nearest', fill_value='extrapolate')

        def f(t, x):
            x1, x2 = x
            T = T_interpolate(t)
            k1 = 4000 * np.exp(-2500 / T)
            k2 = 6.2e5 * np.exp(-5000 / T)
            dx1 = -k1 * x1 ** 2
            dx2 = k1 * x1 ** 2 - k2 * x2
            return [dx1, dx2]

        sol = solve_ivp(f, [t[0], t[-1]], [1, 0], t_eval=t)
        if sol.success:
            x1, x2 = sol.y
            y = -x2[-1]
        else:
            y = 1e+10
        return y