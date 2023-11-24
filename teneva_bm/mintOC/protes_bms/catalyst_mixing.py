import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmCatalystMixing(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Catalyst Mixing problem from the mintOC collection seeks an optimal policy 
            for mixing two catalysts "along the length of a tubular plug ow reactor involving several reactions"
            Details: https://mintoc.de/index.php/Catalyst_mixing_problem
        """)

        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_time()

    @property
    def args_constr(self):
        return {'n': 2}
    
    @property
    def identity(self):
        return ['d']
    
    @property
    def is_tens(self):
        return True
    
    def set_time(self):
        self.t = np.linspace(0, 1, self.d)

    def _ode(self, w):
        w_interpolate = interp1d(self.t, w, kind='nearest', fill_value='extrapolate')

        def f(t, x):
            x1, x2 = x
            w = w_interpolate(t)
            dx1 = w * (10 * x2 - x1)
            dx2 = w * (x1 - 10 * x2) - (1 - w) * x2
            return [dx1, dx2]

        sol = solve_ivp(f, [self.t[0], self.t[-1]], [1, 0], t_eval=self.t)
        return {'x': sol.y, 'success': sol.success}

    def _obj(self, x):
        x1, x2 = x
        return -1 + x1[-1] + x2[-1]

    def target(self, T):
        x = self._ode(T)['x']
        y = self._obj(x)
        return y
    
    # ---------------- constraints ----------------
    @property
    def with_constr(self):
        return True

    def constr(self, T):
        sol = self._ode(T)
        c = self.constr_penalty * ~sol['success']
        return c