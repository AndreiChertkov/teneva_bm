import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmBatchReactor(Bm):
    def __init__(self, d=100, n=101, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Batch Reactor problem from the mintOC collection describes the consecutive reaction 
            of some substance A via substance B into a desired product C. The system is interacted 
            with via the control variable T which stands for the temperature. 
            The goal is to produce as much of substance B as possible within the time limit.
            Details: https://mintoc.de/index.php/Batch_reactor
        """)

        self.set_grid(298, 398)
        self.set_grid_kind('uni')
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_time()

    @property
    def identity(self):
        return ['d']
    
    @property
    def is_func(self):
        return True
    
    def set_time(self):
        self.t = np.linspace(0, 1, self.d)

    def _ode(self, T):
        T_interpolate = interp1d(self.t, T, kind='nearest', fill_value='extrapolate')

        def f(t, x):
            x1, x2 = x
            T = T_interpolate(t)
            k1 = 4e3 * np.exp(-2.5e3 / T)
            k2 = 6.2e5 * np.exp(-5e3 / T)
            dx1 = -k1 * x1 ** 2
            dx2 = k1 * x1 ** 2 - k2 * x2
            return [dx1, dx2]

        sol = solve_ivp(f, [self.t[0], self.t[-1]], [1, 0], t_eval=self.t)
        return {'x': sol.y, 'success': sol.success}

    def _obj(self, x):
        x1, x2 = x
        return -x2[-1]

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