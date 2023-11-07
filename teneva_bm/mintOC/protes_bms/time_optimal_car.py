import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmTimeOptimalCar(Bm):
    def __init__(self, d=100, n=4, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Time Optimal Car problem from the mintOC collection consists of starting
            and stopping a car in minimum for a fixed distance (300 units).
            Details: https://mintoc.de/index.php/Time_optimal_car_problem          
            """)
    
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_time()

    @property
    def args_constr(self):
        return {'n': 4}
    
    @property
    def identity(self):
        return ['d']
    
    @property
    def is_tens(self):
        return True

    def set_time(self):
        self.t = np.linspace(0, 1, self.d)
    
    def _ode(self, u):
        u_interpolate = interp1d(self.t, u, kind='nearest', fill_value='extrapolate')

        def f(t, x):
            x1, x2 = x
            u = u_interpolate(t)
            dx1 = x2
            dx2 = u
            return [dx1, dx2]

        sol = solve_ivp(f, [self.t[0], self.t[-1]], [0, 0], t_eval=self.t)
        return {'x': sol.y, 'success': sol.success}

    def _obj(self, x):
        x1, x2 = x
        return x2[-1]

    def target(self, u):
        x = self._ode(u)['x']
        y = self._obj(x)
        return y
    
    # ---------------- constraints ----------------
    @property
    def with_constr(self):
        return True
    
    def _constr(self, u):
        sol = self._ode(u)
        if sol['success']:
            x1, x2 = sol['x']
            c1 = np.abs(x1[-1] - 300)
            c2 = np.abs(x2[-1])
            c3 = -1 * x1
            c4 = -1 * x2
            c5 = -1 * (33 - x1)
            c6 = -1 * (330 - x2)
            return np.hstack([0, c1, c2, c3, c4, c5, c6])
        else:
            return np.array([self.constr_penalty] + [0] * 6)
        
    def constr(self, u):
        c = self._constr(u)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c