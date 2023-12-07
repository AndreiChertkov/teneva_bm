import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmHangingChain(Bm):
    def __init__(self, d=100, n=31, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Hanging Chain from the mintOC collection is concerned 
            with finding a chain (of uniform density) of length L 
            suspendend between two points a, b with minimal potential energy. 
            Details: https://mintoc.de/index.php/Hanging_chain_problem
        """)
        
        self.set_grid(-10, 20)
        self.set_grid_kind('uni')
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()
    
    @property
    def is_func(self):
        return True
    
    def set_parameters(self):
        self.time = np.linspace(0, 1, self.d)
        self.state_initial = [1, 0, 0]
    
    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2, x3 = state
            u = control_interpolate(t)
            k = np.sqrt(1 + u ** 2)
            dx1 = u
            dx2 = x1 * k
            dx3 = k
            return [dx1, dx2, dx3]

        sol = solve_ivp(f, [self.time[0], self.time[-1]], y0=self.state_initial, t_eval=self.time)
        return {'state': sol.y, 'success': sol.success}

    def _obj(self, state):
        x1, x2, x3 = state
        return x2[-1]

    def target(self, control):
        state = self._ode(control)['state']
        obj = self._obj(state)
        return obj
    
    # ---------------- constraints ----------------
    @property
    def with_constr(self):
        return True
    
    def _constr(self, control):
        sol = self._ode(control)
        if sol['success']:
            x1, x2, x3 = sol['state']
            c1 = np.abs(x1[-1] - 3)
            c2 = np.abs(x3[-1] - 4)
            c3 = -1 * x1
            c4 = -1 * x2
            c5 = -1 * x3
            c6 = -1 * (10 - x1)
            c7 = -1 * (10 - x2)
            c8 = -1 * (10 - x3)
            return np.hstack([0, c1, c2, c3, c4, c5, c6, c7, c8])
        else:
            return np.array([self.constr_penalty] + [0] * 8)
        
    def constr(self, u):
        c = self._constr(u)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c