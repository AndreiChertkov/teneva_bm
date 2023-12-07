import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmFuller(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Fuller's problem from the mintOC collection.
            Details: https://mintoc.de/index.php/Fuller%27s_problem
        """)

        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'n': 2}
    
    @property
    def is_tens(self):
        return True

    def set_parameters(self):
        self.time = np.linspace(0, 10, self.d)
        self.state_initial = [0.01, 0, 0]

    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2, x3 = state
            w = control_interpolate(t)
            dx1 = x2
            dx2 = 1 - 2 * w
            dx3 = x1 ** 2
            return [dx1, dx2, dx3]

        sol = solve_ivp(f, [self.time[0], self.time[-1]], y0=self.state_initial, t_eval=self.time)
        return {'state': sol.y, 'success': sol.success}

    def _obj(self, state):
        x1, x2, x3 = state
        return x3[-1]

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
            c1 = np.abs(x1[-1] - 0.01)
            c2 = np.abs(x2[-1])
            return np.hstack([0, c1, c2])
        else:
            return np.array([self.constr_penalty] + [0] * 2)
        
    def constr(self, control):
        c = self._constr(control)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c