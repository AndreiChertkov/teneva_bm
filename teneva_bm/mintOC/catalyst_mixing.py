import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmCatalystMixing(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Catalyst Mixing problem from the mintOC collection seeks 
            an optimal policy for mixing two catalysts along the length of 
            a tubular plug ow reactor involving several reactions.
            Details: https://mintoc.de/index.php/Catalyst_mixing_problem
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
        self.time = np.linspace(0, 1, self.d)
        self.state_initial = [1, 0]

    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2 = state
            w = control_interpolate(t)
            dx1 = w * (10 * x2 - x1)
            dx2 = w * (x1 - 10 * x2) - (1 - w) * x2
            return [dx1, dx2]

        sol = solve_ivp(f, [self.time[0], self.time[-1]], y0=self.state_initial, t_eval=self.time)
        return {'state': sol.y, 'success': sol.success}

    def _obj(self, state):
        x1, x2 = state
        return -1 + x1[-1] + x2[-1]

    def target(self, control):
        state = self._ode(control)['state']
        obj = self._obj(state)
        return obj
    
    # ---------------- constraints ----------------
    @property
    def with_constr(self):
        return True

    def constr(self, control):
        sol = self._ode(control)
        c = self.constr_penalty * ~sol['success']
        return c