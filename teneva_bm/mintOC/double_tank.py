import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmDoubleTank(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Double Tank problem from the mintOC collection is a basic example for a switching system. 
            It contains the dynamics of an upper and a lower tank, connected to each other with a pipe. 
            The goal is to minimize the deviation of a certain fluid level k2 in the lower tank. 
            Details: https://mintoc.de/index.php/Double_Tank
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
        self.state_initial = [2, 2, 0]
        self.parameters = {'k1': 2, 'k2': 3}

    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2, x3 = state
            sigma = control_interpolate(t)
            k1, k2 = self.parameters.values()
            dx1 = sigma + 1 - np.sqrt(x1)
            dx2 = np.sqrt(x1) - np.sqrt(x2)
            dx3 = k1 * (x2 - k2) ** 2
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

    def constr(self, control):
        sol = self._ode(control)
        c = self.constr_penalty * ~sol['success']
        return c