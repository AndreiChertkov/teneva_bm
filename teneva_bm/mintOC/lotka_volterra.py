import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmLotkaVolterra(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Lotka Volterra problem from the mintOC collection looks for an optimal 
            fishing strategy to be performed on a fixed time horizon to bring the 
            biomasses of both predator as prey fish to a prescribed steady state. 
            Details: https://mintoc.de/index.php/Lotka_Volterra_fishing_problem                      
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
        self.time = np.linspace(0, 12, self.d)
        self.state_initial = [0.5, 0.7, 0]
        self.parameters = {'c0': 0.4, 'c1': 0.2}
    
    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2, x3 = state
            w = control_interpolate(t)
            c0, c1 = self.parameters.values()
            dx1 = x1 - x1 * x2 - c0 * x1 * w
            dx2 = - x2 + x1 * x2 - c1 * x2 * w
            dx3 = (x1 - 1) ** 2 + (x2 - 1) ** 2
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