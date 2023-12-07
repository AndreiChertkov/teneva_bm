import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmVanderpolOscillator(Bm):
    def __init__(self, d=100, n=100, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Van der Pol Oscillator problem from the mintOC collection is an oscillating 
            system with non-linear damping and self regulation. The aim is to control the 
            oscillation such that the system stays in a mean position.
            Details: https://mintoc.de/index.php/Van_der_Pol_Oscillator
        """)

        self.set_grid(-0.75, 0.75)
        self.set_grid_kind('uni')
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()
    
    @property
    def is_func(self):
        return True

    def set_parameters(self):
        self.time = np.linspace(0, 20, self.d)
        self.state_initial = [1, 0, 0]

    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2, x3 = state
            u = control_interpolate(t)
            dx1 = x2
            dx2 = u * (1 - x1 ** 2) * x2 - x1
            dx3 = x1 ** 2 + x2 ** 2 + u ** 2
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