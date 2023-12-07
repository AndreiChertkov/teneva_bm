import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmOilShalePyrolysis(Bm):
    def __init__(self, d=100, n=101, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Oil Shale Pyrolysis problem from the mintOC collection describes the process, which
            starts with kerogen and is decomposed into pyrolytic bitumen, oil and gas, and residual carbon. 
            The objective is to maximize the fraction of pyrolytic bitumen.
            Details: https://mintoc.de/index.php/Oil_Shale_Pyrolysis
        """)
        
        self.set_grid(698.15, 748.15)
        self.set_grid_kind('uni')
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()
    
    @property
    def is_func(self):
        return True
    
    def set_parameters(self):
        self.time = np.linspace(0, 1, self.d)
        self.state_initial = [1, 0, 0, 0]
        self.parameters = {
            'a': np.exp([8.86, 24.25, 23.67, 18.75, 20.7]), # frequency factors
            'b': np.array([20.3, 37.4, 33.8, 28.2, 31.0]),  # activation energies
            'R': 1.9858775e-3                               # universal gas constant
        }

    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2, x3, x4 = state
            T = control_interpolate(t)
            a, b, R = self.parameters.values()
            k = a * np.exp(-b / (R * T))
            dx1 = -k[0] * x1 - (k[2] + k[3] + k[4]) * x1 * x2
            dx2 = k[0] * x1 - k[1] * x2 + k[2] * x1 * x2
            dx3 = k[1] * x2 + k[3] * x1 * x2
            dx4 = k[4] * x1 * x2
            return [dx1, dx2, dx3, dx4]

        sol = solve_ivp(f, [self.time[0], self.time[-1]], y0=self.state_initial, t_eval=self.time)
        return {'state': sol.y, 'success': sol.success}

    def _obj(self, state):
        x1, x2, x3, x4 = state
        return x2[-1]

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