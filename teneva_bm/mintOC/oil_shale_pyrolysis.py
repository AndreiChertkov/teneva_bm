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
        self.set_time()
    
    @property
    def identity(self):
        return ['d']
    
    @property
    def is_func(self):
        return True
    
    def set_parameters(self):
        self.parameters = {
            'a': np.exp([8.86, 24.25, 23.67, 18.75, 20.7]), # frequency factors
            'b': np.array([20.3, 37.4, 33.8, 28.2, 31.0]),  # activation energies
            'R': 1.9858775e-3 # universal gas constant
        }

    def set_time(self):
        self.t = np.linspace(0, 1, self.d)
    
    def _ode(self, T):
        T_interpolate = interp1d(self.t, T, kind='nearest', fill_value='extrapolate')

        def f(t, x):
            x1, x2, x3, x4 = x
            T = T_interpolate(t)
            k = self.parameters['a'] * np.exp(-self.parameters['b'] / (self.parameters['R'] * T))
            dx1 = - k[0] * x1 - (k[2] + k[3] + k[4]) * x1 * x2
            dx2 = k[0] * x1 - k[1] * x2 + k[2] * x1 * x2
            dx3 = k[1] * x2 + k[3] * x1 * x2
            dx4 = k[4] * x1 * x2
            return [dx1, dx2, dx3, dx4]

        sol = solve_ivp(f, [self.t[0], self.t[-1]], [1, 0, 0, 0], t_eval=self.t)
        return {'x': sol.y, 'success': sol.success}

    def _obj(self, x):
        x1, x2, x3, x4 = x
        return x2[-1]

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