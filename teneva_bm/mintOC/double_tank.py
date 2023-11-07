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
        self.set_time()
    
    @property
    def args_constr(self):
        return {'n': 2}
    
    @property
    def identity(self):
        return ['d']
    
    @property
    def is_tens(self):
        return True
    
    def set_parameters(self):
        self.parameters = {'k1': 2, 'k2': 3}

    def set_time(self):
        self.t = np.linspace(0, 10, self.d)
    
    def _ode(self, sigma):
        sigma_interpolate = interp1d(self.t, sigma, kind='nearest', fill_value='extrapolate')

        def f(t, x):
            x1, x2, x3 = x
            sigma = sigma_interpolate(t)
            dx1 = sigma + 1 - np.sqrt(x1)
            dx2 = np.sqrt(x1) - np.sqrt(x2)
            dx3 = self.parameters['k1'] * (x2 - self.parameters['k2']) ** 2
            return [dx1, dx2, dx3]

        sol = solve_ivp(f, [self.t[0], self.t[-1]], [2, 2, 0], t_eval=self.t)
        return {'x': sol.y, 'success': sol.success}

    def _obj(self, x):
        x1, x2, x3 = x
        return x3[-1]

    def target(self, sigma):
        x = self._ode(sigma)['x']
        y = self._obj(x)
        return y
    
    # ---------------- constraints ----------------
    @property
    def with_constr(self):
        return True

    def constr(self, sigma):
        sol = self._ode(sigma)
        c = self.constr_penalty * ~sol['success']
        return c