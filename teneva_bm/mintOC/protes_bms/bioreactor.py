import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmBioreactor(Bm):
    def __init__(self, d=100, n=101, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Bioreactor problem from the mintOC collection describes an substrate 
            that is converted to a product by the biomass in the reactor. It has three 
            states and a control that is describing the feed concentration of the substrate.
            Details: https://mintoc.de/index.php/Bioreactor
        """)
        
        self.set_grid(28.7, 40)
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
            'D': 0.15,    # Dilution
            'K_i': 22,    # Rate coefficient
            'K_m': 1.2,   # Rate coefficient
            'x3_m': 50,   # Rate coefficient
            'Y_xs': 0.4,  # Substrate to Biomass rate
            'alpha': 2.2, # Linear slope
            'beta': 0.2,  # Linear intercept
            'mu_m': 0.48  # Maximal growth rate
        }

    def set_time(self):
        self.t = np.linspace(0, 48, self.d)
    
    def _ode(self, x2_f):
        x2_f_interpolate = interp1d(self.t, x2_f, kind='nearest', fill_value='extrapolate')

        def f(t, x):
            x1, x2, x3, x4 = x
            x2_f = x2_f_interpolate(t)

            mu = self.parameters['mu_m'] * \
                 (1 - x3 / self.parameters['x3_m']) * \
                 x2 / (self.parameters['K_m'] + x2 + x2 ** 2 / self.parameters['K_i'])
            
            dx1 = (mu - self.parameters['D']) * x1
            dx2 = self.parameters['D'] * (x2_f - x2) - (mu / self.parameters['Y_xs']) * x1
            dx3 = - self.parameters['D'] * x3 + (self.parameters['alpha'] * mu + self.parameters['beta']) * x1
            dx4 = self.parameters['D'] * (x2_f - x3) ** 2
            return [dx1, dx2, dx3, dx4]

        sol = solve_ivp(f, [self.t[0], self.t[-1]], [6.5, 12, 22, 0], t_eval=self.t)
        return {'x': sol.y, 'success': sol.success}

    def _obj(self, x):
        x1, x2, x3, x4 = x
        return x4[-1]

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