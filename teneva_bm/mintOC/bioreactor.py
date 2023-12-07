import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmBioreactor(Bm):
    def __init__(self, d=100, n=100, seed=42, name=None):
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
    
    @property
    def is_func(self):
        return True
    
    def set_parameters(self):
        self.time = np.linspace(0, 48, self.d)
        self.state_initial = [6.5, 12, 22, 0]
        self.parameters = {
            'D': 0.15,    # Dilution
            'K_i': 22,    # Rate coefficient
            'K_m': 1.2,   # Rate coefficient
            'P_m': 50,    # Rate coefficient
            'Y_xs': 0.4,  # Substrate to Biomass rate
            'alpha': 2.2, # Linear slope
            'beta': 0.2,  # Linear intercept
            'mu_m': 0.48  # Maximal growth rate
        }

    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2, x3, x4 = state
            x2_f = control_interpolate(t)
            D, K_i, K_m, P_m, Y_xs, alpha, beta, mu_m = self.parameters.values()
            mu = mu_m * (1 - x3 / P_m) * x2 / (K_m + x2 + x2 ** 2 / K_i)
            dx1 = (mu - D) * x1
            dx2 = D * (x2_f - x2) - (mu / Y_xs) * x1
            dx3 = -D * x3 + (alpha * mu + beta) * x1
            dx4 = D * (x2_f - x3) ** 2
            return [dx1, dx2, dx3, dx4]

        sol = solve_ivp(f, [self.time[0], self.time[-1]], y0=self.state_initial, t_eval=self.time)
        return {'state': sol.y, 'success': sol.success}
    
    def _obj(self, state):
        x1, x2, x3, x4 = state
        return x4[-1]

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