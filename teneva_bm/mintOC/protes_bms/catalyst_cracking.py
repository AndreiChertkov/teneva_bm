import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmCatalystCracking(Bm):
    def __init__(self, d=3, n=100, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Catalyst Cracking problem from the mintOC collection tries to determine 
            "reaction coefficients for the catalytic cracking of gas oil into gas and other byproducts."
            Details: https://mintoc.de/index.php/Catalytic_cracking_problem
        """)

        self.set_grid(0, 10)
        self.set_grid_kind('uni')
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()
        self.set_time()

    @property
    def args_constr(self):
        return {'d': 3}
    
    @property
    def identity(self):
        return ['d']
    
    @property
    def is_func(self):
        return True
    
    def set_parameters(self):
        self.parameters = {
            'z': np.array([
                [1.0000, 0.8105, 0.6208, 0.5258, 0.4345, 0.3903, 0.3342, 
                 0.3034, 0.2735, 0.2405, 0.2283, 0.2071, 0.1669, 0.1530, 
                 0.1339, 0.1265, 0.1200, 0.0990, 0.0870, 0.0770, 0.0690],
                [0.0000, 0.2000, 0.2886, 0.3010, 0.3215, 0.3123, 0.2716, 
                 0.2551, 0.2258, 0.1959, 0.1789, 0.1457, 0.1198, 0.0909, 
                 0.0719, 0.0561, 0.0460, 0.0280, 0.0190, 0.0140, 0.0100]
            ])
        }

    def set_time(self):
        self.t = np.arange(21)

    def _ode(self, theta):

        def f(t, x):
            x1, x2 = x
            dx1 = -(theta[0] + theta[2]) * x1 ** 2
            dx2 = theta[0] * x1 ** 2 - theta[1] * x2
            return [dx1, dx2]

        sol = solve_ivp(f, [self.t[0], self.t[-1]], [1, 1], t_eval=self.t)
        return {'x': sol.y, 'success': sol.success}

    def _obj(self, x):
        return ((x - self.parameters['z']) ** 2).sum()

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