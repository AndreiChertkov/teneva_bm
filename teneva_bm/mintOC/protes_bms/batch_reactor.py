import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmBatchReactor(Bm):
    def __init__(self, d=100, n=101, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Batch Reactor problem from the mintOC collection describes the consecutive reaction 
            of some substance A via substance B into a desired product C. The system is interacted 
            with via the control variable T which stands for the temperature. 
            The goal is to produce as much of substance B as possible within the time limit.
            Details: https://mintoc.de/index.php/Batch_reactor
        """)

        self.set_grid(298, 398)
        self.set_grid_kind('uni')
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()
    
    @property
    def is_func(self):
        return True
    
    def set_parameters(self):
        self.time = np.linspace(0, 1, self.d)
        self.state_initial = [1, 0]

    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2 = state
            T = control_interpolate(t)
            k1 = 4e3 * np.exp(-2.5e3 / T)
            k2 = 6.2e5 * np.exp(-5e3 / T)
            dx1 = -k1 * x1 ** 2
            dx2 = k1 * x1 ** 2 - k2 * x2
            return [dx1, dx2]

        sol = solve_ivp(f, [self.time[0], self.time[-1]], y0=self.state_initial, t_eval=self.time)
        return {'state': sol.y, 'success': sol.success}

    def _obj(self, state):
        x1, x2 = state
        return -x2[-1]

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