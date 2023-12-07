import numpy as np
from teneva_bm import Bm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class BmElectricCar(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The Electric Car problem from the mintOC collection tries to find 
            an optimal driving policy for an electric car. 
            The goal is to use minimal energy to finish a given distance.
            Details: https://mintoc.de/index.php/Egerstedt_standard_problem
        """)

        self.set_grid(-1, 1)
        self.set_grid_kind('uni')
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'n': 2}
    
    @property
    def is_func(self):
        return True
    
    def set_parameters(self):
        self.time = np.linspace(0, 10, self.d)
        self.state_initial = [0, 0, 0, 0]
        self.parameters = {
            'K_r': 10,     # Coefficient of reduction
            'rho': 1.293,  # Air density
            'C_x': 0.4,    # Aerodynamic coefficient
            'S': 2,        # Area in the front of the vehicle
            'r': 0.33,     # Radius of the wheel
            'K_f': 0.03,   # Constant representing the friction of the wheels on the road
            'K_m': 0.27,   # Coefficient of the motor torque
            'R_m': 0.03,   # Inductor resistance
            'L_m': 0.05,   # Inductance of the rotor
            'M': 250,      # Mass
            'g': 9.81,     # Gravity constant
            'V_alim': 150, # Battery voltage
            'R_bat': 0.05, # Resistance of the battery
            'i_max': 150   # Max. value of x1
        }

    def _ode(self, control):
        control_interpolate = interp1d(self.time, control, kind='nearest', fill_value='extrapolate')

        def f(t, state):
            x1, x2, x3, x4 = state
            u = control_interpolate(t)
            K_r, rho, C_x, S, r, K_f, K_m, R_m, L_m, M, g, V_alim, R_bat, i_max = self.parameters.values()
            k_1 = r / K_r
            k_2 = M * g * K_f
            k_3 = 0.5 * rho * S * C_x
            dx1 = (V_alim * u - R_m * x1 - K_m * x2) / L_m
            dx2 = (K_m * x1 - k_1 * (k_2 + k_3 * (k_1 * x2) ** 2)) * K_r ** 2 / (M * r ** 2)
            dx3 = k_1 * x2
            dx4 = V_alim * u * x1 + R_bat * x1 ** 2
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
    
    def _constr(self, control):
        sol = self._ode(control)
        if sol['success']:
            x1, x2, x3, x4 = sol['state']
            c1 = np.abs(100 - x3[-1])
            c2 = -1 * (self.parameters['i_max'] + x1)
            c3 = -1 * (self.parameters['i_max'] - x1)
            return np.hstack([0, c1, c2, c3])
        else:
            return np.array([self.constr_penalty] + [0] * 3)
        
    def constr(self, control):
        c = self._constr(control)
        c = sum(~(c < self.constr_eps) * np.abs(c))
        return c