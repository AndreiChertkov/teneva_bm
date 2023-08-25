import numpy as np
from teneva_bm import Bm


try:
    from gekko import GEKKO
    with_gekko = True
except Exception as e:
    with_gekko = False


class BmOdeocSimple(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Discrete optimal control (OC) problem with simple 1D ODE "x**3 - i",
            where "x = x(t)" is a state variable, "x(0) = x_ini" and "i" is a
            binary control variable. The loss function for the optimal control
            problem is "0.5 * (x-x_ref)^2", where "x_ref" is a target value, and
            the maximum time value is "t_max". Note that for some control values
            the solver (gekko) fails, in this case we return the value "y_err".
            By default (see parameters of the "set_opts" function), "x_ini =
            0.8", "x_ref = 0.7", "t_max = 1" and "y_err = 1.E+50". The
            dimension may be any (default is 100), and mode size should be 2.
            The benchmark needs "gekko==1.0.6" (it is used for ODE solution).
        """)

        if not with_gekko:
            msg = 'Need "gekko" module. For installation please run '
            msg += '"pip install gekko==1.0.6"'
            self.set_err(msg)

    @property
    def args_constr(self):
        return {'n': 2}

    @property
    def identity(self):
        return ['d']

    @property
    def is_tens(self):
        return True

    @property
    def opts_info(self):
        return {**super().opts_info,
            'x_ini': {
                'desc': 'Initial condition for the ODE',
                'kind': 'float',
                'form': '.6f',
                'dflt': 0.8
            },
            'x_ref': {
                'desc': 'Target (reference) value',
                'kind': 'float',
                'form': '.6f',
                'dflt': 0.7
            },
            't_max': {
                'desc': 'Upper limit for time variable',
                'kind': 'float',
                'form': '.6f',
                'dflt': 1.
            },
            'y_err': {
                'desc': 'Returned value if error in ODE solver',
                'kind': 'float',
                'form': '7.1e',
                'dflt': 1.E+50
            }
        }

    @property
    def ref(self):
        i = np.ones(100, dtype=int)
        for k in [0, 1, 2, 44, 53, 65, 33]:
            i[k] = 0
        return np.array(i, dtype=int), 5.184363677330866

    def target(self, i):
        solver = GEKKO(remote=False)
        solver.options.IMODE = 4
        solver.time = np.linspace(0, self.t_max, self.d)

        x = solver.Var(value=self.x_ini, name='x')
        c = solver.Param(list(i), name='i')
        y = solver.Var(value=0.)

        solver.Equation(x.dt() == self._ode(x, c))
        solver.Equation(y == self._obj(x, c))

        try:
            solver.solve(disp=False)
            y = sum(y.VALUE)
        except Exception as e:
            y = self.y_err

        return y

    def _ode(self, x, i):
        """Target ordinary differential equation (ODE)."""
        return x**3 - i

    def _obj(self, x, i):
        """Objective function for ODE solution."""
        return 0.5 * (x - self.x_ref)**2
