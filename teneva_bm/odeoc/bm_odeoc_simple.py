import numpy as np


try:
    from gekko import GEKKO
    with_gekko = True
except Exception as e:
    with_gekko = False


from teneva_bm import Bm


DESC = """
    Discrete optimal control (OC) problem with simple 1D ODE "x**3 - i",
    where "x = x(t)" is a state variable, "x(0) = x_ini" and "i" is a
    binary control variable. The loss function for the optimal control
    problem is "0.5 * (x-x_ref)^2", where "x_ref" is a target value, and
    the maximum time value is "t_max". Note that for some control values
    the solver (gekko) fails, in this case we return the value "y_err".
    By default (see parameters of the "set_opts" function), "x_ini = 0.8",
    "x_ref = 0.7", "t_max = 1" and "y_err = 1.E+50". The dimension may be
    any (default is 100), and the mode size should be 2. The benchmark
    needs "gekko==1.0.6" library (it is used for ODE solution).
"""


class BmOdeocSimple(Bm):
    def __init__(self, d=100, n=2, name='OdeocSimple', desc=DESC):
        super().__init__(d, n, name, desc)

        if not self.is_n_equal or self.n[0] != 2:
            self.set_err('Mode size (n) should be "2"')

        if not with_gekko:
            msg = 'Need "gekko" module. For installation please run '
            msg += '"pip install gekko==1.0.6"'
            self.set_err(msg)

    @property
    def identity(self):
        return ['d']

    @property
    def is_tens(self):
        return True

    def get_config(self):
        conf = super().get_config()
        conf['_x_ini'] = self._x_ini
        conf['_x_ref'] = self._x_ref
        conf['_t_max'] = self._t_max
        conf['_y_err'] = self._y_err
        return conf

    def info(self, footer=''):
        text = ''

        text += 'Param x_ini (initial condition)          : '
        v = self._x_ini
        text += f'{v:.6f}\n'

        text += 'Param x_ref (target value)               : '
        v = self._x_ref
        text += f'{v:.6f}\n'

        text += 'Param t_max (upper limit for time)       : '
        v = self._t_max
        text += f'{v:.6f}\n'

        text += 'Param y_err (returned value if error)    : '
        v = self._y_err
        text += f'{v:-7.1e}\n'

        return super().info(text+footer)

    def set_opts(self, x_ini=0.8, x_ref=0.7, t_max=1., y_err=1.E+50):
        """Set options specific to the benchmark.

        There are no plans to manually change the default values.

        Args:
            x_ini (float): initial condition for the ODE.
            x_ref (float): target (reference) value for solution of the ODE.
            t_max (float): upper limit for time variable in the ODE.
            y_err (float): returned value if GEKKO solver ends with error.

        """
        self._x_ini = x_ini
        self._x_ref = x_ref
        self._t_max = t_max
        self._y_err = y_err

        self._times = np.linspace(0, t_max, self.d)

    def target(self, i):
        solver = GEKKO(remote=False)
        solver.options.IMODE = 4
        solver.time = self._times

        x = solver.Var(value=self._x_ini, name='x')
        c = solver.Param(list(i), name='i')
        y = solver.Var(value=0.)

        solver.Equation(x.dt() == self._ode(x, c))
        solver.Equation(y == self._obj(x, c))

        try:
            solver.solve(disp=False)
            y = sum(y.VALUE)
        except Exception as e:
            y = self._y_err

        return y

    def _ode(self, x, i):
        """Target ordinary differential equation (ODE)."""
        return x**3 - i

    def _obj(self, x, i):
        """Objective function for ODE solution."""
        return 0.5 * (x - self._x_ref)**2


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmOdeocSimple().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+2)
    print(bm.info_history())

    text = 'Value at a random multi-index     :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at 3 random multi-indices   :  '
    i1 = [np.random.choice(k) for k in bm.n]
    i2 = [np.random.choice(k) for k in bm.n]
    i3 = [np.random.choice(k) for k in bm.n]
    I = [i1, i2, i3]
    y = bm[I]
    text += '; '.join([f'{y_cur:-10.3e}' for y_cur in y])
    print(text)
