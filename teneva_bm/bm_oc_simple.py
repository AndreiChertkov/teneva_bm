import numpy as np
import teneva


try:
    from gekko import GEKKO
    with_gekko = True
except Exception as e:
    with_gekko = False


from teneva_bm import Bm


DESC = """
    Discrete optimal control (OC) problem with simple 1D ODE "x**3 - i", where
    "i" is a binary control variable.
    The dimension may be any (default is 50), and the mode size should be 2.
    The benchmark needs "gekko==1.0.6" library (it is used for ODE solution).
"""


class BmOcSimple(Bm):
    def __init__(self, d=50, n=2, name='OcSimple', desc=DESC):
        super().__init__(d, n, name, desc)

        if not self.is_n_equal or self.n[0] != 2:
            self.set_err('Mode size (n) should be "2"')
        if not with_gekko:
            self.set_err('Need "gekko" module')

    @property
    def is_tens(self):
        return True

    def bm_constr(self, i):
        """Constraint (optional)."""
        return None

    def bm_ode(self, x, i):
        """Target ordinary differential equation (ODE)."""
        return x**3 - i

    def bm_obj(self, x, i):
        """Objective function for ODE solution."""
        return 0.5 * (x - self.opt_x_ref)**2

    def set_opts(self, x_ini=0.8, x_ref=0.7, t_max=1., y_err=1.E+50):
        """Setting options specific to this benchmark.

        Args:
            x_ini (float): initial condition for the ODE.
            x_ref (float): target (reference) value for solution of the ODE.
            t_max (float): upper limit for time variable in the ODE.
            y_err (float): returned value if GEKKO solver ends with error.

        """
        self.opt_x_ini = x_ini
        self.opt_x_ref = x_ref
        self.opt_times = np.linspace(0, t_max, self.d)
        self.opt_y_err = y_err

    def _f(self, i):
        y_constr = self.bm_constr(i)
        if y_constr is not None:
            return y_constr

        solver = GEKKO(remote=False)
        solver.options.IMODE = 4
        solver.time = self.opt_times

        x = solver.Var(value=self.opt_x_ini, name='x')
        c = solver.Param(list(i), name='i')
        y = solver.Var(value=0.)

        solver.Equation(x.dt() == self.bm_ode(x, c))
        solver.Equation(y == self.bm_obj(x, c))

        try:
            solver.solve(disp=False)
            y = sum(y.VALUE)
        except Exception as e:
            y = self.opt_y_err

        return y


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmOcSimple().prep()
    print(bm.info())

    text = 'Range of y for 100 random samples : '
    bm.build_trn(1.E+2)
    text += f'[{np.min(bm.y_trn):-10.3e},'
    text += f' {np.max(bm.y_trn):-10.3e}] '
    text += f'(avg: {np.mean(bm.y_trn):-10.3e})'
    print(text)

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
