import numpy as np
import teneva


from teneva_bm import BmOcSimple


DESC = """
    Discrete optimal control (OC) problem with simple 1D ODE "x**3 - i" and
    constraint of the special form, where "i" is a binary control variable.
    The dimension may be any (default is 50), and the mode size should be 2.
    The benchmark needs "gekko==1.0.6" library (it is used for ODE solution).
    TODO: add options for constraint.
"""


class BmOcSimpleConstr(BmOcSimple):
    def __init__(self, d=50, n=2, name='OcSimpleConstr', desc=DESC):
        super().__init__(d, n, name, desc)

    def bm_constr(self, i):
        """Constraint."""
        v = 1.E+42
        s = ''.join([str(k) for k in i])
        if s.startswith('10'): return v
        if s.startswith('110'): return v
        if s.endswith('01'): return v
        if s.endswith('011'): return v
        if '010' in s: return v
        if '0110' in s: return v


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmOcSimpleConstr().prep()
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
