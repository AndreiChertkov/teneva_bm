import numpy as np
import teneva


from teneva_bm.oc import BmOcSimple


DESC = """
    Discrete optimal control (OC) problem with constraint of the special
    form. This benchmark is the same as "BmOcSimple", except the constraint.
    Please see the description of BmOcSimple for more details. Note that in
    the case if constraint fails, we return the penalty value "1.E+42".
    The dimension may be any (default is 50), and the mode size should be 2.
    The benchmark needs "gekko==1.0.6" library (it is used for ODE solution).
"""


class BmOcSimpleConstr(BmOcSimple):
    def __init__(self, d=50, n=2, name='OcSimpleConstr', desc=DESC):
        super().__init__(d, n, name, desc)

        self.penalty_constr = 1.E+42
        self.with_constr = True

    def _constr(self, i):
        s = ''.join([str(k) for k in i])

        condition_false = False
        condition_false = condition_false or s.startswith('10')
        condition_false = condition_false or s.startswith('110')
        condition_false = condition_false or s.startswith('01')
        condition_false = condition_false or s.startswith('011')
        condition_false = condition_false or s.startswith('010')
        condition_false = condition_false or s.startswith('0110')

        return not condition_false


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmOcSimpleConstr().prep()
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

    text = 'Value at a valid multi-index      :  '
    i = [1] * bm.d
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at an invalid multi-index   :  '
    i = [1, 0] + [0] * (bm.d-2)
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)
