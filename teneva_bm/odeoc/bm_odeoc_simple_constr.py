import numpy as np
from teneva_bm.odeoc import BmOdeocSimple


class BmOdeocSimpleConstr(BmOdeocSimple):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Discrete optimal control (OC) problem with constraint of the special
            form. This benchmark is the same as "BmOdeocSimple", except the
            constraint. Please see the description of BmOdeocSimple for more
            details. The dimension may be any (default is 100), and the mode
            size should be 2. The benchmark needs "gekko==1.0.6" library (it is
            used for ODE solution). Note that the default penalty for the
            constraint is "1.E+42" and amplitude of the constraint doesn't used.
        """)

        self.set_constr(penalty=1.E+42, with_amplitude=False)

    @property
    def with_constr(self):
        return True

    def constr(self, i):
        s = ''.join([str(k) for k in i])

        condition_false = False
        condition_false = condition_false or s.startswith('10')
        condition_false = condition_false or s.startswith('110')
        condition_false = condition_false or s.startswith('01')
        condition_false = condition_false or s.startswith('011')
        condition_false = condition_false or s.startswith('010')
        condition_false = condition_false or s.startswith('0110')

        return 1. if condition_false else -1.
