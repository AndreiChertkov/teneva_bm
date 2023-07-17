import numpy as np
import teneva


try:
    import qubogen
    with_qubogen = True
except Exception as e:
    with_qubogen = False


from teneva_bm import Bm


DESC = """
    Quadratic unconstrained binary optimization (QUBO) knapsack problem
    represented as a discrete function.
    The dimension may be any (default is 50), and the mode size should be 2.
    The benchmark needs "qubogen==0.1.1" library.
"""


class BmQuboKnapQuad(Bm):
    def __init__(self, d=50, n=2, name='QuboKnapQuad', desc=DESC):
        super().__init__(d, n, name, desc)

        if not self.is_n_equal or self.n[0] != 2:
            self.set_err('Mode size (n) should be "2"')
        if not with_qubogen:
            msg = 'Need "qubogen" module. For installation please run '
            msg += '"pip install qubogen==0.1.1"'
            self.set_err(msg)

    @property
    def is_tens(self):
        return True

    def prep(self):
        self.check_err()

        v = np.diag(np.random.random(self.d)) / 3.
        a = np.random.random(self.d)
        b = np.mean(a)
        self.bm_Q = qubogen.qubo_qkp(v, a, b)

        self.is_prep = True
        return self

    def _f_batch(self, I):
        return ((I @ self.bm_Q) * I).sum(axis=1)


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmQuboKnapQuad().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+4)
    print(bm.info_history())

    text = 'Value at a random multi-index            :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at 3 random multi-indices          :  '
    i1 = [np.random.choice(k) for k in bm.n]
    i2 = [np.random.choice(k) for k in bm.n]
    i3 = [np.random.choice(k) for k in bm.n]
    I = [i1, i2, i3]
    y = bm[I]
    text += '; '.join([f'{y_cur:-10.3e}' for y_cur in y])
    print(text)
