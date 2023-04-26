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
            self.set_err('Need "qubogen" module')

    @property
    def is_tens(self):
        return True

    def prep(self):
        v = np.diag(np.random.random(self.d)) / 3.
        a = np.random.random(self.d)
        b = np.mean(a)
        self.bm_Q = qubogen.qubo_qkp(v, a, b)

        self._is_prep = True
        return self

    def _f_batch(self, I):
        return ((I @ self.bm_Q) * I).sum(axis=1)


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmQuboKnapQuad().prep()
    print(bm.info())

    text = 'Range of y for 10K random samples : '
    bm.build_trn(1.E+4)
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
