import numpy as np
from teneva_bm import Bm


try:
    import qubogen
    with_qubogen = True
except Exception as e:
    with_qubogen = False


class BmQuboKnapQuad(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Quadratic unconstrained binary optimization (QUBO) knapsack problem
            represented as a discrete function. The dimension may be any
            (default is 100), and the mode size should be 2. It needs
            "qubogen==0.1.1" library.
        """)

        if not with_qubogen:
            msg = 'Need "qubogen" module. For installation please run '
            msg += '"pip install qubogen==0.1.1"'
            self.set_err(msg)

    @property
    def args_constr(self):
        return {'n': 2}

    @property
    def identity(self):
        return ['d', 'seed']

    @property
    def is_tens(self):
        return True

    @property
    def ref(self):
        i = np.zeros(100, dtype=int)
        for k in [0, 12, 34, 44, 53, 65, 99]:
            i[k] = 1
        return np.array(i, dtype=int), 86.65042431410035

    def prep_bm(self):
        v = np.diag(self.rand.random(self.d)) / 3.
        a = self.rand.random(self.d)
        b = np.mean(a)
        self._Q = qubogen.qubo_qkp(v, a, b)

    def target_batch(self, I):
        return ((I @ self._Q) * I).sum(axis=1)
