import numpy as np
import teneva
from teneva_bm.func.func import Func


class BmFuncSphere(Func):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

        self.set_desc("""
            Analytical Sphere function (continuous).
            The dimension and mode size may be any (default are d=7, n=16).
            Default grid limits are [-10, 10] (with small random shift);
            the exact global minimum is known: x = [0, ..., 0], y = 0.
            See the work Momin Jamil, Xin-She Yang. "A literature survey of
            benchmark functions for global optimization problems". Journal of
            Mathematical Modelling and Numerical Optimisation 2013; 4:150-194
            ("136. Sphere Function"; Continuous, Differentiable,
            Separable, Scalable, Multimodal).
        """)

        self.set_grid(-10., +10., sh=True)

        self.set_min(x=[0.]*self.d, y=0.)

    @property
    def ref(self):
        i = [5, 3, 9, 11, 14, 3, 10]
        return np.array(i, dtype=int), 325.3751972441734

    def target_batch(self, X):
        return np.sum(X**2, axis=1)


if __name__ == '__main__':
    # Service code just for test.
    bm = BmFuncSphere().prep()
    print(bm[bm.ref[0]])
    print(bm(bm.x_min_real))
    print(bm.y_min_real)
