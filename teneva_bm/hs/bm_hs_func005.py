import numpy as np
from teneva_bm import Bm


class BmHsFunc005(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 005 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem without constraints:
            .-------------.
            | F(x) -> min |
            .-------------.
            x - continuous control
                x[0] >= -1.5 | <= 4
                x[1] >= -3   | <= 3
            F - objective function
                sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1
            The exact global minimum is known: 
                y = -sqrt(3)/2 - pi/3
                x[0] = -pi/3 + 1/2
                x[1] = -pi/3 - 1/2
            Hyperparameters: 
            * The dimension d should be 2
            * The mode size n may be any (default is 64)
            * The default limits for function inputs are [-10, 10].
        """)
        
        self.set_grid([-1.5, -3], [+4, +3])
        self.set_min(
            x=[-np.pi/3 + 0.5, -np.pi/3 - 0.5], 
            y=(-np.sqrt(3)/2 - np.pi/3)
        )

    @property
    def args_constr(self):
        return {'d': 2}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    def target_batch(self, X):
        return np.sin(X[:, 0] + X[:, 1]) + (X[:, 0] - X[:, 1]) ** 2 - 1.5 * X[:, 0] + 2.5 * X[:, 1] + 1