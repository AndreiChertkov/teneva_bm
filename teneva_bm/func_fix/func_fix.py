import numpy as np
from teneva_bm import Bm


class FuncFix(Bm):
    def __init__(self, d=None, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

    @property
    def is_func(self):
        return True
