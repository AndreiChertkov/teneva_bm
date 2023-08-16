import numpy as np
from teneva_bm import Bm


class Func(Bm):
    def __init__(self, d=7, n=16, seed=42):
        super().__init__(d, n, seed)

    @property
    def is_func(self):
        return True
