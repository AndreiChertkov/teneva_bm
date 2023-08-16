"""Special test for teneva_bm (check reference values for all benchmarks).

Test that for a reference multi-index benchmark returns the expected value.
Run it as "clear && python test_ref.py" from the root folder of the project.

"""
import numpy as np
from teneva_bm import *
import unittest


class TestRef(unittest.TestCase):
    def setUp(self):
        self.eps = 1.E-16

    def test_base(self):
        for Bm in teneva_bm_get():
            with self.subTest(Bm=Bm):
                bm = Bm()
                bm.prep()

                i, z = bm.ref
                y = bm.get(i)

                e = np.abs(y - z)

                msg = '\n>>> Ref. value does not match for '
                msg += f'"{Bm.__name__}" [d={bm.d}; y={y}]'
                self.assertLess(e, self.eps, msg)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()
