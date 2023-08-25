"""Tests for teneva_bm.

Run it as "clear && python test_all.py" from the root folder of the project.

"""
import numpy as np
import os
import teneva
from teneva_bm import *
import unittest


class TestArgs(unittest.TestCase):
    def test_base(self):
        d = 12
        n = 42
        seed = 55

        bm = BmFuncAckley(d, n, seed)
        bm.prep()

        self.assertTrue(bm.is_n_equal)

        self.assertTrue('d' in bm.args)
        self.assertTrue('n' in bm.args)
        self.assertTrue('seed' in bm.args)

        self.assertEqual(bm.args['d'], d)
        self.assertEqual(bm.args['n'], n)
        self.assertEqual(bm.args['seed'], seed)

    def test_list(self):
        d = 5
        n = [9, 8, 7, 6, 5]

        bm = BmFuncAckley(d, n)
        bm.prep()

        self.assertFalse(bm.is_n_equal)

        self.assertEqual(len(bm.args['n']), 5)
        self.assertEqual(bm.args['n'][0], n[0])
        self.assertEqual(bm.args['n'][-1], n[-1])

    def test_list_equal(self):
        d = 12
        n = [42] * d

        bm = BmFuncAckley(d, n)
        bm.prep()

        self.assertTrue(bm.is_n_equal)

        self.assertEqual(bm.args['n'], n[0])


class TestBmBudgetOverException(unittest.TestCase):
    def setUp(self):
        self.m = 100

    def test_base(self):
        bm = BmFuncAckley()
        bm.set_opts(budget_raise=True)
        bm.set_budget(m=self.m)
        bm.prep()

        for k in range(self.m):
            bm.get([0]*bm.d)

        with self.assertRaises(BmBudgetOverException) as cm:
            bm.get([0]*bm.d)

        text = f'Computation budget (m={self.m}) exceeded'
        self.assertEqual(str(cm.exception), text)
        self.assertEqual(bm.m, self.m)

    def test_cache(self):
        bm = BmFuncAckley()
        bm.set_opts(budget_raise=True)
        bm.set_budget(m=self.m, m_cache=self.m)
        bm.set_cache(True)
        bm.prep()

        for k in range(self.m+1):
            bm.get([0]*bm.d)

        with self.assertRaises(BmBudgetOverException) as cm:
            bm.get([0]*bm.d)

        text = f'Computation budget for cache (m={self.m}) exceeded'
        self.assertEqual(str(cm.exception), text)
        self.assertEqual(bm.m, 1)
        self.assertEqual(bm.m_cache, self.m)


class TestBuildTrn(unittest.TestCase):
    def setUp(self):
        self.m = 5
        self.eps = 1.E-11

    def test_base(self):
        bm = BmFuncAckley()
        bm.prep()

        I_trn, y_trn = bm.build_trn(self.m)

        self.assertEqual(len(I_trn.shape), 2)
        self.assertEqual(I_trn.shape[0], self.m)
        self.assertEqual(I_trn.shape[1], bm.d)
        self.assertEqual(len(y_trn.shape), 1)
        self.assertEqual(y_trn.shape[0], self.m)

        self.assertEqual(bm.m, self.m)
        self.assertEqual(bm.m_cache, 0)

        e = np.abs(bm.get(I_trn[0]) - y_trn[0])

        self.assertLess(e, self.eps)

    def test_without_budget(self):
        bm = BmFuncAckley()
        bm.prep()

        I_trn, y_trn = bm.build_trn(self.m, skip_process=True)

        self.assertEqual(bm.m, 0)
        self.assertEqual(bm.m_cache, 0)


class TestBuildTst(unittest.TestCase):
    def setUp(self):
        self.m = 5
        self.eps = 1.E-11

    def test_base(self):
        bm = BmFuncAckley()
        bm.prep()

        I_tst, y_tst = bm.build_tst(self.m)

        self.assertEqual(len(I_tst.shape), 2)
        self.assertEqual(I_tst.shape[0], self.m)
        self.assertEqual(I_tst.shape[1], bm.d)
        self.assertEqual(len(y_tst.shape), 1)
        self.assertEqual(y_tst.shape[0], self.m)

        self.assertEqual(bm.m, 0)
        self.assertEqual(bm.m_cache, 0)

        e = np.abs(bm.get(I_tst[0]) - y_tst[0])

        self.assertLess(e, self.eps)

    def test_with_budget(self):
        bm = BmFuncAckley()
        bm.prep()

        I_tst, y_tst = bm.build_tst(self.m, skip_process=False)

        self.assertEqual(bm.m, self.m)
        self.assertEqual(bm.m_cache, 0)


class TestCores(unittest.TestCase):
    def setUp(self):
        self.eps = 1.E-15

    def test_base(self):
        for Bm in teneva_bm_get(with_cores=True):
            bm = Bm()
            bm.prep()

            Y = bm.build_cores()

            self.assertEqual(len(Y), bm.d)
            self.assertEqual(len(Y[0].shape), 3)
            self.assertEqual(Y[0].shape[0], 1)
            self.assertEqual(Y[0].shape[1], bm.n[0])
            self.assertEqual(Y[-1].shape[2], 1)

            I_trn, y_trn = bm.build_trn(1.E+2)
            e = teneva.accuracy_on_data(Y, I_trn, y_trn)

            msg = f'\n>>> Test failed for "{Bm.__name__}" [e = {e:-7.1e}]'
            self.assertLess(e, self.eps, msg)


class TestGet(unittest.TestCase):
    def test_base(self):
        for Bm in teneva_bm_get():
            with self.subTest(Bm=Bm):
                bm = Bm()
                bm.prep()

                i = [np.random.choice(k) for k in bm.n]
                y = bm[i]

                msg = f'\n>>> Test failed for "{Bm.__name__}"'
                self.assertTrue(isinstance(y, float), msg)

    def test_base_many(self):
        for Bm in teneva_bm_get():
            with self.subTest(Bm=Bm):
                bm = Bm()
                bm.prep()

                i1 = [np.random.choice(k) for k in bm.n]
                i2 = [np.random.choice(k) for k in bm.n]
                i3 = [np.random.choice(k) for k in bm.n]
                I = [i1, i2, i3]
                y = bm[I]

                msg = f'\n>>> Test failed for "{Bm.__name__}"'

                self.assertTrue(isinstance(y, np.ndarray), msg)
                self.assertEqual(y.shape, (3,), msg)
                self.assertTrue(isinstance(y[0], float), msg)
                self.assertTrue(isinstance(y[1], float), msg)
                self.assertTrue(isinstance(y[2], float), msg)


class TestGetPoi(unittest.TestCase):
    def test_base(self):
        for Bm in teneva_bm_get(is_func=True):
            with self.subTest(Bm=Bm):
                bm = Bm()
                bm.prep()

                self.assertFalse(bm.a is None)
                self.assertFalse(bm.b is None)

                x = bm.a + (bm.b - bm.a) * 0.5
                y = bm(x)

                msg = f'\n>>> Test failed for "{Bm.__name__}"'
                self.assertTrue(isinstance(y, float), msg)

    def test_base_many(self):
        for Bm in teneva_bm_get(is_func=True):
            with self.subTest(Bm=Bm):
                bm = Bm()
                bm.prep()

                self.assertFalse(bm.a is None)
                self.assertFalse(bm.b is None)

                x1 = bm.a + (bm.b - bm.a) * 0.3
                x2 = bm.a + (bm.b - bm.a) * 0.5
                x3 = bm.a + (bm.b - bm.a) * 0.7
                X = [x1, x2, x3]
                y = bm(X)

                msg = f'\n>>> Test failed for "{Bm.__name__}"'

                self.assertTrue(isinstance(y, np.ndarray), msg)
                self.assertEqual(y.shape, (3,), msg)
                self.assertTrue(isinstance(y[0], float), msg)
                self.assertTrue(isinstance(y[1], float), msg)
                self.assertTrue(isinstance(y[2], float), msg)


class TestHist(unittest.TestCase):
    def test_base(self):
        bm = BmFuncAckley(d=2, n=3)
        bm.set_cache(True)
        bm.prep()

        I = [
            [0, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 0],
        ]
        y = [bm[i] for i in I]

        self.assertEqual(bm.hist['m'], len(I)-2)
        self.assertEqual(bm.hist['m_cache'], 2)
        self.assertEqual(bm.hist['y_max'], np.max(y))
        self.assertEqual(bm.hist['y_min'], np.min(y))
        self.assertEqual(len(bm.hist['y_list']), len(I)-2)
        self.assertEqual(len(bm.hist['y_list_full']), len(I))

        self.assertEqual(bm.hist['y_list'][0], y[0])
        self.assertEqual(bm.hist['y_list'][1], y[1])
        self.assertEqual(bm.hist['y_list'][2], y[4])


class TestInfo(unittest.TestCase):
    def test_base(self):
        for Bm in teneva_bm_get():
            with self.subTest(Bm=Bm):
                bm = Bm().prep()
                text = bm.info()
                self.assertTrue('Benchmark class name' in text)


class TestInfoHistory(unittest.TestCase):
    def test_base(self):
        for Bm in teneva_bm_get():
            with self.subTest(Bm=Bm):
                bm = Bm().prep()
                I = [[np.random.choice(k) for k in bm.n] for _ in range(3)]
                y = bm[I]
                text = bm.info_history()
                msg = 'Number of requests                       :  3.000e+00'
                self.assertTrue(msg in text)


class TestMax(unittest.TestCase):
    def setUp(self):
        self.eps = 1.E-14

    def test_base(self):
        for Bm in teneva_bm_get(with_max=True):
            with self.subTest(Bm=Bm):
                bm = Bm()
                bm.prep()

                eps = self.eps
                if Bm.__name__ == 'BmFuncSchwefel':
                    eps = 2.E-5

                if bm.i_max_real is not None:
                    y = bm[bm.i_max_real]
                    e = np.abs(y - bm.y_max_real)
                    msg = f'\n>>> Test failed for "{Bm.__name__}"'
                    msg += f' [e = {e:-7.1e}]'
                    self.assertLess(e, seps, msg)

                if bm.x_max_real is not None:
                    y = bm(bm.x_max_real)
                    e = np.abs(y - bm.y_max_real)
                    msg = f'\n>>> Test failed for "{Bm.__name__}"'
                    msg += f' [e = {e:-7.1e}]'
                    self.assertLess(e, eps, msg)


class TestMin(unittest.TestCase):
    def setUp(self):
        self.eps = 1.E-14

    def test_base(self):
        for Bm in teneva_bm_get(with_min=True):
            with self.subTest(Bm=Bm):
                bm = Bm()
                bm.prep()

                eps = self.eps
                if Bm.__name__ == 'BmFuncSchwefel':
                    eps = 2.E-5

                if bm.i_min_real is not None:
                    y = bm[bm.i_min_real]
                    e = np.abs(y - bm.y_min_real)
                    msg = f'\n>>> Test failed for "{Bm.__name__}"'
                    msg += f' [e = {e:-7.1e}]'
                    self.assertLess(e, eps, msg)

                if bm.x_min_real is not None:
                    y = bm(bm.x_min_real)
                    e = np.abs(y - bm.y_min_real)
                    msg = f'\n>>> Test failed for "{Bm.__name__}"'
                    msg += f' [e = {e:-7.1e}]'
                    self.assertLess(e, eps, msg)


class TestOpts(unittest.TestCase):
    def test_base(self):
        bm = BmFuncAckley()
        bm.prep()

        self.assertTrue('opt_A' in bm.opts)
        self.assertEqual(bm.opts['opt_A'], bm.opts_info['opt_A']['dflt'])

    def test_custom(self):
        opt_A = 42.

        bm = BmFuncAckley()
        bm.set_opts(opt_A=opt_A)
        bm.prep()

        self.assertTrue('opt_A' in bm.opts)
        self.assertEqual(bm.opts['opt_A'], opt_A)


class TestProcess(unittest.TestCase):
    def test_base(self):
        bm = BmFuncAckley(d=2, n=3)
        bm.prep()

        I = [
            [0, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 0],
        ]
        y = bm[I]

        bm.build_tst(10)

        self.assertEqual(bm.m, len(I))
        self.assertEqual(bm.y_max, np.max(y))
        self.assertEqual(bm.y_min, np.min(y))
        self.assertEqual(len(bm.y_list), bm.m)
        self.assertEqual(len(bm.y_list_full), bm.m)

        for k in range(len(I)):
            self.assertEqual(bm.y_list[k], y[k])

        for k in range(len(I)):
            self.assertEqual(bm.y_list_full[k], y[k])

    def test_cache(self):
        bm = BmFuncAckley(d=2, n=3)
        bm.set_cache(True)
        bm.prep()

        I = [
            [0, 1],
            [1, 1],
            [0, 1], # Cache
            [1, 1], # Cache
            [0, 0],
        ]
        y = [bm[i] for i in I]

        bm.build_tst(10)

        self.assertEqual(bm.m, len(I)-2)
        self.assertEqual(bm.m_cache, 2)
        self.assertEqual(bm.y_max, np.max(y))
        self.assertEqual(bm.y_min, np.min(y))
        self.assertEqual(len(bm.y_list), len(I)-2)
        self.assertEqual(len(bm.y_list_full), len(I))

        for k in range(len(I)):
            self.assertEqual(bm.y_list_full[k], y[k])

        self.assertEqual(bm.y_list[0], y[0])
        self.assertEqual(bm.y_list[1], y[1])
        self.assertEqual(bm.y_list[2], y[4])


class TestPrps(unittest.TestCase):
    def test_base(self):
        bm = BmFuncAckley()
        bm.prep()

        self.assertEqual(bm.prps['name_class'], 'BmFuncAckley')
        self.assertEqual(bm.prps['is_func'], True)
        self.assertEqual(bm.prps['is_tens'], False)

    def test_spec(self):
        bm = BmFuncAckley()
        bm.set_grid(-1., +1.)
        bm.set_constr(penalty=42)
        bm.prep()

        self.assertTrue(bm.is_a_equal)
        self.assertTrue(bm.is_b_equal)

        self.assertEqual(bm.prps['a'], -1.)
        self.assertEqual(bm.prps['b'], +1.)
        self.assertEqual(bm.prps['constr_penalty'], 42)


class TestRender(unittest.TestCase):
    def test_base(self):
        for Bm in teneva_bm_get(with_render=True):
            with self.subTest(Bm=Bm):
                bm = Bm()
                bm.prep()

                i = [np.random.choice(k) for k in bm.n]
                y = bm[i]

                fpath = f'result/test/render/{Bm.__name__}.mp4'
                bm.render(fpath)

                is_ok = os.path.isfile(fpath)
                is_ok = is_ok and os.path.getsize(fpath) > 1000

                msg = f'\n>>> Test failed for "{Bm.__name__}"'
                self.assertTrue(is_ok, msg)


class TestShow(unittest.TestCase):
    def test_base(self):
        for Bm in teneva_bm_get(with_show=True):
            with self.subTest(Bm=Bm):
                bm = Bm()
                bm.prep()

                i = [np.random.choice(k) for k in bm.n]
                y = bm[i]

                fpath = f'result/test/show/{Bm.__name__}.png'
                bm.show(fpath)

                is_ok = os.path.isfile(fpath)
                is_ok = is_ok and os.path.getsize(fpath) > 1000

                msg = f'\n>>> Test failed for "{Bm.__name__}"'
                self.assertTrue(is_ok, msg)


if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()
