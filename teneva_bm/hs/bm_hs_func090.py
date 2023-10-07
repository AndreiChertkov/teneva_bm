import numpy as np
from teneva_bm import Bm


class BmHsFunc090(Bm):
    def __init__(self, d=4, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 090 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .-------------------------------------.
            | F(x | p, t) -> min s.t. C(x) = True |
            .-------------------------------------.
            p - parameters
                mu12 = [0.8603335890194, 3.4256184594817, 6.4372981791719, 9.5293344053620, 
                        12.6452872238566, 15.7712848748159, 18.9024099568600, 22.0364967279386, 
                        25.1724463266467, 28.3096428544520, 31.4477146375462, 34.5864242152889, 
                        37.7256128277765, 40.8651703304881, 44.0050179208308, 47.1450977367610, 
                        50.2853663377737, 53.4257904773947, 56.5663442798215, 59.7070073053355, 
                        62.8477631944545, 65.9885986984904, 69.1295029738953, 72.2704670603090, 
                        75.4114834888482, 78.5525459842429, 81.6936492356017, 84.8347887180423, 
                        87.9759605524932, 91.1171613944647]
            t - intermediates
                tg12 = tan(mu12)
                mu24 = (1 + mu12 ** 2 * (1 + tg12 ** 2)) / (tg12 + mu12 * (1 + tg12 ** 2))
                tg24 = tan(mu24)
                mu48 = (1 + mu24 ** 2 * (1 + tg24 ** 2)) / (tg24 + mu24 * (1 + tg24 ** 2))
                mu = mu48
                isign = [1, -1, 1, -1, ..., -1] # len = 30
                snmu = isign * sqrt(1 / (1 + mu ** 2))
                csmu = isign * sqrt(mu ** 2 / (1 + mu ** 2))
                snmuxcsmu = mu / (1 + mu ** 2)
                A = 2 * snmu / (mu + snmuxcsmu)
                emx[0] = exp(-mu ** 2 * x[0] ** 2)
                emx[1] = exp(-mu ** 2 * x[1] ** 2)
                emx[2] = exp(-mu ** 2 * x[2] ** 2)
                emx[3] = exp(-mu ** 2 * x[3] ** 2)
                r = emx[0] * emx[1] * emx[2] * emx[3] - \
                    2 * emx[1] * emx[2] * emx[3] + \
                    2 * emx[2] * emx[3] - \
                    2 * emx[3] + \
                    1
                aux1 = A ** 2 * r ** 2
                aux2 = snmuxcsmu / (2 * mu) + 1 / 2
                aux3 = -1 * A * r / mu ** 2
                aux4 = (-2) * snmu / mu + 2 * csmu
                h = sum(aux1 * aux2 + aux3 * aux4)
            x - continuous control
                x[0]
                x[1]
                x[2]
                x[3]
            F - objective function
                x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
            C - constraint function
                0.0001 - h + 2 / 15 >= 0
            The exact global minimum is approx. known:
                y ~= 0.717
                x[0] ~= 1.074
                x[1] ~= 0.0
                x[2] ~= 0.8
                x[3] ~= -0.457
            Hyperparameters: 
                * The dimension d should be 4
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10, -10, -10], [+10, +10, +10, +10])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 4}

    @property
    def identity(self):
        return ['n']

    @property
    def is_func(self):
        return True

    @property
    def with_constr(self):
        return True

    def set_parameters(self):
        mu12 = np.array([
            0.8603335890194, 3.4256184594817, 6.4372981791719, 9.5293344053620, 
            12.6452872238566, 15.7712848748159, 18.9024099568600, 22.0364967279386, 
            25.1724463266467, 28.3096428544520, 31.4477146375462, 34.5864242152889, 
            37.7256128277765, 40.8651703304881, 44.0050179208308, 47.1450977367610, 
            50.2853663377737, 53.4257904773947, 56.5663442798215, 59.7070073053355, 
            62.8477631944545, 65.9885986984904, 69.1295029738953, 72.2704670603090, 
            75.4114834888482, 78.5525459842429, 81.6936492356017, 84.8347887180423, 
            87.9759605524932, 91.1171613944647
        ])
        tg12 = np.tan(mu12)
        mu24 = (1 + mu12 ** 2 * (1 + tg12 ** 2)) / (tg12 + mu12 * (1 + tg12 ** 2))
        tg24 = np.tan(mu24)
        mu48 = (1 + mu24 ** 2 * (1 + tg24 ** 2)) / (tg24 + mu24 * (1 + tg24 ** 2))
        self.parameters = {'mu': mu48}

    def constr_batch(self, X):
        mu = self.parameters['mu']
        mu_2 = mu ** 2
        mu_1_sqrt = np.sqrt(1 + mu_2)
        isign = np.ones(30)
        isign[::2] *= -1
        snmu = isign / mu_1_sqrt
        csmu = isign * mu / mu_1_sqrt
        snmuxcsmu = mu / (1 + mu ** 2)
        A = 2 * snmu / (mu + snmuxcsmu)
        emx_0 = np.exp(-mu_2[None, :] * X[:, 0][:, None] ** 2)
        emx_1 = np.exp(-mu_2[None, :] * X[:, 1][:, None] ** 2)
        emx_2 = np.exp(-mu_2[None, :] * X[:, 2][:, None] ** 2)
        emx_3 = np.exp(-mu_2[None, :] * X[:, 3][:, None] ** 2)
        r = emx_0 * emx_1 * emx_2 * emx_3 - \
            2 * emx_1 * emx_2 * emx_3 + \
            2 * emx_2 * emx_3 - \
            2 * emx_3 + \
            1
        aux1 = A ** 2 * r ** 2
        aux2 = snmuxcsmu / (2 * mu) + 1 / 2
        aux3 = -1 * A * r / mu ** 2
        aux4 = (-2) * snmu / mu + 2 * csmu
        h = (aux1 * aux2 + aux3 * aux4).sum(-1)
        return -1 * (0.0001 - h + 2/15)

    def target_batch(self, X):
        return X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2