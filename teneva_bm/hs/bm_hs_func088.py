import numpy as np
from teneva_bm import Bm


class BmHsFunc088(Bm):
    def __init__(self, d=2, n=64, seed=42, name=None):
        super().__init__(d, n, seed, name)
        self.set_desc("""
            The function 088 from the Hock & Schittkowski collection.
            Continuous optimal control (OC) problem with constraints:
            .----------------------------------.
            | F(x | p) -> min s.t. C(x) = True |
            .----------------------------------.
            p - parameters
                mu12 = [0.8603335890194, 3.4256184594817, 6.4372981791719, 9.5293344053620, 12.6452872238566, 15.7712848748159, 18.9024099568600, 22.0364967279386, 25.1724463266467, 28.3096428544520, 31.4477146375462, 34.5864242152889, 37.7256128277765, 40.8651703304881, 44.0050179208308, 47.1450977367610, 50.2853663377737, 53.4257904773947, 56.5663442798215, 59.7070073053355, 62.8477631944545, 65.9885986984904, 69.1295029738953, 72.2704670603090, 75.4114834888482, 78.5525459842429, 81.6936492356017, 84.8347887180423, 87.9759605524932, 91.1171613944647]
            x - continuous control
                x[0]
                x[1]
            F - objective function
                x[0] ** 2 + x[1] ** 2
            C - constraint function
                tg12 = tan(mu12)
                mu24 = (1 + mu12 ** 2 * (1 + tg12 ** 2)) / (tg12 + mu12 * (1 + tg12 ** 2))
                tg24 = tan(mu24)
                mu = (1 + mu24 ** 2 * (1 + tg24 ** 2)) / (tg24 + mu24 * (1 + tg24 ** 2))
                isign = [1, -1, -1, ..., -1] # 30
                snmu = isign * sqrt(1 / (1 + mu ** 2))
                csmu = isign * sqrt(mu ** 2 / (1 + mu ** 2))
                snmuxcsmu = mu / (1 + mu ** 2)
                A = 2 * snmu / (mu + snmuxcsmu)
                r = (exp(-mu ** 2 * x[0] ** 2) - 2) * exp(-mu ** 2 * x[1] ** 2) + 1
                aux = A ** 2 * r ** 2 * (snmuxcsmu / (2 * mu) + 1 / 2) + 2 * A * r / mu ** 2 * (snmu / mu + csmu)
                0.0001 - sum(aux) >= 0
            The exact global minimum is approx. known:
                y ~= 1.363
                x[0] ~= 1.074
                x[1] ~= -0.457
            Hyperparameters: 
                * The dimension d should be 2
                * The mode size n may be any (default is 64)
                * The default limits for function inputs are [-10, 10].
        """)

        self.set_grid([-10, -10], [+10, +10])
        self.set_constr(penalty=1.E+3, eps=1.E-2, with_amplitude=True)
        self.set_parameters()

    @property
    def args_constr(self):
        return {'d': 2}

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

        isign = np.ones(30)
        isign[::2] *= -1

        snmu = isign / np.sqrt(1 + mu_2)
        A = 2 * snmu / (mu * (1 / (1 + mu_2) + 1))
        r = (np.exp(-mu_2[None, :] * X[:, 0][:, None] ** 2) - 2) * \
             np.exp(-mu_2[None, :] * X[:, 1][:, None] ** 2) + 1
        aux = (A ** 2) * (r ** 2) * (1 / (1 + mu_2) + 1) / 2 + \
              2 * A * r / mu_2 * snmu * (1 / mu + mu)
        return -1 * (0.0001 - aux.sum(-1))

    def target_batch(self, X):
        return X[:, 0] ** 2 + X[:, 1] ** 2