import numpy as np


try:
    import networkx as nx
    with_networkx = True
except Exception as e:
    with_networkx = False


try:
    import qubogen
    with_qubogen = True
except Exception as e:
    with_qubogen = False


from teneva_bm import Bm


DESC = """
    Quadratic unconstrained binary optimization (QUBO) Max-Cut problem
    represented as a discrete function. The dimension may be any (default
    is 100), and the mode size should be 2. The benchmark needs
    "networkx==3.0" and "qubogen==0.1.1" libraries.
"""


class BmQuboMaxcut(Bm):
    def __init__(self, d=100, n=2, name='QuboMaxcut', desc=DESC):
        super().__init__(d, n, name, desc)

        if not self.is_n_equal or self.n0 != 2:
            self.set_err('Mode size (n) should be "2"')

        if not with_networkx:
            msg = 'Need "networkx" module. For installation please run '
            msg += '"pip install networkx==3.0"'
            self.set_err(msg)

        if not with_qubogen:
            msg = 'Need "qubogen" module. For installation please run '
            msg += '"pip install qubogen==0.1.1"'
            self.set_err(msg)

    @property
    def identity(self):
        return ['d', 'seed']

    @property
    def is_tens(self):
        return True

    def get_config(self):
        conf = super().get_config()
        conf['_prob_con'] = self._prob_con
        return conf

    def info(self, footer=''):
        text = ''

        text += 'Param prob_con (connection probability)  : '
        v = self._prob_con
        text += f'{v:.6f}\n'

        return super().info(text+footer)

    def prep_bm(self):
        d = self.d
        p = self._prob_con
        graph = nx.fast_gnp_random_graph(n=d, p=p, seed=self.seed)
        edges = np.array(list([list(e) for e in graph.edges]))
        n_nodes = len(np.unique(np.array(edges).flatten()))

        g = qubogen.Graph(edges=edges, n_nodes=n_nodes)
        self._Q = qubogen.qubo_max_cut(g)

    def set_opts(self, prob_con=0.5):
        """Setting options specific to the benchmark.

        There are no plans to manually change the default values.

        Args:
            prob_con (float): probability of the connection in the graph.

        """
        self._prob_con = prob_con

    def target_batch(self, I):
        return ((I @ self._Q) * I).sum(axis=1)


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmQuboMaxcut().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+4)
    print(bm.info_history())

    text = 'Value at a random multi-index            :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at 3 random multi-indices          :  '
    i1 = [np.random.choice(k) for k in bm.n]
    i2 = [np.random.choice(k) for k in bm.n]
    i3 = [np.random.choice(k) for k in bm.n]
    I = [i1, i2, i3]
    y = bm[I]
    text += '; '.join([f'{y_cur:-10.3e}' for y_cur in y])
    print(text)
