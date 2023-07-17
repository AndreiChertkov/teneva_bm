import numpy as np
import teneva


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
    Quadratic unconstrained binary optimization (QUBO) Minimum Vertex Cover
    (MVC) problem represented as a discrete function.
    The dimension may be any (default is 50), and the mode size should be 2.
    The benchmark needs "networkx==3.0" and "qubogen==0.1.1" libraries.
"""


class BmQuboMvc(Bm):
    def __init__(self, d=50, n=2, name='QuboMVC', desc=DESC):
        super().__init__(d, n, name, desc)

        if not self.is_n_equal or self.n[0] != 2:
            self.set_err('Mode size (n) should be "2"')
        if not with_networkx:
            self.set_err('Need "networkx" module')
        if not with_qubogen:
            self.set_err('Need "qubogen" module')

    @property
    def is_tens(self):
        return True

    def prep(self):
        graph = nx.fast_gnp_random_graph(n=self.d, p=self.opt_prob_con,
            seed=self.opt_seed)
        edges = np.array(list([list(e) for e in graph.edges]))
        n_nodes = len(np.unique(np.array(edges).flatten()))
        g = qubogen.Graph(edges=edges, n_nodes=n_nodes)
        self.bm_Q = qubogen.qubo_mvc(g)

        self._is_prep = True
        return self

    def set_opts(self, prob_con=0.5, seed=42):
        """Setting options specific to this benchmark.

        Args:
            prob_con (float): probability of the connection in the graph.
            seed (int): seed for the random graph.

        """
        self.opt_prob_con = prob_con
        self.opt_seed = seed

    def _f_batch(self, I):
        return ((I @ self.bm_Q) * I).sum(axis=1)


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmQuboMvc().prep()
    print(bm.info())

    text = 'Range of y for 10K random samples : '
    bm.build_trn(1.E+4)
    text += f'[{np.min(bm.y_trn):-10.3e},'
    text += f' {np.max(bm.y_trn):-10.3e}] '
    text += f'(avg: {np.mean(bm.y_trn):-10.3e})'
    print(text)

    text = 'Value at a random multi-index     :  '
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Value at 3 random multi-indices   :  '
    i1 = [np.random.choice(k) for k in bm.n]
    i2 = [np.random.choice(k) for k in bm.n]
    i3 = [np.random.choice(k) for k in bm.n]
    I = [i1, i2, i3]
    y = bm[I]
    text += '; '.join([f'{y_cur:-10.3e}' for y_cur in y])
    print(text)