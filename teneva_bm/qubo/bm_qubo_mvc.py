import numpy as np
from teneva_bm import Bm


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


class BmQuboMvc(Bm):
    def __init__(self, d=100, n=2, seed=42, name=None, pcon=5):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Quadratic unconstrained binary optimization (QUBO) Minimum Vertex
            Cover (MVC) problem represented as a discrete function. The
            dimension may be any (default is 100), and the mode size should be
            2. The default value of the probability of the connection in the
            graph if 0.5 (pcon=5). The benchmark needs "networkx==3.0" and
            "qubogen==0.1.1" libraries.
        """)

        if not with_networkx:
            msg = 'Need "networkx" module. For installation please run '
            msg += '"pip install networkx==3.0"'
            self.set_err(msg)

        if not with_qubogen:
            msg = 'Need "qubogen" module. For installation please run '
            msg += '"pip install qubogen==0.1.1"'
            self.set_err(msg)

        self.pcon = pcon

    @property
    def args_constr(self):
        return {'n': 2, 'pcon': [1, 2, 3, 4, 5, 6, 7, 8, 9]}

    @property
    def args_info(self):
        return {**super().args_info,
            'pcon': {
                'desc': 'Connection probability',
                'kind': 'int'
            }
        }

    @property
    def identity(self):
        return ['d', 'pcon', 'seed']

    @property
    def is_tens(self):
        return True

    @property
    def ref(self):
        i = np.zeros(100, dtype=int)
        for k in [0, 12, 34, 44, 53, 65, 99]:
            i[k] = 1
        return np.array(i, dtype=int), -3583.0

    def prep_bm(self):
        d = self.d
        p = 0.1 * self.pcon
        graph = nx.fast_gnp_random_graph(n=d, p=p, seed=self.seed)
        edges = np.array(list([list(e) for e in graph.edges]))
        n_nodes = len(np.unique(np.array(edges).flatten()))

        g = qubogen.Graph(edges=edges, n_nodes=n_nodes)
        self._Q = qubogen.qubo_mvc(g)

    def target_batch(self, I):
        return ((I @ self._Q) * I).sum(axis=1)
