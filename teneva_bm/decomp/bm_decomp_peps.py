import numpy as np
import opt_einsum
from teneva_bm import Bm


try:
    import jax
    import jax.numpy as jnp
    with_jax = True
except Exception as e:
    with_jax = False


class BmDecompPeps(Bm):
    def __init__(self, d=None, n=16, seed=42, d_x=4, d_y=4, r_x=3, r_y=3):
        super().__init__(d_x*d_y, n, seed)

        self.set_desc("""
            Tensor network Pairwise-Entangled Paired States (PEPS).
            As part of this benchmark, a random PEPS network is generated, for
            which the task of finding the minimum (default) or maximum value
            can then be set.
        """)

        if not with_jax:
            msg = 'Need "jax" module. Please, install it manually'
            self.set_err(msg)

        if d != None:
            self.set_err('Dimension should not be set manually')

        if not self.is_n_equal:
            self.set_err('Mode size (n) should be constant')

        self.d_x = d_x
        self.d_y = d_y

        self.r_x = r_x
        self.r_y = r_y

        self._path = None
        self._path_full = None

    @property
    def args_info(self):
        return {**super().args_info,
            'd_x': {
                'desc': 'Internal "horizontal" dimension',
                'kind': 'int'
            },
            'd_y': {
                'desc': 'Internal "vertical" dimension',
                'kind': 'int'
            },
            'r_x': {
                'desc': 'Constant "horizontal" rank',
                'kind': 'int'
            },
            'r_y': {
                'desc': 'Constant "vertical" rank',
                'kind': 'int'
            }
        }

    @property
    def identity(self):
        return ['d_x', 'd_y', 'r_x', 'r_y', 'seed']

    @property
    def is_tens(self):
        return True

    @property
    def ref(self):
        i = [11, 2, 5, 9, 5, 4, 7, 8, 2, 3, 11, 8, 13, 4, 15, 8]
        return np.array(i, dtype=int), 185516.484375

    def prep_bm(self):
        rng = jax.random.PRNGKey(self.seed)

        self._cores = []
        for x in range(self.d_x):
            for y in range(self.d_y):
                rng, key = jax.random.split(rng)
                self._cores.append(jax.random.uniform(key, (
                    self.n0,
                    self.r_x if x > 0 else 1,
                    self.r_y if y > 0 else 1,
                    self.r_x if x < self.d_x - 1 else 1,
                    self.r_y if y < self.d_y - 1 else 1)))

        self._c_idx = _make_contract_idx(self.d_x, self.d_y)

    def full(self):
        k = 0
        a = []
        for x in range(self.d_x):
            for y in range(self.d_y):
                a.extend([self._cores[k], [k] + self._c_idx[k]])
                k += 1
        a.append(list(np.arange(self.d_x * self.d_y)))

        if self._path_full is None:
            self._path_full = opt_einsum.contract_path(*a)

        return jnp.einsum(*a, optimize=self._path_full[0])

    def target(self, i):
        k = 0
        a = []
        for x in range(self.d_x):
            for y in range(self.d_y):
                a.extend([self._cores[k][i[k], ...], self._c_idx[k]])
                k += 1
        a.append([])

        if self._path is None:
            self._path = opt_einsum.contract_path(*a)

        return float(jnp.einsum(*a, optimize=self._path[0]))


def _make_contract_idx(d_x, d_y, big_num_x=1000, big_num_y=2000):
    return [[
            big_num_x * (y + 1) + x,
            big_num_y + x + (y - 1) * d_x,
            big_num_x * (y + 1) + x + 1,
            big_num_y + x + y * d_x,
        ] for x in range(d_x) for y in range(d_y)]


if __name__ == "__main__":
    # Service code just for test.
    np.random.seed(42)

    bm = BmDecompPeps().prep()
    print(bm.info())
    print(bm[bm.ref[0]])

    I_trn, y_trn = bm.build_trn(1.E+2)
    print(bm.info_history())

    text = "Value at a random multi-index     :  "
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f"{y:-10.3e}"
    print(text)

    text = "Value at 3 random multi-indices   :  "
    i1 = [np.random.choice(k) for k in bm.n]
    i2 = [np.random.choice(k) for k in bm.n]
    i3 = [np.random.choice(k) for k in bm.n]
    I = [i1, i2, i3]
    y = bm[I]
    text += "; ".join([f"{y_cur:-10.3e}" for y_cur in y])
    print(text)
