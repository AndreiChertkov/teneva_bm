from teneva_bm import Bm
import opt_einsum
import jax.numpy as jnp
import numpy as np
import teneva


DESC = """
    Pairwise-entangled paired states. 
    This is some kind of physical quantum prikol, so this optimization problem 
    is of high importance.
"""


class BmDecompPeps(Bm):
    def __init__(
        self,
        size_x,
        size_y,
        dim_inner_x,
        dim_inner_y,
        dim_boundary,
        name="DecompPeps",
        desc=DESC,
    ):
        super().__init__(d=size_x * size_y, n=dim_boundary, name=name, desc=desc)
        self.size_x = size_x
        self.size_y = size_y

        self.dim_boundary = dim_boundary
        self.dim_inner_x = dim_inner_x
        self.dim_inner_y = dim_inner_y

        p = self.pepes = []
        for x in range(size_x):
            for y in range(size_y):
                rr = [dim_boundary, dim_inner_x, dim_inner_y, dim_inner_x, dim_inner_y]
                if x == 0:
                    rr[2] = 1

                if x == size_x - 1:
                    rr[4] = 1

                if y == 0:
                    rr[1] = 1

                if y == size_y - 1:
                    rr[3] = 1

                p.append(jnp.array(np.random.rand(*rr)))

        self.c_idx = self.make_contract_idx()

        self.prep()

    @property
    def is_func(self):
        return False

    def make_contract_idx(self):
        size_x = self.size_x
        size_y = self.size_y

        big_num_x = 1000
        big_num_y = 2 * big_num_x

        return [
            [
                big_num_x * (y + 1) + x,
                big_num_y + x + (y - 1) * size_x,
                big_num_x * (y + 1) + x + 1,
                big_num_y + x + y * size_x,
            ]
            for x in range(size_x)
            for y in range(size_y)
        ]

    def full(self):
        a = []
        cnt = 0
        for x in range(self.size_x):
            for y in range(self.size_y):
                # a.extend([self.pepes[cnt][idxs[x, y], ...], self.c_idx[cnt]])
                a.extend([self.pepes[cnt], [cnt] + self.c_idx[cnt]])
                cnt += 1

        a.append(list(np.arange(self.size_x * self.size_y)))

        path_full = self.path_full = opt_einsum.contract_path(*a)
        return jnp.einsum(*a, optimize=path_full[0])

    def target(self, idxs):
        a = []
        cnt = 0
        for x in range(self.size_x):
            for y in range(self.size_y):
                a.extend([self.pepes[cnt][idxs[cnt], ...], self.c_idx[cnt]])
                cnt += 1

        a.append([])

        try:
            path = self.path
        except:
            path = self.path = opt_einsum.contract_path(*a)

        return jnp.einsum(*a, optimize=path[0])


if __name__ == "__main__":
    np.random.seed(42)

    bm = BmDecompPeps(
        size_x=2, size_y=2, dim_inner_x=3, din_inner_y=3, dim_boundary=3
    ).prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.0e2)
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
