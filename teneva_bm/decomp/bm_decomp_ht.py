import numpy as np
import opt_einsum
from teneva_bm import Bm


try:
    import jax
    import jax.numpy as jnp
    with_jax = True
except Exception as e:
    with_jax = False


class BmDecompHt(Bm):
    def __init__(self, d=128, n=32, seed=42, name=None, r=5):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Tensor network Hierarchical Tucker (HT).
            As part of this benchmark, a random HT network is generated, for
            which the task of finding the maximum (default) or minimum value
            can then be set.
        """)

        if not with_jax:
            msg = 'Need "jax" module. Please, install it manually'
            self.set_err(msg)

        self.r = r

    @property
    def args_constr(self):
        return {'n': 'equal'}

    @property
    def args_info(self):
        return {
            **super().args_info,
            'r': {
                'desc': 'Constant rank',
                'kind': 'int'
            }
        }

    @property
    def identity(self):
        return ['d', 'n' 'r', 'seed']

    @property
    def is_opti_max(self):
        return True

    @property
    def is_tens(self):
        return True

    @property
    def opts_info(self):
        return {**super().opts_info,
            'rand_a': {
                'desc': 'Lower limit for random HT-cores',
                'kind': 'float',
                'form': '.2f',
                'dflt': -1.
            },
            'rand_b': {
                'desc': 'Upper limit for random HT-cores',
                'kind': 'float',
                'form': '.2f',
                'dflt': +1.
            }
        }

    @property
    def ref(self):
        i = [11, 2, 5, 9, 5, 4, 7, 8, 2, 3, 11, 8, 13, 4, 15, 8]
        i += [10] * (self.d - len(i))
        return np.array(i, dtype=int), -7.731874618171982e+18

    def prep_bm(self):
        rng = jax.random.PRNGKey(self.seed)
        rng, key = jax.random.split(rng)
        self._Y = _rand(self.d, self.n0, self.r, key,
            a=self.rand_a, b=self.rand_b)
        self._get_many = jax.jit(jax.vmap(_get, (None, 0)))

    def target_batch(self, I):
        return np.array(self._get_many(self._Y, I), dtype=float)


def _get(Y, i):
    """Compute the element of the HT-tensor."""
    def body_leaf(q, data):
        i1, i2, G1, G2, G = data
        q = jnp.einsum('r,q,rsq->s', G1[i1], G2[i2], G)
        return None, q

    def body(q, data):
        g1, g2, G = data
        q = jnp.einsum('r,q,rsq->s', g1, g2, G)
        return None, q

    # Compute for the first level (leafs):
    _, q = jax.lax.scan(body_leaf, None,
        (i[0::2], i[1::2], Y[0][0::2], Y[0][1::2], Y[1]))

    # Compute for the inner levels:
    for k in range(1, len(Y)-2):
        _, q = jax.lax.scan(body, None, (q[0::2], q[1::2], Y[k+1]))

    # Compute for the last level (root):
    q = jnp.einsum('r,q,rq->', q[0], q[1], Y[-1])

    return q


def _rand(d, n, r, key, a=-1., b=1.):
    """Construct a random HT-tensor from the uniform distribution."""
    q = d.bit_length() # Full number of levels (e.g., d=8 -> q=4)

    if isinstance(r, int):
        r = [r] * (q-1)

    if len(r) != (q-1):
        raise ValueError('Invalid length of ranks list')

    Y = []

    def _rand_level(key, sh):
        key, key_cur = jax.random.split(key)
        Yl = jax.random.uniform(key_cur, sh, minval=a, maxval=b)
        return Yl, key

    # Build the first level (leafs):
    Yl, key = _rand_level(key, sh=(d, n, r[0]))
    Y.append(Yl) # 3D tensor (len, n, r_up)

    # Build the inner levels:
    for k in range(1, q-1):
        dl = 2**(q-k-1) # Length of the current layer
        Yl, key = _rand_level(key, sh=(dl, r[k-1], r[k], r[k-1]))
        Y.append(Yl) # 4D tensor (len, r_down, r_up, r_down)

    # Build the last level (root):
    Yl, key = _rand_level(key, sh=(r[-1], r[-1]))
    Y.append(Yl) # 2D tensor (r_down, r_down)

    return Y
