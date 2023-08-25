import numpy as np
import opt_einsum
from teneva_bm import Bm


try:
    import jax
    import jax.numpy as jnp
    with_jax = True
except Exception as e:
    with_jax = False


class BmDecompTt(Bm):
    def __init__(self, d=128, n=32, seed=42, name=None, r=5):
        super().__init__(d, n, seed, name)

        self.set_desc("""
            Tensor network Tensor Train (TT).
            As part of this benchmark, a random TT network is generated, for
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
                'desc': 'Lower limit for random TT-cores',
                'kind': 'float',
                'form': '.2f',
                'dflt': -1.
            },
            'rand_b': {
                'desc': 'Upper limit for random TT-cores',
                'kind': 'float',
                'form': '.2f',
                'dflt': +1.
            }
        }

    @property
    def ref(self):
        i = [11, 2, 5, 9, 5, 4, 7, 8, 2, 3, 11, 8, 13, 4, 15, 8]
        i += [10] * (self.d - len(i))
        return np.array(i, dtype=int), -764965056.0

    def prep_bm(self):
        rng = jax.random.PRNGKey(self.seed)
        rng, key = jax.random.split(rng)
        self._Y = _rand(self.d, self.n0, self.r, key,
            a=self.rand_a, b=self.rand_b)
        self._get_many = jax.jit(_get_many)

    def full(self):
        return np.array(_full(self._Y))

    def target_batch(self, I):
        return np.array(self._get_many(self._Y, I), dtype=float)


def _full(Y):
    """Export TT-tensor to the full (jax.numpy) format."""
    Yl, Ym, Yr = Y

    Z = Yl[0, :, :]
    for G in Ym:
        Z = jnp.tensordot(Z, G, 1)
    Z = jnp.tensordot(Z, Yr[:, :, 0], 1)

    return Z


def _get_many(Y, I):
    """Compute the elements of the TT-tensor on many multi-indices."""
    def body(Q, data):
        i, G = data
        Q = jnp.einsum('kq,qkr->kr', Q, G[:, i, :])
        return Q, None

    Yl, Ym, Yr = Y

    Q = Yl[0, I[:, 0], :]
    Q, _ = jax.lax.scan(body, Q, (I[:, 1:-1].T, Ym))
    Q, _ = body(Q, (I[:, -1], Yr))

    return Q[:, 0]


def _rand(d, n, r, key, a=-1., b=1.):
    """Construct a random TT-tensor from the uniform distribution."""
    keyl, keym, keyr = jax.random.split(key, 3)

    Yl = jax.random.uniform(keyl, (1, n, r), minval=a, maxval=b)
    Ym = jax.random.uniform(keym, (d-2, r, n, r), minval=a, maxval=b)
    Yr = jax.random.uniform(keyr, (r, n, 1), minval=a, maxval=b)

    return [Yl, Ym, Yr]
