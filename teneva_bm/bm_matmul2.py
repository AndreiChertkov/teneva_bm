import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Problem for fast 2x2 matrix multiplication (like Strassen algorithm)
    in terms of the tensor based formulation (the search for the rank-7 CP
    decomposition is performed).
    The dimension and mode size are determined automatically.
"""


class BmMatmul2(Bm):
    def __init__(self, name='Matmul2', desc=DESC, only2=False):
        size = 2
        rank = 7
        E = [-1, 0, 1] # Possible items of the factor matrices
        T = _tensor_generate(size, size, size)
        d = (2 if only2 else 3) * T.shape[0] * rank
        n = len(E)

        super().__init__(d, n, name, desc)

        self.set_min(i=np.array([
            2, 1, 2, 1, 2, 0, 1,
            1, 1, 1, 1, 2, 1, 2,
            1, 2, 1, 1, 1, 2, 1,
            2, 2, 1, 2, 1, 1, 0,

            2, 2, 1, 0, 1, 2, 1,
            1, 1, 2, 1, 1, 2, 1,
            1, 1, 1, 2, 1, 1, 2,
            2, 1, 0, 1, 2, 1, 2,

            2, 1, 1, 2, 0, 1, 2,
            1, 1, 2, 1, 2, 1, 1,
            1, 2, 1, 2, 1, 1, 1,
            2, 0, 2, 1, 1, 2, 1], dtype=int),
            y=0.)

        self.T = T
        self.E = E
        self.size = size
        self.rank = rank
        self.only2 = only2

    @property
    def is_tens(self):
        return True

    def prep(self):
        self.loss = _loss_build(self.T, self.E, self.rank, self.only2)

        self._is_prep = True
        return self

    def recover(self, i):
        x = _ind_to_poi(i, self.E)

        if self.only2:
            U, V = _factor_from_poi(x, self.rank, True)
            W = _factor_recover(U, V, T)
        else:
            U, V, W = _factor_from_poi(x, self.rank, False)

        return U, V, W

    def _f(self, i):
        return self.loss(i)


def _factor_from_poi(x, q, only2=False, order_spec=False):
    """Build canonical rank-q factors from flatten "x"."""
    k = 2 if only2 else 3
    n = x.size // (k * q)

    if order_spec and not only2:
        raise ValueError('Is not supported')
    elif order_spec:
        U = np.array([x[n*2*j:n*2*j+n] for j in range(q)]).T
        V = np.array([x[n*2*j+n:n*2*j+2*n] for j in range(q)]).T
    else:
        U = x[:n*q].reshape((n, q))
        V = x[n*q:2*n*q].reshape((n, q))

    if only2:
        return U, V

    W = x[2*n*q:].reshape((n, q))

    return U, V, W


def _factor_recover(U, V, T):
    """Build 3th factor matrix from 2 given factor matrices and 3D tensor."""
    n = T.shape[-1]
    q = U.shape[-1]

    A = np.einsum('nr,mr->nmr', U, V).reshape(-1, q)
    R = T.reshape(-1, n)

    W = np.linalg.lstsq(A, R, rcond=-1)[0].T

    return W


def _ind_to_poi(I, E=[-1, 0, 1]):
    """Transform tensor multi-index into point from discrete values "E"."""
    return np.asarray(E)[list(I)]


def _loss_build(T_real, E, q, only2=True, order_spec=False, fast=False):
    """Prepare the loss function for the tensor based formulation."""
    if fast:
        T = T_real.reshape(-1, T_real.shape[-1])

    def loss(i):
        x = _ind_to_poi(i, E)

        if only2:
            U, V = _factor_from_poi(x, q, True, order_spec)
            if not fast:
                W = _factor_recover(U, V, T_real)
        else:
            U, V, W = _factor_from_poi(x, q, False, order_spec)

        if only2 and fast:
            # NOTE: this code may be invalid now (compare with fast=False)
            A = np.einsum('nr,mr->nmr', U, V).reshape(-1, U.shape[-1])
            Q = np.linalg.qr(A)[0]
            D = T - Q @ (Q.T @ T)
            e = np.linalg.norm(D.reshape(-1))
        else:
            T_appr = np.einsum('nr,mr,sr->nms', U, V, W)
            e = np.linalg.norm(T_appr.reshape(-1) - T_real.reshape(-1))

        return e

    return loss


def _tensor_generate(a, b, c):
    """Generate the matrix multiplication tensor T_{a, b, c}."""
    T = np.full((a*b, b*c, c*a), 0, dtype=int)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                T[i * b + j][j * c + k][k + i * c] = 1
    return T


if __name__ == '__main__':
    np.random.seed(42)

    bm = BmMatmul2().prep()
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

    text = 'Value at minimum (real vs calc)   :  '
    y_real = bm.y_min_real
    y_calc = bm[bm.i_min_real]
    text += f'{y_real:-10.3e}/ {y_calc:-10.3e}'
    print(text)
