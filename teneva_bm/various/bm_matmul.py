import numpy as np
import teneva


from teneva_bm import Bm


DESC = """
    Problem for fast "size x size" matrix multiplication (like Strassen
    algorithm in the case "size = 2") in terms of the tensor based
    formulation (e.g., the search for the rank-7 CP decomposition is
    performed in the case of the "size = 2"). The objective function is
    the norm of the difference between the exact tensor and the result of
    multiplication using the chosen operations (determined by CP factors).
    The dimension is determined automatically, but the rank argument should
    be set manually in the case if "size > 2" (e.g., for the "size = 3", the
    best known rank is "23"). The default mode size is "3", which relates to
    a standard matrix multiplication task; in this case, we are looking for a
    decomposition with factor matrix entries "-1, 0, 1". A mode size of "5"
    is also supported, which relates to the possible values "-2, -1, 0, 1, 2".
    If the "only2" flag is set during initialization, then only two factor
    matrices will be constructed, and the third matrix will be restored as a
    solution to the corresponding system of linear equations.
    For more details, see the work Fawzi, A., et al. "Discovering faster
    matrix multiplication algorithms with reinforcement learning." Nature
    610.7930 (2022): 47-53.
"""


class BmMatmul(Bm):
    def __init__(self, d=None, n=3, name='Matmul', desc=DESC, size=2, rank=7,
                 only2=False):
        T_real = _tensor_generate(size, size, size)
        d_real = (2 if only2 else 3) * size**2 * rank

        super().__init__(d_real, n, name, desc)

        if d is not None:
            self.set_err('Dimension number (d) should not be set manually')

        # Possible items of the factor matrices:
        if n == 3:
            E = [-1, 0, 1]
        elif n == 5:
            E = [-2, -1, 0, 1, 2]
        else:
            E = []
            self.set_err('Invalid mode size. May be only 3 and 5')

        if size == 2 and n == 3:
            # Strassen algorithm:
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

        self.T = T_real
        self.E = E
        self.size = size
        self.rank = rank
        self.only2 = only2

    @property
    def is_tens(self):
        return True

    def prep(self):
        self.check_err()

        self.loss = _loss_build(self.T, self.E, self.rank, self.only2)

        self.is_prep = True
        return self

    def recover(self, i):
        x = _ind_to_poi(i, self.E)

        if self.only2:
            U, V = _factor_from_poi(x, self.rank, True)
            W = _factor_recover(U, V, self.T)
        else:
            U, V, W = _factor_from_poi(x, self.rank, False)

        return U, V, W

    def _f(self, i):
        return self.loss(i)


def _factor_from_poi(x, rank, only2=False, order_spec=False):
    """Build canonical rank-"rank" factors from flatten "x"."""
    k = 2 if only2 else 3
    n = x.size // (k * rank)

    if order_spec and not only2:
        raise ValueError('Is not supported')

    elif order_spec:
        U = np.array([x[n*2*j:n*2*j+n] for j in range(rank)]).T
        V = np.array([x[n*2*j+n:n*2*j+2*n] for j in range(rank)]).T

    else:
        U = x[:n*rank].reshape((n, rank))
        V = x[n*rank:2*n*rank].reshape((n, rank))

    if only2:
        return U, V

    W = x[2*n*rank:].reshape((n, rank))

    return U, V, W


def _factor_recover(U, V, T):
    """Build 3th factor matrix from 2 given factor matrices and 3D tensor."""
    n = T.shape[-1]
    rank = U.shape[-1]

    A = np.einsum('nr,mr->nmr', U, V).reshape(-1, rank)
    R = T.reshape(-1, n)

    W = np.linalg.lstsq(A, R, rcond=-1)[0].T

    return W


def _ind_to_poi(I, E=[-1, 0, 1]):
    """Transform tensor multi-index into point from discrete values "E"."""
    return np.asarray(E)[list(I)]


def _loss_build(T_real, E, rank, only2=True, order_spec=False, fast=False):
    """Prepare the loss function for the tensor based formulation."""
    if fast:
        T = T_real.reshape(-1, T_real.shape[-1])

    def loss(i):
        x = _ind_to_poi(i, E)

        if only2:
            U, V = _factor_from_poi(x, rank, True, order_spec)
            if not fast:
                W = _factor_recover(U, V, T_real)
        else:
            U, V, W = _factor_from_poi(x, rank, False, order_spec)

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

    bm = BmMatmul(size=2, rank=7).prep()
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

    if bm.size == 2 and bm.n[0] == 3:
        text = 'Value at the minimum (real vs calc)      :  '
        y_real = bm.y_min_real
        y_calc = bm[bm.i_min_real]
        text += f'{y_real:-10.3e}       /      {y_calc:-10.3e}'
        print(text)

    text = 'Check "only2" case                       :  '
    bm = BmMatmul(size=2, rank=7, only2=True).prep()
    i = [np.random.choice(k) for k in bm.n]
    y = bm[i]
    text += f'{y:-10.3e}'
    print(text)

    text = 'Check "recover" error                    :  '
    bm = BmMatmul(size=2, rank=7).prep()
    U, V, W = bm.recover(bm.i_min_real)
    T_appr = np.einsum('nr,mr,sr->nms', U, V, W)
    e = np.linalg.norm(T_appr.reshape(-1) - bm.T.reshape(-1))
    text += f'{e:-10.3e}'
    print(text)
