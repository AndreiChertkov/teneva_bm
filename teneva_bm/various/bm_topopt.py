import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from teneva_bm import Bm


class BmTopopt(Bm):
    def __init__(self, d=None, n=2, seed=42, name=None, nx=128, ny=32):
        super().__init__(int(nx*ny), n, seed, name)

        self.set_desc("""
            DRAFT!!! Discrete Topology optimization task.
            The dimension is defined from "nx" and "ny" parameters and mode size
            should be 2.
        """)

        if d is not None:
            self.set_err('Dimension number (d) should not be set manually')

        self.nx = int(nx)
        self.ny = int(ny)

    @property
    def args_constr(self):
        return {'n': 2}

    @property
    def args_info(self):
        return {**super().args_info,
            'nx': {
                'desc': 'Grid x-size',
                'kind': 'int'
            },
            'ny': {
                'desc': 'Grid y-size',
                'kind': 'int'
            }
        }

    @property
    def identity(self):
        return ['nx', 'ny']

    @property
    def is_tens(self):
        return True

    @property
    def opts_info(self):
        return {**super().opts_info,
            'k_frac': {
                'desc': 'Param k_frac for Topopt',
                'kind': 'float',
                'form': '.6f',
                'dflt': 0.4
            },
            'penal': {
                'desc': 'Param penal for Topopt',
                'kind': 'float',
                'form': '.6f',
                'dflt': 3.
            },
            'rmin': {
                'desc': 'Param rmin for Topopt',
                'kind': 'float',
                'form': '.6f',
                'dflt': 5.4
            }
        }

    @property
    def ref(self):
        i = np.ones(4096, dtype=int)
        for k in [1, 10, 12, 40, 55, 100, 254, 999, 2044, 3046]:
            i[k] = 0
        return np.array(i, dtype=int), 283.8184626040463

    @property
    def with_show(self):
        return True

    def prep_bm(self):
        self._solver = _topopt_lite(self.nx, self.ny,
            self.k_frac, self.penal, self.rmin)

    def show(self, fpath=None, i=None, best=True):
        i, y = self.get_solution(i, best)

        x = i.reshape(self.nx, self.ny).T

        fig, ax = plt.subplots(1, 1, figsize=(21, 7))
        ax.imshow(x, cmap='gray', interpolation='none')

        k_frac_real = np.sum(x) / np.size(x)

        title = ''
        title += f'Shape: {x.shape[0]} x {x.shape[1]}. '
        title += f'Frac: {k_frac_real:.2f} ({self.k_frac:.2f}). '
        title += f'Value: {y:.2f}.'
        fig.suptitle(title, fontsize=18)

        fpath = self.path_build(fpath, 'png')
        plt.savefig(fpath, bbox_inches='tight') if fpath else plt.show()

    def target(self, i):
        return self._solver(i)

    def _optimize_baseline(self):
        """Draft!"""
        solver = TopOptLite(self.nx, self.ny, self.penal, self.rmin)
        solver.init(Emin=1.E-9, Emax=1.0)
        solver.prep()
        x_ini = self.k_frac * np.ones(self.nx * self.ny, dtype=float)
        x_opt = solver.solve(x_ini)
        return x_opt


class TopOptLite:
    def __init__(self, nx, ny, penal, rmin):
        self.nx = nx
        self.ny = ny
        self.penal = penal
        self.rmin = rmin

    def init(self, Emin=1.E-9, Emax=1.0):
        self.Emin = Emin
        self.Emax = Emax

        self.n = self.nx * self.ny
        self.ndof = 2 * (self.nx+1) * (self.ny+1)

        # Load (RHS vector):
        self.f = np.zeros((self.ndof, 1))
        self.f[1, 0] = -1

        return self

    def prep(self):
        self.edofMat = _edof_matrix(self.nx, self.ny)
        self.KE = _stiffness_matrix()
        self.iK = np.kron(self.edofMat, np.ones((8,1))).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1,8))).flatten()

        # Filter: Build and assemble the index+data vectors for the coo matrix:
        nfilter = int(self.n*((2*(np.ceil(self.rmin)-1)+1)**2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(self.nx):
            for j in range(self.ny):
                row = i*self.ny + j
                kk1 = int(np.maximum(i - (np.ceil(self.rmin)-1), 0))
                kk2 = int(np.minimum(i + np.ceil(self.rmin), self.nx))
                ll1 = int(np.maximum(j - (np.ceil(self.rmin)-1), 0))
                ll2 = int(np.minimum(j + np.ceil(self.rmin), self.ny))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k*self.ny + l
                        fac = self.rmin - np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0., fac)
                        cc = cc + 1

        # Finalize assembly and convert to csc format:
        self.H = coo_matrix((sH, (iH, jH)), shape=(self.n, self.n)).tocsc()
        self.Hs = self.H.sum(1)

        # BC's and support:
        dofs = np.arange(self.ndof)
        fixed = np.union1d(dofs[0:2*(self.ny+1):2], np.array([self.ndof-1]))
        self.free = np.setdiff1d(dofs, fixed)

        return self

    def iter(self, x, with_opt=False):
        obj_penal = self.Emin + (self.Emax - self.Emin) * x**self.penal

        K = self.KE.flatten()[np.newaxis].T * obj_penal
        K = coo_matrix((K.flatten(order='F'), (self.iK, self.jK)),
            shape=(self.ndof, self.ndof)).tocsc()
        K = K[self.free, :][:, self.free]

        u = np.zeros((self.ndof, 1))
        u[self.free, 0] = spsolve(K, self.f[self.free, 0])
        u_curr = u[self.edofMat].reshape(self.n, 8)
        ce = (np.dot(u_curr, self.KE) * u_curr).sum(1)

        obj = (obj_penal * ce).sum()

        return (obj, ce) if with_opt else obj

    def solve(self, x, dx_min=0.01, loop_max=2000):
        for loop in range(loop_max):
            obj, ce = self.iter(x, with_opt=True)

            dc = -self.penal * (self.Emax - self.Emin) * x**(self.penal-1) * ce
            dc = np.asarray((self.H * (x * dc))[np.newaxis].T / self.Hs)[:, 0]
            dc /= np.maximum(0.001, x)
            dv = np.ones(self.n)

            x0 = x.copy()
            x = self._optimize(x, dc, dv)
            dx = np.linalg.norm(x.reshape(-1, 1) - x0.reshape(-1, 1), np.inf)

            print(f'# {loop:-5d} | J = {obj:-14.8f} | dx = {dx:14.8f}')

            if dx <= dx_min:
                break

        return x

    def _optimize(self, x, dc, dv, g=0, l1=0., l2=1.E+9, move=0.2):
        xnew = np.zeros(self.n)

        while (l2-l1) / (l1+l2) > 1.E-3:
            lmid = 0.5 * (l2+l1)
            xnew = np.maximum(
                0.,
                np.maximum(
                    x - move,
                    np.minimum(
                        1.,
                        np.minimum(x+move, x*np.sqrt(-dc/dv/lmid))
                    )
                )
            )

            gt = g + np.sum((dv * (xnew-x)))

            if gt > 0:
                l1 = lmid
            else:
                l2 = lmid

        return xnew


def _edof_matrix(nx, ny):
    edofMat = np.zeros((nx * ny, 8), dtype=int)
    for elx in range(nx):
        for ely in range(ny):
            el = ely + elx*ny
            n1 = (ny+1)*elx + ely
            n2 = (ny+1)*(elx+1) + ely
            edofMat[el, :]= np.array(
                [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
    return edofMat


def _lk(E=1, nu=0.3):
    """Element stiffness matrix."""
    k = np.array([
        1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
        -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])

    return E / (1-nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])


def _stiffness_matrix(E=1, nu=0.3):
    """Element stiffness matrix."""
    k = np.array([
        1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
        -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])

    return E / (1-nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])


def _topopt_lite(nelx, nely, volfrac, penal, rmin, ft=1):
    Emin = 1.E-9
    Emax = 1.0
    ndof = 2 * (nelx+1) * (nely+1)

    # FE: Build the index vectors for the for coo matrix format.
    KE = _lk()
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx*nely
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edofMat[el, :]= np.array(
                [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])

    # Construct the index pointers for the coo format:
    iK = np.kron(edofMat,np.ones((8,1))).flatten()
    jK = np.kron(edofMat,np.ones((1,8))).flatten()

    # Filter: Build and assemble the index+data vectors for the coo matrix:
    nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i*nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin)-1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin)-1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k*nely + l
                    fac = rmin - np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0., fac)
                    cc = cc + 1

    # Finalize assembly and convert to csc format:
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = H.sum(1)

    # BC's and support:
    dofs = np.arange(2*(nelx+1)*(nely+1))
    fixed = np.union1d(dofs[0:2*(nely+1):2], np.array([2*(nelx+1)*(nely+1)-1]))
    free = np.setdiff1d(dofs, fixed)

    # Solution and RHS vectors:
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # Set load:
    f[1, 0] = -1

    def f_loss(x):

        # Filter design variables:
        xPhys = np.asarray(H * x[np.newaxis].T / Hs)[:, 0] if ft == 1 else x

        # Setup and solve FE problem:
        sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin)))
        sK = sK.flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

        # Remove constrained dofs from matrix:
        K = K[free, :][:, free]

        # Solve system:
        u[free, 0] = spsolve(K, f[free, 0])

        # Objective and sensitivity:
        ce = np.dot(u[edofMat].reshape(nelx*nely, 8), KE)
        ce *= u[edofMat].reshape(nelx*nely, 8)
        ce = ce.sum(1)
        obj = Emin + (Emax - Emin) * xPhys**penal
        obj = (obj * ce).sum()
        return obj

    return f_loss


if __name__ == '__main__':
    # Service code just for test.
    np.random.seed(42)

    bm = BmTopopt().prep()
    print(bm.info())

    I_trn, y_trn = bm.build_trn(1.E+2)
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

    text = 'Generate image for a random multi-index  :  '
    fpath = f'result/topopt_show'
    bm.show(fpath)
    text += f' see {fpath}'
    print(text)
