import numpy as np
import teneva


class Bm:
    def __init__(self, d=None, n=None, name='', desc=''):
        self.err = ''

        self.set_size(d, n)
        self.set_name(name)
        self.set_desc(desc)

        self.set_min()
        self.set_max()

        self.set_grid()
        self.set_grid_kind()

        self.set_opts()

        self.build_trn()
        self.build_tst()

        # TODO: add support for cache of the requested values
        # TODO: add support of min/max log of the requested values

        self.with_cores = False
        self._is_prep = False

    def __call__(self, X):
        """Return a value or batch of values for continuous function."""
        self.check()

        X = np.asanyarray(X, dtype=float)

        if self.is_tens:
            msg = f'BM "{self.name}" is a tensor. Can`t compute it in the point'
            raise ValueError(msg)
        else:
            return self._f_batch(X) if len(X.shape) == 2 else self._f(X)

    def __getitem__(self, I):
        """Return a value or batch of values for discrete function."""
        self.check()

        I = np.asanyarray(I, dtype=int)

        if self.is_func:
            X = teneva.ind_to_poi(I, self.a, self.b, self.n, self.grid_kind)
            return self(X)
        else:
            return self._f_batch(I) if len(I.shape) == 2 else self._f(I)

    @property
    def is_func(self):
        """Check if BM relates to function (i.e., continuous function)."""
        return not self.is_tens

    @property
    def is_n_equal(self):
        """Check if all the mode sizes are the same."""
        if self.n is None:
            return True
        return len(set(self.n)) == 1

    @property
    def is_n_even(self):
        """Check if all the mode sizes are even."""
        if self.n is None:
            return True
        for k in self.n:
            if k % 2 == 1:
                return False
        return True

    @property
    def is_n_odd(self):
        """Check if all the mode sizes are odd."""
        return not self.is_n_even

    @property
    def is_tens(self):
        """Check if BM relates to tensor (i.e., discrete function)."""
        return not self.is_func

    def build_cores(self):
        """Return exact TT-cores for the TT-representation of the tensor."""
        if self.is_tens:
            msg = 'Construction of the TT-cores does not work for tensors'
            raise ValueError(msg)

        I = np.array([teneva.grid_flat(k) for k in self.n], dtype=int).T
        X = teneva.ind_to_poi(I, self.a, self.b, self.n, self.grid_kind)

        return self._cores(X)

    def build_trn(self, m=0):
        """Generate random (from LHS) train dataset of (index, value)."""
        m = int(m)

        if m < 1:
            self.I_trn = None
            self.y_trn = None
        else:
            self.I_trn = teneva.sample_lhs(self.n, m)
            self.y_trn = self[self.I_trn]

        return self.I_trn, self.y_trn

    def build_tst(self, m=0):
        """Generate random (from "choice") test dataset of (index, value)."""
        m = int(m)

        if m < 1:
            self.I_tst = None
            self.y_tst = None
        else:
            self.I_tst = np.vstack([np.random.choice(k, m) for k in self.n]).T
            self.y_tst = self[self.I_tst]

        return self.I_tst, self.y_tst

    def check(self):
        """Check that benchmark's configuration is valid."""
        if not self._is_prep:
            self.set_err('Run "prep" method for BM before call it')

        if self.err:
            msg = f'BM "{self.name}" is not ready\n   Error > {self.err}'
            raise ValueError(msg)
            return False

        return True

    def get(self, I):
        """Alias for "__getitem__" method."""
        return self[I]

    def info(self):
        """Returns a detailed description of the benchmark as text."""
        text = '-' * 78 + '\n' + 'BM: '
        text += self.name + ' ' * max(0, 36-len(self.name)) +  ' | '
        text += f'DIMS = {self.d:-5d} | <MODE SIZE> = {np.mean(self.n):-6.1f}\n'

        if self.y_min_real is not None or self.y_max_real is not None:
            text += ' ' * 35
            if self.y_min_real is not None:
                text += f'y_min = {self.y_min_real:-12.5e} | '
            if self.y_max_real is not None:
                text += f'y_max = {self.y_max_real:-12.5e}'
            text += '\n'

        if self.desc:
            desc = f'  [ {self.desc.strip()} ]'
            text += '\n' + desc.replace('            ', '    ') # TODO

        text += '\n' + '=' * 78 + '\n'
        return text

    def prep(self):
        """A function with a specific benchmark preparation code."""
        self._is_prep = True
        return self

    def set_desc(self, desc=''):
        """Set text description of the problem."""
        self.desc = desc

    def set_err(self, err=''):
        """Set the error text (can not import external module, etc.)."""
        self.err = (self.err + '; ' if self.err else '') + err

    def set_grid(self, a=None, b=None):
        """Set grid lower (a) and upper (b) limits for the function-like BM."""
        self.a = teneva.grid_prep_opt(a, self.d)
        self.b = teneva.grid_prep_opt(b, self.d)

    def set_grid_kind(self, kind='cheb'):
        """Set the kind of the grid ('cheb' or 'uni')."""
        self.grid_kind = kind

        if not self.grid_kind in ['uni', 'cheb']:
            msg = f'Invalid kind of the grid (should be "uni" or "cheb")'
            raise ValueError(msg)

    def set_max(self, i=None, x=None, y=None):
        """Set exact (real) global maximum (index, point and related value)."""
        self.i_max_real = i
        self.x_max_real = x
        self.y_max_real = y

    def set_min(self, i=None, x=None, y=None):
        """Set exact (real) global minimum (index, point and related value)."""
        self.i_min_real = i
        self.x_min_real = x
        self.y_min_real = y

    def set_name(self, name=''):
        """Set display name for the problem."""
        self.name = name

    def set_opts(self):
        """Setting options specific to the benchmark."""
        return

    def set_size(self, d=None, n=None):
        """Set dimension (d) and sizes for all d-modes (n: int or list)."""
        self.d = None if d is None else int(d)
        self.n = teneva.grid_prep_opt(n, self.d, int)

    def _cores(self, X):
        """Return the exact TT-cores for the provided points."""
        raise NotImplementedError()

    def _cores_add(self, X, a0=0):
        """Helper function for the construction of the TT-cores."""
        Y = []

        for x in X:
            G = np.ones([2, len(x), 2])
            G[1, :, 0] = 0.
            G[0, :, 1] = x
            Y.append(G)

        Y[0] = Y[0][0:1, ...].copy()
        Y[-1] = Y[-1][..., 1:2].copy()
        Y[-1][0, :, 0] += a0

        return Y

    def _cores_mul(self, X):
        """Helper function for the construction of the TT-cores."""
        return [x[None, :, None] for x in X]

    def _f(self, x):
        """Function that computes value for a given point/index."""
        return self._f_batch(np.array(x).reshape(1, -1))[0]

    def _f_batch(self, X):
        """Function that computes values for a given batch of points/indices."""
        return np.array([self._f(x) for x in X])
