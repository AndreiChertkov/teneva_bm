import numpy as np
import teneva
from time import perf_counter as tpc


class Bm:
    def __init__(self, d=None, n=None, name='', desc=''):
        self._init()

        self.set_size(d, n)
        self.set_quantization()

        self.set_name(name)
        self.set_desc(desc)

        self.set_constr()

        self.set_min()
        self.set_max()

        self.set_grid()
        self.set_grid_kind()

        self.set_opts()

        self.set_cache()

        self.set_log()

    def __call__(self, X):
        """Return a value or batch of values for provided x-point."""
        return self.get_poi(X)

    def __getitem__(self, I):
        """Return a value or batch of values for provided multi-index."""
        return self.get(I)

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

    @property
    def n0(self):
        """Return the mode size value if it is constant."""
        if not self.is_n_equal:
            raise ValueError('Mode size is not constant, can not get n0')
        return self.n[0]

    @property
    def with_constr(self):
        """Return True if benchmark has a constant."""
        return False

    def build_cores(self):
        """Return exact TT-cores for the TT-representation of the tensor."""
        if self.is_tens:
            msg = 'Construction of the TT-cores does not work for tensors'
            raise ValueError(msg)

        I = np.array([teneva.grid_flat(k) for k in self.n], dtype=int).T
        X = teneva.ind_to_poi(I, self.a, self.b, self.n, self.grid_kind)

        return self._cores(X)

    def build_trn(self, m=0, skip_process=False):
        """Generate random (from LHS) train dataset of (index, value)."""
        m = int(m)
        n = [2]*self.quantization*self.d if self.with_quantization else self.n

        if m < 1:
            return None, None

        I_trn = teneva.sample_lhs(n, m)
        y_trn = self.get(I_trn, skip_process)

        return I_trn, y_trn

    def build_tst(self, m=0, skip_process=True):
        """Generate random (from "choice") test dataset of (index, value)."""
        m = int(m)
        n = [2]*self.quantization*self.d if self.with_quantization else self.n

        if m < 1:
            return None, None

        I_tst = np.vstack([np.random.choice(k, m) for k in n]).T
        y_tst = self.get(I_tst, skip_process)

        return I_tst, y_tst

    def check(self):
        """Check that benchmark's configuration is valid."""
        if not self.is_prep:
            msg = 'Run "prep" method for BM before call it'
            self.set_err(msg)

        return self.check_err()

    def check_err(self):
        """Check that benchmark has not errors."""
        if len(self.err):
            msg = f'BM "{self.name}" is not ready'
            for e in self.err:
                msg += f'\n    Error > {e}'
            raise ValueError(msg)

        return True

    def get(self, I, skip_process=False):
        """Return a value or batch of values for provided multi-index."""
        t = tpc()

        self.check()

        I, X, is_batch = self._parse_input(I=I)

        if self.with_cache:
            m = I.shape[0]
            ind = [k for k in range(m) if tuple(I[k]) not in self.cache]

            m_new = len(ind)
            m_cache = m - m_new

            if m_new > 0:
                y_new = self._compute(X[ind] if self.is_func else I[ind])
                for k in range(m_new):
                    self.cache[tuple(I[ind[k]])] = y_new[k]

            y = np.array([self.cache[tuple(i)] for i in I])

        else:
            m_cache = 0

            y = self._compute(X if self.is_func else I)

        return self._process(I, X, y, m_cache, t, is_batch, skip_process)

    def get_poi(self, X, skip_process=False):
        """Return a value or batch of values for provided x-point."""
        t = tpc()

        self.check()

        I, X, is_batch = self._parse_input(X=X)

        y = self._compute(X)

        return self._process(I, X, y, 0, t, is_batch, skip_process)

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

    def info_history(self):
        """Returns an information about the request history (text)."""
        text = ''

        text = '-' * 78 + '\n' + 'BM: '
        text += self.name + ' ' * max(0, 36-len(self.name)) +  ' | '
        text += f'DIMS = {self.d:-5d} | <MODE SIZE> = {np.mean(self.n):-6.1f}\n'
        text += '-' * 41 + '|              '
        text += '> History of requests.'
        text += '\n'

        if self.m == 0:
            text += '        ... history is empty ...\n'
            text += '=' * 78 + '\n'
            return text

        text += 'Number of requests                       : '
        text += f'{self.m:-10.3e}\n'

        text += 'Number of cache uses                     : '
        text += f'{self.m_cache:-10.3e}\n'

        text += 'Average time of one request (sec)        : '
        text += f'{self.time/self.m:-10.3e}\n'

        if self.y_min is not None and self.y_min_real is not None:
            text += 'Minimum (found / real)                   : '
            text += f'{self.y_min:-10.3e}   / {self.y_min_real:-10.3e}\n'
        elif self.y_min is not None:
            text += 'Minimum (found)                          : '
            text += f'{self.y_min:-10.3e}\n'

        if len(self.y_list) > 0:
            text += 'Average (found)                          : '
            text += f'{np.mean(self.y_list):-10.3e}\n'

        if self.y_max is not None and self.y_max_real is not None:
            text += 'Maximum (found / real)                   : '
            text += f'{self.y_max:-10.3e}   / {self.y_max_real:-10.3e}\n'
        elif self.y_max is not None:
            text += 'Maximum (found)                          : '
            text += f'{self.y_max:-10.3e}\n'

        text += '=' * 78 + '\n'
        return text

    def log(self, postfix='', out=False):
        self.log_m_last = self.m
        t = tpc() - self.log_t

        text = ''

        if self.log_prefix:
            text += self.log_prefix + ' > '

        text += f'm {self.m:-7.1e}'
        if self.with_cache:
            text += f' [+ {self.m_cache:-7.1e}]'
        text += ' | '

        text += f't {t:-7.1e} | '

        if self.log_with_min:
            text += f'min {self.y_min:-10.3e} | '

        if self.log_with_max:
            text += f'max {self.y_max:-10.3e} | '

        if postfix:
            text = text + postfix

        if out:
            print(text)

        return text

    def prep(self):
        """A function with a specific benchmark preparation code."""
        # Note that when inherited, the function in the child class
        # must starts with the following line:
        self.check_err()

        # and should ends with the following two lines:
        self.is_prep = True
        return self

    def set_cache(self, with_cache=False, cache=None, m_max=1.E+8):
        self.with_cache = with_cache
        self.cache = {} if cache is None else cache
        self.cache_m_max = int(m_max) if m_max else None

    def set_constr(self, penalty=1.E+42, eps=1.E-16, with_amplitude=True):
        """Set constraint options."""
        self.constr_penalty = penalty
        self.constr_eps = eps
        self.constr_with_amplitude = with_amplitude

    def set_desc(self, desc=''):
        """Set text description of the problem."""
        self.desc = desc

    def set_err(self, err=''):
        """Set the error text (can not import external module, etc.)."""
        self.err.append(err)

    def set_grid(self, a=None, b=None):
        """Set grid lower (a) and upper (b) limits for the function-like BM."""
        self.a = teneva.grid_prep_opt(a, self.d)
        self.b = teneva.grid_prep_opt(b, self.d)

    def set_grid_kind(self, kind='cheb'):
        """Set the kind of the grid ('cheb' or 'uni').

        Note:
            In some benchmarks, when setting the exact global optimum, it is
            used that the central multi-index for a grid with an odd number of
            nodes lies at the origin. When new types of grids appear, this
            point should be taken into account.

        """
        self.grid_kind = kind

        if not self.grid_kind in ['uni', 'cheb']:
            msg = f'Invalid kind of the grid (should be "uni" or "cheb")'
            raise ValueError(msg)

    def set_log(self, with_log=False, cond='min-max', step=1000, prefix='bm',
                with_min=True, with_max=True):
        """Set the log options."""
        self.with_log = with_log
        self.log_cond = cond
        self.log_step = int(step) if step else None
        self.log_prefix = prefix
        self.log_with_min = with_min
        self.log_with_max = with_max

        self.log_t = tpc()

        if not self.log_cond in ['min', 'max', 'min-max', 'step']:
            raise ValueError('Invalid "log_cond" argument')

    def set_max(self, i=None, x=None, y=None):
        """Set exact (real) global maximum (index, point and related value)."""
        self.i_max_real = i
        self.x_max_real = x
        self.y_max_real = y

        if self.i_max_real is not None:
            self.i_max_real = np.asanyarray(self.i_max_real, dtype=int)
        if self.x_max_real is not None:
            self.x_max_real = np.asanyarray(self.x_max_real, dtype=float)
        if self.y_max_real is not None:
            self.y_max_real = float(self.y_max_real)

    def set_min(self, i=None, x=None, y=None):
        """Set exact (real) global minimum (index, point and related value)."""
        self.i_min_real = i
        self.x_min_real = x
        self.y_min_real = y

        if self.i_min_real is not None:
            self.i_min_real = np.asanyarray(self.i_min_real, dtype=int)
        if self.x_min_real is not None:
            self.x_min_real = np.asanyarray(self.x_min_real, dtype=float)
        if self.y_min_real is not None:
            self.y_min_real = float(self.y_min_real)

    def set_name(self, name=''):
        """Set display name for the problem."""
        self.name = name

    def set_opts(self):
        """Setting options specific to the benchmark."""
        return

    def set_quantization(self, with_quantization=False):
        self.with_quantization = with_quantization

        if not self.with_quantization:
            self.quantization = None
            return

        if not self.is_n_equal:
            msg = 'Quantization now works only if all mode sizes are equal'
            raise NotImplementedError(msg)

        n = self.n[0]
        self.quantization = int(np.log2(n))
        if 2**self.quantization != n:
            msg = 'Invalid mode size for quantization '
            msg += '(it should be a power of two)'
            raise ValueError(msg)

    def set_size(self, d=None, n=None):
        """Set dimension (d) and sizes for all d-modes (n: int or list)."""
        self.d = None if d is None else int(d)
        self.n = teneva.grid_prep_opt(n, self.d, int)

    def _c(self, x):
        """Function that check constraint for a given point/index."""
        return self._c_batch(np.array(x).reshape(1, -1))[0]

    def _c_batch(self, X):
        """Function that check constraint for a given batch of poi./indices."""
        return np.array([self._c(x) for x in X])

    def _compute(self, X):
        if not self.with_constr:
            return self._f_batch(X)

        y = np.ones(X.shape[0]) * self.constr_penalty
        c = self._c_batch(X)
        ind = c < self.constr_eps

        y[ind] = self._f_batch(X[ind])
        if self.constr_with_amplitude:
            y[~ind] *= c[~ind]

        return y

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

    def _init(self):
        self.err = []

        self.is_y_max_new = False
        self.i_max = None
        self.x_max = None
        self.y_max = None

        self.is_y_min_new = False
        self.i_min = None
        self.x_min = None
        self.y_min = None

        self.y_list = []

        self.m = 0
        self.m_cache = 0
        self.time = 0.

        self.log_m_last = 0

        self.is_prep = False
        self.with_cores = False

    def _log_check(self):
        if not self.with_log:
            return False

        if self.log_cond == 'min':
            return self.is_y_min_new

        if self.log_cond == 'max':
            return self.is_y_max_new

        if self.log_cond == 'min-max':
            return self.is_y_min_new or self.is_y_max_new

        if self.log_cond == 'step':
            return self.log_step and self.m - self.log_m_last > self.log_step

    def _parse_input(self, I=None, X=None):
        if I is not None and X is not None:
            raise ValueError('Invalid case')

        if I is None and X is None:
            raise ValueError('Invalid case')

        if X is not None and self.is_tens:
            msg = f'BM "{self.name}" is a tensor. '
            msg += 'Can`t compute it in the point'
            raise ValueError(msg)

        if I is not None:
            I = np.asanyarray(I, dtype=int)

            is_batch = len(I.shape) == 2
            if not is_batch:
                I = I.reshape(1, -1)

            if self.with_quantization:
                I = self._unquantize(I)

            if self.is_func:
                X = teneva.ind_to_poi(I, self.a, self.b, self.n, self.grid_kind)

        elif X is not None:
            X = np.asanyarray(X, dtype=float)

            is_batch = len(X.shape) == 2
            if not is_batch:
                X = X.reshape(1, -1)

            if self.with_quantization:
                X = self._unquantize(X)

            if self.is_func:
                I = teneva.poi_to_ind(X, self.a, self.b, self.n, self.grid_kind)

        return I, X, is_batch

    def _process(self, I, X, y, dm_cache, t, is_batch, skip=False):
        if skip:
            return y if is_batch else y[0]

        self.y_list.extend(list(y))

        self.m += y.shape[0] - dm_cache
        self.m_cache += dm_cache

        self.time += tpc() - t

        ind = np.argmax(y)
        if self.y_max is None or self.y_max < y[ind]:
            self.is_y_max_new = True
            self.i_max = I[ind, :] if I is not None else None
            self.x_max = X[ind, :] if X is not None else None
            self.y_max = y[ind]
        else:
            self.is_y_max_new = False


        ind = np.argmin(y)
        if self.y_min is None or self.y_min > y[ind]:
            self.is_y_min_new = True
            self.i_min = I[ind, :] if I is not None else None
            self.x_min = X[ind, :] if X is not None else None
            self.y_min = y[ind]
        else:
            self.is_y_min_new = False

        if self._log_check():
            print(self.log())

        if self.cache_m_max and len(self.cache.keys()) > self.cache_m_max:
            self._wrn('The maximum cache size has been exceeded. Cache cleared')
            self.cache = {}

        return y if is_batch else y[0]

    def _unquantize(self, I_qtt):
        if len(I_qtt.shape) == 1:
            is_many = False
            I_qtt = I_qtt.reshape(1, -1)
        else:
            is_many = True

        d = self.d
        q = self.quantization
        n = [2] * q
        m = I_qtt.shape[0]

        I = np.zeros((m, d), dtype=I_qtt.dtype)
        for k in range(d):
            I_qtt_curr = I_qtt[:, q*k:q*(k+1)].T
            I[:, k] = np.ravel_multi_index(I_qtt_curr, n, order='F')

        return I if is_many else I[0, :]

    def _wrn(self, text):
        text = '!!! BM-WARNING | ' + text
        print(text)
