import numpy as np
import teneva
from time import perf_counter as tpc


from teneva_bm import __version__


class Bm:
    def __init__(self, d=None, n=None, name='', desc=''):
        self._init()

        self.set_seed()

        self.set_size(d, n)
        self.set_quantization()
        self.set_constr()

        self.set_grid()
        self.set_grid_kind()

        self.set_name(name)
        self.set_desc(desc)

        self.set_opts()
        self.set_cache()
        self.set_budget()

        self.set_min()
        self.set_max()

        self.set_log()

    def __call__(self, X):
        """Return a value or batch of values for provided x-point."""
        return self.get_poi(X)

    def __getitem__(self, I):
        """Return a value or batch of values for provided multi-index."""
        return self.get(I)

    @property
    def a0(self):
        """Return the lower grid size value if it is constant."""
        if not self.is_a_equal:
            raise ValueError('Lower grid size is not constant, can`t get a0')
        return self.a[0]

    @property
    def b0(self):
        """Return the upper grid size value if it is constant."""
        if not self.is_b_equal:
            raise ValueError('Upper grid size is not constant, can`t get b0')
        return self.b[0]

    @property
    def is_a_equal(self):
        """Check if all the lower grid sizes are the same."""
        v = self.list_convert(self.a, 'float')
        return v is None or isinstance(v, (float,))

    @property
    def is_b_equal(self):
        """Check if all the upper grid sizes are the same."""
        v = self.list_convert(self.b, 'float')
        return v is None or isinstance(v, (float,))

    @property
    def is_func(self):
        """Check if BM relates to function (i.e., continuous function)."""
        return not self.is_tens

    @property
    def is_n_equal(self):
        """Check if all the mode sizes are the same."""
        v = self.list_convert(self.n, 'int')
        return v is None or isinstance(v, (int,))

    @property
    def is_n_even(self):
        """Check if all the mode sizes are even (2, 4, ...)."""
        if self.n is None:
            return True
        for k in self.n:
            if k % 2 == 1:
                return False
        return True

    @property
    def is_n_odd(self):
        """Check if all the mode sizes are odd (1, 3, ...)."""
        return not self.is_n_even

    @property
    def is_tens(self):
        """Check if BM relates to tensor (i.e., discrete function)."""
        return not self.is_func

    @property
    def n0(self):
        """Return the mode size value if it is constant."""
        if not self.is_n_equal:
            raise ValueError('Mode size is not constant, can`t get n0')
        return self.n[0]

    @property
    def with_constr(self):
        """Return True if benchmark has a constraint."""
        return False

    @property
    def with_cores(self):
        """Return True if exact TT-cores can be constructed for benchmark."""
        return False

    def build_cores(self):
        """Return exact TT-cores for the TT-representation of the tensor."""
        if self.is_tens:
            # TODO: check why
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

        # TODO: add fixed random seed support
        I_trn = teneva.sample_lhs(n, m)
        y_trn = self.get(I_trn, skip_process)

        if y_trn is None:
            raise ValueError('The specified budget is exceeded')

        return I_trn, y_trn

    def build_tst(self, m=0, skip_process=True):
        """Generate random (from "choice") test dataset of (index, value)."""
        m = int(m)
        n = [2]*self.quantization*self.d if self.with_quantization else self.n

        if m < 1:
            return None, None

        # TODO: add fixed random seed support
        I_tst = np.vstack([np.random.choice(k, m) for k in n]).T
        y_tst = self.get(I_tst, skip_process)

        if y_tst is None:
            raise ValueError('The specified budget is exceeded')

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
            dm_cache = m - m_new

            if self.budget_m_cache:
                if self.budget_is_strict:
                    if self.m_cache + dm_cache > self.budget_m_cache:
                        return None
                else:
                    if self.m_cache > self.budget_m_cache:
                        return None

            if m_new > 0:
                Z = X[ind] if self.is_func else I[ind]
                y_new = self._compute(Z, skip_process)
                if y_new is None:
                    return None

                for k in range(m_new):
                    self.cache[tuple(I[ind[k]])] = y_new[k]

            y = np.array([self.cache[tuple(i)] for i in I])

        else:
            dm_cache = 0

            Z = X if self.is_func else I
            y = self._compute(Z, skip_process)
            if y is None:
                return None

        return self._process(I, X, y, dm_cache, t, is_batch, skip_process)

    def get_config(self):
        """Return a dict with configuration of the benchmark."""
        conf = {
            'd': self.d,
            'n': self.list_convert(self.n, 'int'),
            'seed': self.seed,
            'name': self.name,
            'benchmark': self.__class__.__name__,
            'version': __version__,
            'is_tens': self.is_tens,
            'is_func': self.is_func,
            'with_quantization': self.with_quantization,
            'with_cache': self.with_cache,
            'with_constr': self.with_constr,
            'with_cores': self.with_cores,
        }

        if self.is_func:
            conf['a'] = self.list_convert(self.a, 'float')
            conf['b'] = self.list_convert(self.b, 'float')
            conf['grid_kind'] = self.grid_kind

        if self.with_constr:
            conf['constr_penalty'] = self.constr_penalty
            conf['constr_eps'] = self.constr_eps
            conf['constr_with_amplitude'] = self.constr_with_amplitude

        if self.budget_m:
            conf['budget_m'] = self.budget_m
        if self.budget_m_cache:
            conf['budget_m_cache'] = self.budget_m_cache
        if self.budget_m or self.budget_m_cache:
            conf['budget_is_strict'] = self.budget_is_strict

        if self.i_max_real is not None:
            conf['i_max_real'] = self.list_convert(self.i_max_real, 'int')
        if self.x_max_real is not None:
            conf['x_max_real'] = self.list_convert(self.x_max_real, 'float')
        if self.y_max_real is not None:
            conf['y_max_real'] = self.y_max_real

        if self.i_min_real is not None:
            conf['i_min_real'] = self.list_convert(self.i_min_real, 'int')
        if self.x_min_real is not None:
            conf['x_min_real'] = self.list_convert(self.x_min_real, 'float')
        if self.y_min_real is not None:
            conf['y_min_real'] = self.y_min_real

        return conf

    def get_history(self):
        """Return a dict with results of requests to the benchmark."""
        hist = {
            'm': self.m,
            'm_cache': self.m_cache,
            'i_max': self.i_max,
            'x_max': self.x_max,
            'y_max': self.y_max,
            'i_min': self.i_min,
            'x_min': self.x_min,
            'y_min': self.y_min,
            'y_list': self.y_list,
            'time': self.time,
            'time_full': tpc() - self.log_t,
            'err': '; '.join(self.err) if len(self.err) else '',
        }

        return hist

    def get_poi(self, X, skip_process=False):
        """Return a value or batch of values for provided x-point."""
        t = tpc()

        self.check()

        I, X, is_batch = self._parse_input(X=X)

        y = self._compute(X, skip_process)
        if y is None:
            return None

        return self._process(I, X, y, 0, t, is_batch, skip_process)

    def info(self, footer=''):
        """Returns a detailed description of the benchmark as text."""
        text = '-' * 78 + '\n' + 'BM: '
        text += self.name + ' ' * max(0, 36-len(self.name)) +  ' | '
        text += f'DIMS = {self.d:-4d} | '
        n = np.mean(self.n)
        text += '<MODE SIZE> = ' + (f'{n:-7.1f}' if n<9999 else f'{n:-7.1e}')
        text += '\n'
        text += '-' * 41 + '|             '
        text += '>           Description'
        text += '\n'

        text += '.' * 78 + '\n'
        desc = f'    {self.desc.strip()}'
        text += desc.replace('            ', '    ')
        text += '\n'
        text += '.' * 78 + '\n'

        text += '-' * 41 + '|            '
        text += '>          Configuration'
        text += '\n'

        text += 'Package version                          : '
        v = __version__
        text += f'{v}\n'

        text += 'Random seed                              : '
        v = self.seed
        text += f'{v}\n'

        text += 'Benchmark                                : '
        v = self.__class__.__name__
        text += f'{v}\n'

        text += 'Dimension                                : '
        v = self.d
        text += f'{v}\n'

        text += 'Mode size                                : '
        v = self.list_convert(self.n, 'int')
        text += f'{v}\n'

        if self.is_func:
            text += 'Lower grid limit                         : '
            va = self.list_convert(self.a, 'float')
            if not isinstance(va, (int, float)) and self.d > 3:
                va = f'[{va[0]:.2f}, {va[1]:.2f}, <...>, {va[-1]:.2f}]'
            text += f'{va}\n'

            text += 'Upper grid limit                         : '
            vb = self.list_convert(self.b, 'float')
            if not isinstance(vb, (int, float)) and self.d > 3:
                vb = f'[{vb[0]:.2f}, {vb[1]:.2f}, <...>, {vb[-1]:.2f}]'
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                if va < 0 and vb > 0:
                    vb = f'+{vb}'
            text += f'{vb}\n'

            text += 'Grid kind                                : '
            v = self.grid_kind
            text += f'{v}\n'

        text += 'Function kind                            : '
        v = 'discrete' if self.is_tens else 'continuous'
        text += f'{v}\n'

        text += 'With quantization                        : '
        v = 'YES' if self.with_quantization else 'no'
        text += f'{v}\n'

        text += 'With cache                               : '
        v = 'YES' if self.with_cache else 'no'
        text += f'{v}\n'

        text += 'With constraint                          : '
        v = 'YES' if self.with_constr else 'no'
        text += f'{v}\n'

        text += 'With TT-cores                            : '
        v = 'YES' if self.with_cores else 'no'
        text += f'{v}\n'

        if self.with_constr:
            text += 'Constraint penalty                       : '
            v = self.constr_penalty
            text += f'{v}\n'

            text += 'Constraint epsilon                       : '
            v = self.constr_eps
            text += f'{v}\n'

            text += 'Constraint with amplitude                : '
            v = 'YES' if self.constr_with_amplitude else 'no'
            text += f'{v}\n'

        if self.budget_m:
            text += 'Computation budget                       : '
            v = self.budget_m
            text += f'{v}\n'

        if self.budget_m_cache:
            text += 'Computation budget for cache requests    : '
            v = self.budget_m_cache
            text += f'{v}\n'

        if self.budget_m or self.budget_m_cache:
            text += 'Computation budget is strict             : '
            v = 'YES' if self.budget_is_strict else 'no'
            text += f'{v}\n'

        if self.i_max_real is not None:
            text += 'Exact max (multi-index)                  : '
            v = self.list_convert(self.i_max_real, 'int')
            if not isinstance(v, (int, float)) and self.d > 3:
                v = f'[{v[0]}, {v[1]}, <...>, {v[-1]}]'
            text += f'{v}\n'

        if self.x_max_real is not None:
            text += 'Exact max (point)                        : '
            v = self.list_convert(self.x_max_real, 'float')
            if not isinstance(v, (int, float)) and self.d > 3:
                v = f'[{v[0]:.2f}, {v[1]:.2f}, <...>, {v[-1]:.2f}]'
            text += f'{v}\n'

        if self.y_max_real is not None:
            text += 'Exact max (value)                        : '
            v = self.y_max_real
            text += f'{v}\n'

        if self.i_min_real is not None:
            text += 'Exact min (multi-index)                  : '
            v = self.list_convert(self.i_min_real, 'int')
            if not isinstance(v, (int, float)) and self.d > 3:
                v = f'[{v[0]}, {v[1]}, <...>, {v[-1]}]'
            text += f'{v}\n'

        if self.x_min_real is not None:
            text += 'Exact min (point)                        : '
            v = self.list_convert(self.x_min_real, 'float')
            if not isinstance(v, (int, float)) and self.d > 3:
                v = f'[{v[0]:.2f}, {v[1]:.2f}, <...>, {v[-1]:.2f}]'
            text += f'{v}\n'

        if self.y_min_real is not None:
            text += 'Exact min (value)                        : '
            v = self.y_min_real
            text += f'{v}\n'

        if footer:
            text += '-' * 41 + '|             '
            text += '>               Options'
            text += '\n'
            text += footer

        text += '=' * 78 + '\n'
        return text

    def info_history(self):
        """Returns an information about the request history (text)."""
        text = ''

        text = '-' * 78 + '\n' + 'BM: '
        text += self.name + ' ' * max(0, 36-len(self.name)) +  ' | '
        text += f'DIMS = {self.d:-4d} | '
        n = np.mean(self.n)
        text += '<MODE SIZE> = ' + (f'{n:-7.1f}' if n<9999 else f'{n:-7.1e}')
        text += '\n'
        text += '-' * 41 + '|             '
        text += '>   History of requests'
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

        text += 'Total requests time (sec)                : '
        text += f'{self.time:-10.3e}\n'

        text += 'Total work time (sec)                    : '
        text += f'{tpc() - self.log_t:-10.3e}\n'

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

    def list_convert(self, x, kind='float', eps=1.E-16):
        """Convert list of equal values to one number and back."""
        if x is None:
            return None
        if kind == 'int':
            if isinstance(x, (int, float)):
                return np.array([x]*self.d, dtype=int)
            return int(x[0]) if len(set(x))==1 else np.asanyarray(x, dtype=int)
        elif kind == 'float':
            if isinstance(x, (int, float)):
                return np.array([x]*self.d, dtype=float)
            for v in x:
                if np.abs(v - x[0]) > eps:
                    return np.asanyarray(x, dtype=float)
            return float(x[0])
        else:
            raise ValueError('Unsupported kind for list conversion')

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

        if self.log_with_min and self.y_min is not None:
            text += f'min {self.y_min:-10.3e} | '

        if self.log_with_max and self.y_max is not None:
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

        # and it should ends with the following two lines:
        self.is_prep = True
        return self

    def set_budget(self, m=None, m_cache=None, is_strict=True):
        """Set computation buget."""
        self.budget_m = int(m) if m else None
        self.budget_m_cache = int(m_cache) if m_cache else None
        self.budget_is_strict = is_strict

    def set_cache(self, with_cache=False, cache=None, m_max=1.E+8):
        """Set cache options."""
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
            nodes lies at the origin. When new types of grids appear, it should
            be taken into account.

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
            raise ValueError(f'Invalid "log_cond" argument "{self.log_cond}"')

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
        """Set options specific to the benchmark."""
        return

    def set_quantization(self, with_quantization=False):
        """Set quantization usage."""
        self.with_quantization = with_quantization

        if not self.with_quantization:
            self.quantization = None
            return

        if not self.is_n_equal:
            msg = 'Quantization now works only if all mode sizes are equal'
            raise NotImplementedError(msg)

        n = self.n0
        self.quantization = int(np.log2(n))
        if 2**self.quantization != n:
            msg = 'Invalid mode size for quantization '
            msg += '(it should be a power of two)'
            raise ValueError(msg)

    def set_seed(self, seed=42):
        self.seed = seed
        self.rand = np.random.default_rng(self.seed)

    def set_size(self, d=None, n=None):
        """Set dimension (d) and sizes for all d-modes (n: int or list)."""
        self.d = None if d is None else int(d)
        self.n = teneva.grid_prep_opt(n, self.d, int)

    def shift_grid(self, scale=25):
        """Apply random shift for the grid limits."""
        shift = self.rand.normal(size=self.d) / scale
        self.a = self.a - (self.b-self.a) * shift
        self.b = self.b + (self.b-self.a) * shift

    def _c(self, x):
        """Function that check constraint for a given point/index."""
        return self._c_batch(x.reshape(1, -1))[0]

    def _c_batch(self, X):
        """Function that check constraint for a given batch of poi./indices."""
        return np.array([self._c(x) for x in X])

    def _compute(self, X, skip_process=False):
        m = self.m
        m_cur = X.shape[0]
        m_max = self.budget_m

        if not skip_process and m_max:
            if (m >= m_max) or (m + m_cur > m_max and self.budget_is_strict):
                return None

        if not self.with_constr:
            return self._f_batch(X)

        y = np.ones(X.shape[0]) * self.constr_penalty

        c = self._c_batch(X)
        ind_good = c < self.constr_eps

        y[ind_good] = self._f_batch(X[ind_good])
        if self.constr_with_amplitude:
            y[~ind_good] *= c[~ind_good]

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
        return self._f_batch(x.reshape(1, -1))[0]

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
            return self.log_step and self.m - self.log_m_last >= self.log_step

    def _parse_input(self, I=None, X=None):
        if I is not None and X is not None:
            raise ValueError('Can`t parse input. Invalid case')

        if I is None and X is None:
            raise ValueError('Can`t parse input. Invalid case')

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
