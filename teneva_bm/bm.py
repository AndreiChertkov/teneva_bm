import numpy as np
import os
import teneva
from time import perf_counter as tpc


from teneva_bm import __version__


class Bm:
    def __init__(self, d=None, n=None, name='', desc=''):
        self.init()

        self.set_log()

        self.set_seed()

        self.set_dimension(d)
        self.set_size(n)
        self.set_constr()

        self.set_grid()
        self.set_grid_kind()

        self.set_name(name)
        self.set_desc(desc)

        self.set_opts()
        self.set_cache()
        self.set_budget()

        self.set_max()
        self.set_min()

    def __call__(self, X):
        """Return a value or batch of values for provided x-point."""
        return self.get_poi(X)

    def __getitem__(self, I):
        """Return a value or batch of values for provided multi-index."""
        return self.get(I)

    @property
    def a0(self):
        """Return the lower grid size float value if it is constant."""
        if not self.is_a_equal:
            raise ValueError('Lower grid size is not constant, can`t get a0')
        return self.a[0] if self.a is not None else None

    @property
    def b0(self):
        """Return the upper grid size float value if it is constant."""
        if not self.is_b_equal:
            raise ValueError('Upper grid size is not constant, can`t get b0')
        return self.b[0] if self.b is not None else None

    @property
    def identity(self):
        """Returns a list of parameter names that define the benchmark."""
        return ['d', 'n']

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
    def is_opti_max(self):
        """If the benchmark relates to maximization task."""
        return False

    @property
    def is_tens(self):
        """Check if BM relates to tensor (i.e., discrete function)."""
        return not self.is_func

    @property
    def n0(self):
        """Return the mode size int value if it is constant."""
        if not self.is_n_equal:
            raise ValueError('Mode size is not constant, can`t get n0')
        return self.n[0] if self.n is not None else None

    @property
    def time_full(self):
        """Full time of benchmark existence in seconds."""
        return tpc() - self.time_stamp_start

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
            msg = 'Construction of the TT-cores does not work for tensors'
            raise ValueError(msg)

        if not self.with_cores:
            msg = 'Construction of the TT-cores does not supported for this BM'
            raise ValueError(msg)

        I = np.array([teneva.grid_flat(k) for k in self.n], dtype=int).T
        X = teneva.ind_to_poi(I, self.a, self.b, self.n, self.grid_kind)

        return self.cores(X)

    def build_trn(self, m=0, skip_process=False):
        """Generate random (from LHS) train dataset of (index, value)."""
        if m < 1:
            return None, None

        I_trn = teneva.sample_lhs(self.n, m, seed=self.rand)
        y_trn = self.get(I_trn, skip_process)

        if y_trn is None:
            raise ValueError('The specified budget is exceeded')

        return I_trn, y_trn

    def build_tst(self, m=0, skip_process=True):
        """Generate random (from "choice") test dataset of (index, value)."""
        if m < 1:
            return None, None

        I_tst = teneva.sample_random(self.n, m, seed=self.rand)
        y_tst = self.get(I_tst, skip_process)

        if y_tst is None:
            raise ValueError('The specified budget is exceeded')

        return I_tst, y_tst

    def check(self):
        """Check that benchmark's configuration is valid."""
        if not self.is_prep:
            msg = 'Run "prep" method for BM before call it'
            self.set_err(msg)

        if self.d is None:
            msg = 'Dimension "d" should be set'
            self.set_err(msg)

        if self.n is None:
            msg = 'Mode size "n" should be set'
            self.set_err(msg)

        if self.is_func and self.a is None:
            msg = 'Lower grid limit "a" should be set for continuous function'
            self.set_err(msg)

        if self.is_func and self.b is None:
            msg = 'Lower grid limit "b" should be set for continuous function'
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

    def compute(self, X, skip_process=False):
        m = self.m
        m_cur = X.shape[0]
        m_max = self.budget_m

        if not skip_process and m_max:
            if (m >= m_max) or (m + m_cur > m_max and self.budget_is_strict):
                return None

        if not self.with_constr:
            return self.target_batch(X)

        y = np.ones(m_cur, dtype=float) * self.constr_penalty

        c = self.constr_batch(X)
        ind_good = c < self.constr_eps
        y[ind_good] = self.target_batch(X[ind_good])
        if self.constr_with_amplitude:
            y[~ind_good] *= c[~ind_good]

        return y

    def constr(self, x):
        """Function that check constraint for a given point/index."""
        return self.constr_batch(x.reshape(1, -1))[0]

    def constr_batch(self, X):
        """Function that check constraint for a given batch of poi./indices."""
        return np.array([self.constr(x) for x in X])

    def cores(self, X):
        """Return the exact TT-cores for the provided points."""
        raise NotImplementedError()

    def cores_add(self, X, a0=0):
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

    def cores_mul(self, X):
        """Helper function for the construction of the TT-cores."""
        return [x[None, :, None] for x in X]

    def get(self, I, skip_process=False, skip_cache=False):
        """Return a value or batch of values for provided multi-index."""
        t = tpc()

        self.check()

        I, X, is_batch = self.parse_input(I=I)

        if self.with_cache and not skip_cache:
            m = I.shape[0]
            ind = [k for k in range(m) if tuple(I[k]) not in self.cache]

            m_new = len(ind)
            dm_cache = m - m_new

            if self.budget_m_cache:
                if self.budget_is_strict:
                    if self.m_cache + dm_cache > self.budget_m_cache:
                        return self.process_last()
                else:
                    if self.m_cache > self.budget_m_cache:
                        return self.process_last()

            if m_new > 0:
                Z = X[ind] if self.is_func else I[ind]
                y_new = self.compute(Z, skip_process)
                if y_new is None:
                    return self.process_last()

                for k in range(m_new):
                    self.cache[tuple(I[ind[k]])] = y_new[k]

            y = np.array([self.cache[tuple(i)] for i in I])

        else:
            dm_cache = 0

            Z = X if self.is_func else I
            y = self.compute(Z, skip_process)
            if y is None:
                return self.process_last()

        return self.process(I, X, y, dm_cache, t, is_batch, skip_process)

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
            'i_max': self.list_copy(self.i_max, 'int'),
            'x_max': self.list_copy(self.x_max, 'float'),
            'y_max': self.y_max,
            'i_min': self.list_copy(self.i_min, 'int'),
            'x_min': self.list_copy(self.x_min, 'float'),
            'y_min': self.y_min,
            'y_list': self.list_copy(self.y_list, 'float'),
            'time': self.time,
            'time_full': self.time_full,
            'err': '; '.join(self.err) if len(self.err) else '',
        }

        return hist

    def get_solution(self, i=None, best=True):
        """Return the solution for given i or current solution or the best."""
        if i is None:
            if best:
                i = self.i_max if self.is_opti_max else self.i_min
            else:
                i = self.i

        if i is None:
            raise ValueError('Input is not set')

        y = self.get(i, skip_process=True, skip_cache=True)

        return i, y

    def get_poi(self, X, skip_process=False):
        """Return a value or batch of values for provided x-point."""
        t = tpc()

        self.check()

        I, X, is_batch = self.parse_input(X=X)

        y = self.compute(X, skip_process)
        if y is None:
            return self.process_last()

        return self.process(I, X, y, 0, t, is_batch, skip_process)

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

    def info_current(self, footer=''):
        text = ''

        if self.log_prefix:
            text += self.log_prefix + ' > '

        text += f'm {self.m:-7.1e}'
        if self.with_cache:
            text += f' [+ {self.m_cache:-7.1e}]'
        text += ' | '

        text += f't {self.time_full:-7.1e} | '

        if self.log_with_min and self.y_min is not None:
            text += f'min {self.y_min:-10.3e} | '

        if self.log_with_max and self.y_max is not None:
            text += f'max {self.y_max:-10.3e} | '

        if footer:
            text = text + footer

        self.log_m_last = self.m

        return text

    def info_history(self, footer=''):
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
        text += f'{self.time_full:-10.3e}\n'

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

        if footer:
            text += '-' * 41 + '|             '
            text += '>               Options'
            text += '\n'
            text += footer

        text += '=' * 78 + '\n'
        return text

    def init(self):
        self.err = []

        self.i = None
        self.x = None
        self.y = None

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

        self.time_stamp_start = tpc()

        self.cache = {}

        self.is_prep = False

    def list_convert(self, x, kind='float', eps=1.E-16):
        """Convert list of equal values to one number and back."""
        if x is None:
            return None

        if kind == 'int':
            if isinstance(x, (int, float)):
                return np.array([x]*self.d, dtype=int)

            if len(set(x)) == 1:
                return int(x[0])
            else:
                return self.list_copy(x, 'int')

        elif kind == 'float':
            if isinstance(x, (int, float)):
                return np.array([x]*self.d, dtype=float)
            for v in x:
                if np.abs(v - x[0]) > eps:
                    return self.list_copy(x, 'float')
            return float(x[0])

        else:
            raise ValueError('Unsupported kind for list conversion')

    def list_copy(self, x, kind=None):
        """Copy list or array and return the new array."""
        if x is None:
            return None

        if kind:
            if kind == 'int':
                x = np.asanyarray(x, dtype=int)
            elif kind == 'float':
                x = np.asanyarray(x, dtype=float)
            else:
                raise ValueError('Unsupported kind for list copy')
        else:
            x = np.asanyarray(x)

        return x.copy()

    def log_check(self):
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

    def parse_input(self, I=None, X=None):
        if I is None and X is None or I is not None and X is not None:
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

            if self.is_func:
                X = teneva.ind_to_poi(I, self.a, self.b, self.n, self.grid_kind)

        elif X is not None:
            X = np.asanyarray(X, dtype=float)

            is_batch = len(X.shape) == 2
            if not is_batch:
                X = X.reshape(1, -1)

            if self.is_func:
                I = teneva.poi_to_ind(X, self.a, self.b, self.n, self.grid_kind)

        return I, X, is_batch

    def path_build(self, fpath=None, ext=None):
        if not fpath:
            return

        fold = os.path.dirname(fpath)
        if fold:
            os.makedirs(fold, exist_ok=True)

        if ext and not fpath.endswith('.' + ext):
            fpath += '.' + ext

        return fpath

    def prep(self):
        """A function with a specific benchmark preparation code."""
        self.check_err()
        self.prep_bm()
        self.is_prep = True
        return self

    def prep_bm(self):
        """A function with a specific benchmark preparation code (inner)."""
        return

    def process(self, I, X, y, dm_cache, t, is_batch, skip=False):
        if skip:
            return y if is_batch else y[0]

        self.y_list.extend(list(y))

        self.m += y.shape[0] - dm_cache
        self.m_cache += dm_cache

        self.time += tpc() - t

        self.i = I[-1, :].copy() if I is not None else None
        self.x = X[-1, :].copy() if X is not None else None
        self.y = y[-1]

        ind = np.argmax(y)
        if self.y_max is None or self.y_max < y[ind]:
            self.is_y_max_new = True
            self.i_max = I[ind, :].copy() if I is not None else None
            self.x_max = X[ind, :].copy() if X is not None else None
            self.y_max = y[ind]
        else:
            self.is_y_max_new = False


        ind = np.argmin(y)
        if self.y_min is None or self.y_min > y[ind]:
            self.is_y_min_new = True
            self.i_min = I[ind, :].copy() if I is not None else None
            self.x_min = X[ind, :].copy() if X is not None else None
            self.y_min = y[ind]
        else:
            self.is_y_min_new = False

        if self.log_check():
            self.log(self.info_current())

        if self.cache_m_max and len(self.cache.keys()) > self.cache_m_max:
            self.cache = {}
            self.wrn('The maximum cache size has been exceeded. Cache cleared')

        return y if is_batch else y[0]

    def process_last(self):
        if self.with_log:
            self.log(self.info_current('<<< DONE\n'))

    def recover(self, i=None, best=True):
        """Restores some benchmark-specific values."""
        raise NotImplementedError
        i, y = self.get_solution(i, best)

    def render(self, fpath=None, i=None, best=True):
        """Render the solution for benchmark."""
        raise NotImplementedError
        i, y = self.get_solution(i, best)
        fpath = self.path_build(fpath)

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
        """Set text description of the benchmark."""
        self.desc = desc

    def set_dimension(self, d=None):
        """Set dimension (d)."""
        self.d = None if d is None else int(d)

        if getattr(self, 'n', None) is not None:
            raise ValueError('Dimension shanged but "n" already set')
        if getattr(self, 'a', None) is not None:
            raise ValueError('Dimension shanged but "a" already set')
        if getattr(self, 'b', None) is not None:
            raise ValueError('Dimension shanged but "b" already set')

    def set_err(self, err=''):
        """Set the error text (can not import external module, etc.)."""
        self.err.append(err)

    def set_grid(self, a=None, b=None):
        """Set grid lower (a) and upper (b) limits for the function-like BM."""
        if (a is not None or b is not None) and not self.d:
            raise ValueError('Please, set dimension "d" before')

        self.a = teneva.grid_prep_opt(a, self.d)
        self.b = teneva.grid_prep_opt(b, self.d)

        if self.a is not None and self.b is not None:
            for k in range(self.d):
                if self.a[k] >= self.b[k]:
                    raise ValueError('Invalid grid limits (a >= b)')

    def set_grid_kind(self, kind='cheb'):
        """Set the kind of the grid ('cheb' or 'uni').

        Note:
            In some benchmarks, when setting the exact global optimum, it is
            used that the central multi-index for a grid with an odd number of
            nodes lies at the origin. When new types of grids appear, it should
            be taken into account.

        """
        if not kind in ['uni', 'cheb']:
            msg = f'Invalid kind of the grid (should be "uni" or "cheb")'
            raise ValueError(msg)

        self.grid_kind = kind

    def set_log(self, log=False, cond='min-max', step=1000, prefix='bm',
                with_min=True, with_max=True, log_wrn=None):
        """Set the log options. The "log" may be bool or print-like function."""
        if log:
            self.with_log = True
            self.log = print if isinstance(log, bool) else log
        else:
            self.with_log = False
            self.log = lambda text: None

        if log_wrn:
            self.log_wrn = log_wrn
        else:
            self.log_wrn = print

        if not cond in ['min', 'max', 'min-max', 'step']:
            raise ValueError(f'Invalid "log_cond" argument "{cond}"')

        self.log_cond = cond
        self.log_step = int(step) if step else None
        self.log_prefix = prefix
        self.log_with_min = with_min
        self.log_with_max = with_max

    def set_max(self, i=None, x=None, y=None):
        """Set exact (real) global maximum (index, point and related value)."""
        self.i_max_real = i
        self.x_max_real = x
        self.y_max_real = y

        if self.i_max_real is not None:
            self.i_max_real = self.list_copy(self.i_max_real, 'int')
        if self.x_max_real is not None:
            self.x_max_real = self.list_copy(self.x_max_real, 'float')
        if self.y_max_real is not None:
            self.y_max_real = float(self.y_max_real)

        if self.i_max_real is not None:
            if getattr(self, 'n', None) is None:
                raise ValueError('Please set mode sizes before the max')

            for k in range(self.d):
                is_out_a = self.i_max_real[k] < 0
                is_out_b = self.i_max_real[k] > self.n[k] - 1
                if is_out_a or is_out_b:
                    raise ValueError('The i_max is out of grid bounds')

        if self.x_max_real is not None:
            if not self.is_func:
                raise ValueError('Can not set x_max for discrete function')
            if getattr(self, 'a', None) is None:
                raise ValueError('Please set lower grid limit before the max')
            if getattr(self, 'b', None) is None:
                raise ValueError('Please set upper grid limit before the max')

            for k in range(self.d):
                is_out_a = self.x_max_real[k] < self.a[k]
                is_out_b = self.x_max_real[k] > self.b[k]
                if is_out_a or is_out_b:
                    raise ValueError('The x_max is out of grid bounds')

    def set_min(self, i=None, x=None, y=None):
        """Set exact (real) global minimum (index, point and related value)."""
        self.i_min_real = i
        self.x_min_real = x
        self.y_min_real = y

        if self.i_min_real is not None:
            self.i_min_real = self.list_copy(self.i_min_real, 'int')
        if self.x_min_real is not None:
            self.x_min_real = self.list_copy(self.x_min_real, 'float')
        if self.y_min_real is not None:
            self.y_min_real = float(self.y_min_real)

        if self.i_min_real is not None:
            if getattr(self, 'n', None) is None:
                raise ValueError('Please set mode sizes before the min')

            for k in range(self.d):
                is_out_a = self.i_min_real[k] < 0
                is_out_b = self.i_min_real[k] > self.n[k] - 1
                if is_out_a or is_out_b:
                    raise ValueError('The i_min is out of grid bounds')

        if self.x_min_real is not None:
            if not self.is_func:
                raise ValueError('Can not set x_min for discrete function')
            if getattr(self, 'a', None) is None:
                raise ValueError('Please set lower grid limit before the min')
            if getattr(self, 'b', None) is None:
                raise ValueError('Please set upper grid limit before the min')

            for k in range(self.d):
                is_out_a = self.x_min_real[k] < self.a[k]
                is_out_b = self.x_min_real[k] > self.b[k]
                if is_out_a or is_out_b:
                    raise ValueError('The x_min is out of grid bounds')

    def set_name(self, name=''):
        """Set display name for the benchmark."""
        self.name = name

    def set_opts(self):
        """Set options specific to the benchmark."""
        return

    def set_seed(self, seed=42):
        """Set random seed and inner generator of random numbers."""
        self.seed = seed
        self.rand = np.random.default_rng(self.seed)

    def set_size(self, n=None):
        """Set sizes for all d-modes (n should be int or list)."""
        if n is not None and not self.d:
            raise ValueError('Please, set dimension "d" before')

        self.n = teneva.grid_prep_opt(n, self.d, int)

    def shift_grid(self, scale=100., sign=1):
        """Apply random shift for the grid limits."""
        if self.a is None or self.b is None:
            raise ValueError('Please, set grid before')

        rand = np.random.default_rng(42)

        a_shift = rand.uniform(0, (self.b-self.a) / scale, size=self.d)
        b_shift = rand.uniform(0, (self.b-self.a) / scale, size=self.d)

        self.a = self.a + a_shift * sign
        self.b = self.b - b_shift * sign

    def show(self, fpath=None, i=None, best=True):
        """Present the state of the benchmark (image, graph, etc.)."""
        raise NotImplementedError
        i, y = self.get_solution(i, best)
        fpath = self.path_build(fpath)

    def target(self, x):
        """Function that computes value for a given point/index."""
        return self.target_batch(x.reshape(1, -1))[0]

    def target_batch(self, X):
        """Function that computes values for a given batch of points/indices."""
        return np.array([self.target(x) for x in X])

    def wrn(self, text):
        self.log_wrn('!!! BM-WARNING | ' + text)
