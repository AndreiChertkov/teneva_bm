import numpy as np
import os
import teneva
from teneva_bm import __version__
from time import perf_counter as tpc


class Bm:
    def __init__(self, d=None, n=None, seed=42, name=None):
        self.is_prep = False
        self.set_dimension(d)
        self.set_size(n)
        self.seed = seed
        self.rand = np.random.default_rng(self.seed)
        self.set_name(name or self.name_class[2:])
        self.set_desc('benchmark_description')
        self.set_opts_dflt()
        self.set_grid()
        self.set_grid_kind()
        self.set_constr()
        self.set_cache()
        self.set_budget()
        self.set_max()
        self.set_min()
        self.set_log()
        self.init()

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
    def args(self):
        """Dict with values of benchmark's arguments (i.e., main params)."""
        return self.build_dict(self.args_info)

    @property
    def args_constr(self):
        """Dict with constraints on the benchmark's arguments."""
        # Note that constraints should be number, list (enum) or "equal".
        return {}

    @property
    def args_info(self):
        """Dict with info about benchmark's arguments."""
        return {
            'd': {
                'desc': 'Dimension',
                'kind': 'int'
            },
            'n': {
                'desc': 'Mode size',
                'kind': 'int',
                'list': True
            },
            'seed': {
                'desc': 'Random seed',
                'kind': 'int'
            },
            'name': {
                'desc': 'Benchmark name',
                'kind': 'str'
            }
        }

    @property
    def b0(self):
        """Return the upper grid size float value if it is constant."""
        if not self.is_b_equal:
            raise ValueError('Upper grid size is not constant, can`t get b0')
        return self.b[0] if self.b is not None else None

    @property
    def dict(self):
        """Return the dict with full benchmark info and history of requests."""
        return {
            'args': self.args,
            'opts': self.opts,
            'prps': self.prps,
            'hist': self.hist}

    @property
    def err(self):
        """Errors while benchmark usage in the text format."""
        return '; '.join(self.err_list) if len(self.err_list) else ''

    @property
    def hist(self):
        """Dict with history values (requests to the benchmark)."""
        return self.build_dict(self.hist_info)

    @property
    def hist_info(self):
        """Dict with info about benchmark's history parameters."""
        return {
            'err': {
                'desc': 'Error message',
                'kind': 'str',
                'info_skip_if_none': True
            },
            'm': {
                'desc': 'Number of requests',
                'kind': 'int',
                'form': '-10.3e'
            },
            'm_cache': {
                'desc': 'Number of cache uses',
                'kind': 'int',
                'form': '-10.3e'
            },
            'y_max': {
                'desc': 'Found maximum value',
                'kind': 'float',
                'form': '-10.3e',
                'info_add': 'y_max_real'
            },
            'y_min': {
                'desc': 'Found minimum value',
                'kind': 'float',
                'form': '-10.3e',
                'info_add': 'y_min_real'
            },
            'time_call_one': {
                'desc': 'Time per one request (sec)',
                'kind': 'float',
                'form': '-10.3e'
            },
            'time_call': {
                'desc': 'Total time per requests (sec)',
                'kind': 'float',
                'form': '-10.3e'
            },
            'time_full': {
                'desc': 'Total work time (sec)',
                'kind': 'float',
                'form': '-10.3e'
            },
            'i_max': {
                'desc': 'Multi-index for found maximum',
                'kind': 'int',
                'list': True,
                'list_skip_convert': True
            },
            'i_min': {
                'desc': 'Multi-index for found minimum',
                'kind': 'int',
                'list': True,
                'list_skip_convert': True
            },
            'x_max': {
                'desc': 'Point for found maximum',
                'kind': 'float',
                'list': True,
                'form': '.3f',
                'list_skip_convert': True
            },
            'x_min': {
                'desc': 'Point for found minimum',
                'kind': 'float',
                'list': True,
                'form': '.3f',
                'list_skip_convert': True
            },
            'y_list': {
                'desc': 'List of requested values',
                'kind': 'float',
                'list': True,
                'form': '.2f',
                'info_skip': True,
                'list_skip_convert': True
            },
            'y_list_full': {
                'desc': 'List of requested values (with cache)',
                'kind': 'float',
                'list': True,
                'form': '.2f',
                'info_skip': True,
                'list_skip_convert': True
            },
        }

    @property
    def identity(self):
        """Returns a list of arg names that define the benchmark."""
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
    def is_opti_min(self):
        """If the benchmark relates to minimization task (auto computed)."""
        return not self.is_opti_max

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
    def name_class(self):
        return self.__class__.__name__

    @property
    def opts(self):
        """Dict with values of benchmark's options (i.e., addit. parameters)."""
        return self.build_dict(self.opts_info)

    @property
    def opts_info(self):
        """Dict with info about benchmark's options."""
        return {
            'budget_raise': {
                'desc': 'Raise then out of budget',
                'kind': 'bool',
                'dflt': False,
                'info_skip_if_none': True
            },
        }

    @property
    def prps(self):
        """Dict with values of benchmark's properties."""
        return self.build_dict(self.prps_info)

    @property
    def prps_info(self):
        """Dict with info about benchmark's properties."""
        return {
            'version': {
                'desc': 'Package version',
                'kind': 'str'
            },
            'name_class': {
                'desc': 'Benchmark class name',
                'kind': 'str'
            },
            'is_tens': {
                'desc': 'Benchmark is discrete',
                'kind': 'bool',
                'info_skip': 'is_func'
            },
            'is_func': {
                'desc': 'Benchmark is continuous',
                'kind': 'bool',
                'info_skip': 'is_tens'
            },
            'is_opti_max': {
                'desc': 'Benchmark with maximization task',
                'kind': 'bool',
                'info_skip': 'is_opti_min'
                # TODO: see below
            },
            'is_opti_min': {
                'desc': 'Benchmark with minimization task',
                'kind': 'bool',
                'info_skip': 'is_opti_max'
                # TODO: In the rare case where the benchmark matches the mix of
                # minimization and maximization, the "info_skip" flag should be
                # removed from here manually in the child class.
            },
            'a': {
                'desc': 'Lower grid limit',
                'kind': 'float',
                'list': True,
                'form': '.2f',
                'info_skip_if_none': 'is_func'
            },
            'b': {
                'desc': 'Upper grid limit',
                'kind': 'float',
                'list': True,
                'form': '.2f',
                'info_skip_if_none': 'is_func'
            },
            'grid_kind': {
                'desc': 'Grid kind',
                'kind': 'str',
                'info_skip_if_none': 'is_func'
            },
            'with_plot': {
                'desc': 'The "plot" is available',
                'kind': 'bool',
                'info_skip_if_none': True
            },
            'with_show': {
                'desc': 'The "show" is available',
                'kind': 'bool',
                'info_skip_if_none': True
            },
            'with_render': {
                'desc': 'The "render" is available',
                'kind': 'bool',
                'info_skip_if_none': True
            },
            'with_cores': {
                'desc': 'TT-cores are available',
                'kind': 'bool',
                'info_skip_if_none': True
            },
            'with_cache': {
                'desc': 'Use cache',
                'kind': 'bool'
            },
            'with_constr': {
                'desc': 'Has constraint',
                'kind': 'bool'
            },
            'constr_with_amplitude': {
                'desc': 'Constraint with amplitude',
                'kind': 'bool',
                'info_skip_if_none': 'with_constr'
            },
            'constr_penalty': {
                'desc': 'Constraint penalty',
                'kind': 'float',
                'form': '.2e',
                'info_skip_if_none': 'with_constr'
            },
            'constr_eps': {
                'desc': 'Constraint epsilon',
                'kind': 'float',
                'form': '.2e',
                'info_skip_if_none': 'with_constr'
            },
            'budget_m': {
                'desc': 'Computation budget',
                'kind': 'int',
                'form': '.2e'
            },
            'budget_m_cache': {
                'desc': 'Computation budget for cache requests',
                'kind': 'int',
                'form': '.2e'
            },
            'budget_is_strict': {
                'desc': 'Computation budget is strict',
                'kind': 'bool',
                'info_skip_if_none': 'budget_m'
            },
            'y_max_real': {
                'desc': 'Exact max (value)',
                'kind': 'float',
                'form': '.4e'
            },
            'y_min_real': {
                'desc': 'Exact min (value)',
                'kind': 'float',
                'form': '.4e'
            },
            'i_max_real': {
                'desc': 'Exact max (multi-index)',
                'kind': 'int',
                'list': True
            },
            'i_min_real': {
                'desc': 'Exact min (multi-index)',
                'kind': 'int',
                'list': True
            },
            'x_max_real': {
                'desc': 'Exact max (point)',
                'kind': 'float',
                'list': True,
                'form': '.2f'
            },
            'x_min_real': {
                'desc': 'Exact min (point)',
                'kind': 'float',
                'list': True,
                'form': '.2f'
            }
        }

    @property
    def ref(self):
        """Get reference value (i, y) to check the benchmark."""
        raise NotImplementedError

    @property
    def time_call_one(self):
        """Average time for one value request from benchmark."""
        return self.time_call / self.m if self.m else 0.

    @property
    def time_full(self):
        """Full time of benchmark existence in seconds."""
        return tpc() - self.timestamp_start

    @property
    def version(self):
        """The version of the package."""
        return __version__

    @property
    def with_constr(self):
        """Return True if benchmark has a constraint."""
        return False

    @property
    def with_cores(self):
        """Return True if exact TT-cores can be constructed for benchmark."""
        return False

    @property
    def with_plot(self):
        """Return True if benchmark supports "plot" method."""
        return False

    @property
    def with_render(self):
        """Return True if benchmark supports "render" method."""
        return False

    @property
    def with_show(self):
        """Return True if benchmark supports "show" method."""
        return False

    def build_cores(self):
        """Return exact TT-cores for the TT-representation of the tensor."""
        if self.is_tens:
            # TODO: check and fix
            msg = 'Construction of the TT-cores does not work for tensors'
            raise ValueError(msg)

        if not self.with_cores:
            msg = 'Construction of the TT-cores does not supported for this BM'
            raise ValueError(msg)

        I = np.array([teneva.grid_flat(k) for k in self.n], dtype=int).T
        X = teneva.ind_to_poi(I, self.a, self.b, self.n, self.grid_kind)

        return self.cores(X)

    def build_dict(self, info):
        """Build a dictionary with class variables."""
        res = {}
        for name, opts in info.items():
            if not hasattr(self, name):
                raise ValueError(f'Variable "{name}" does not exist')
            res[name] = getattr(self, name, None)
            if opts.get('list') and not opts.get('list_skip_convert'):
                res[name] = self.list_convert(res[name], opts['kind'])
        return res

    def build_trn(self, m=0, seed=None, skip_process=False):
        """Generate random (from LHS) train dataset of (index, value)."""
        if m < 1:
            return None, None

        if seed is None:
            seed = self.seed

        I_trn = teneva.sample_lhs(self.n, m, seed=seed)
        y_trn = self.get(I_trn, skip_process)

        if y_trn is None:
            raise ValueError('The specified budget is exceeded')

        return I_trn, y_trn

    def build_tst(self, m=0, seed=None, skip_process=True):
        """Generate random (from "choice") test dataset of (index, value)."""
        if m < 1:
            return None, None

        if seed is None:
            seed = self.seed

        I_tst = teneva.sample_rand(self.n, m, seed=seed)
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
            msg = 'Upper grid limit "b" should be set for continuous function'
            self.set_err(msg)

        return self.check_err()

    def check_args(self, **kwargs):
        """Check that benchmark has valid arguments."""
        is_valid = True

        for name, constr in self.args_constr.items():
            if not hasattr(self, name):
                raise ValueError(f'Invalid name "{name}" in args_constr')

            opts = self.args_info.get(name)
            if opts is None:
                raise ValueError(f'Name "{name}" is not in args')

            v = kwargs.get(name, getattr(self, name, None))

            if isinstance(constr, str): # List of equal values
                if constr != 'equal':
                    raise ValueError(f'Invalid constraint for "{name}"')
                if not opts.get('list'):
                    raise ValueError(f'Invalid constraint for "{name}"')

                v = self.list_convert(v, opts['kind'])
                if v is not None and not isinstance(v, (int, float, str)):
                    msg = f'Arg "{name}" should be list '
                    msg += 'of equal values'
                    self.set_err(msg)
                    is_valid = False

            elif isinstance(constr, (int, float)): # Exact value
                if opts.get('list'):
                    v = self.list_convert(v, opts['kind'])
                    if v is not None and not isinstance(v, (int, float, str)):
                        msg = f'Arg "{name}" should be list of '
                        msg += f'values equal to "{constr}"'
                        self.set_err(msg)
                        is_valid = False
                        continue

                if v != constr:
                    msg = f'Arg "{name}" should be "{constr}"'
                    self.set_err(msg)
                    is_valid = False

            else: # Enum
                if opts.get('list'):
                    raise NotImplementedError
                if not v in constr:
                    msg = f'Arg "{name}" should be from {constr}'
                    self.set_err(msg)
                    is_valid = False

        return is_valid

    def check_err(self):
        """Check that benchmark has not errors."""
        if len(self.err_list):
            msg = f'BM "{self.name}" has errors:'
            msg = '\n\n' + '-' * len(msg) + '\n' + msg
            for err in self.err_list:
                msg += f'\n    Error > {err}'
            raise ValueError(msg + '\n\n')

        return True

    def compute(self, X, skip_process=False):
        m = self.m
        m_cur = X.shape[0]
        m_max = self.budget_m

        if not skip_process and m_max:
            if (m >= m_max) or (self.budget_is_strict and m + m_cur > m_max):
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

    def cores_add(self, X, a0=0.):
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
            ind_new = [k for k in range(m) if tuple(I[k]) not in self.cache]

            m_new = len(ind_new)
            dm_cache = m - m_new

            if self.budget_m_cache:
                if self.budget_is_strict:
                    if self.m_cache + dm_cache > self.budget_m_cache:
                        return self.process_last(is_cache=True)
                else:
                    if self.m_cache > self.budget_m_cache:
                        return self.process_last(is_cache=True)

            if m_new > 0:
                Z = X[ind_new] if self.is_func else I[ind_new]
                y_new = self.compute(Z, skip_process)
                if y_new is None:
                    return self.process_last()

                for k in range(m_new):
                    self.cache[tuple(I[ind_new[k]])] = y_new[k]

            y = np.array([self.cache[tuple(i)] for i in I])

        else:
            ind_new = None

            Z = X if self.is_func else I
            y = self.compute(Z, skip_process)
            if y is None:
                return self.process_last()

        return self.process(I, X, y, ind_new, t, is_batch, skip_process)

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

        return self.process(I, X, y, None, t, is_batch, skip_process)

    def info(self, footer=''):
        """Returns a detailed description of the benchmark as text."""
        text = self.info_prefix()

        text += self.info_section('Description')
        text += self.info_desc()

        text_section = ''
        for name, opt in self.args_info.items():
            text_section += self.info_var(name, opt, with_name=True)
        if text_section:
            text += self.info_section('Arguments') + text_section

        text_section = ''
        for name, opt in self.opts_info.items():
            text_section += self.info_var(name, opt, with_name=True)
        if text_section:
            text += self.info_section('Options') + text_section

        text_section = ''
        for name, opt in self.prps_info.items():
            text_section += self.info_var(name, opt, skip_none=True)
        if text_section:
            text += self.info_section('Properties') + text_section

        return text + footer + '=' * 78 + '\n'

    def info_current(self, footer=''):
        text = ''
        form = '{:-' + str(8 + self.log_prec) + '.' + str(self.log_prec) + 'e}'

        if self.log_prefix:
            text += self.log_prefix + ' > '

        text += f'm {self.m:-7.1e}'
        if self.with_cache:
            text += f' [+ {self.m_cache:-7.1e}]'
        text += ' | '

        text += f't {self.time_full:-7.1e} | '

        if self.log_with_min and self.y_min is not None:
            value = form.format(self.y_min)
            text += f'min {value} | '

        if self.log_with_max and self.y_max is not None:
            value = form.format(self.y_max)
            text += f'max {value} | '

        self.log_m_last = self.m

        return text + footer

    def info_desc(self):
        text = '.' * 78 + '\n'
        desc = f'    {self.desc.strip()}'
        text += desc.replace('            ', '    ')
        text += '\n'
        text += '.' * 78 + '\n'
        return text

    def info_history(self, footer=''):
        """Returns an information about the history of requests (text)."""
        text = self.info_prefix()

        text += self.info_section('History of requests')

        if self.m == 0:
            text += '        ... history is empty ...\n'

        else:
            for name, opt in self.hist_info.items():
                text += self.info_var(name, opt, skip_none=True)

        return text + footer + '=' * 78 + '\n'

    def info_prefix(self):
        text = '-' * 78 + '\n' + 'BM: '
        text += self.name + ' ' * max(0, 36-len(self.name)) +  ' | '
        text += f'DIMS = {self.d:-4d} | '
        n = 0 if self.n is None else np.mean(self.n)
        text += '<MODE SIZE> = ' + (f'{n:-7.1f}' if n<9999 else f'{n:-7.1e}')
        return text + '\n'

    def info_section(self, name):
        text = '-' * 41 + '|             >'
        text += ' ' * max(0, 22-len(name)) + name
        return text + '\n'

    def info_var(self, name, opt, with_name=False, skip_none=False):
        kind = opt.get('kind', 'str')
        form = opt.get('form', None)

        def is_none(v, with_bool=False):
            if v is None:
                return True
            if isinstance(v, str) and v == '':
                return True
            if isinstance(v, bool) and not v and with_bool:
                return True
            return False

        v = getattr(self, name, None)
        if opt.get('list') and not opt.get('list_skip_convert'):
            v = self.list_convert(v, kind)

        if skip_none and is_none(v):
            return ''

        cond = opt.get('info_skip')
        if cond:
            if cond is True:
                return ''
            elif isinstance(cond, str):
                v_ref = getattr(self, opt['info_skip'], False)
                if not is_none(v_ref, with_bool=True):
                    return ''

        cond = opt.get('info_skip_if_none')
        if cond:
            if cond is True:
                if v is None or v == '' or v == False:
                    return ''
            elif isinstance(cond, str):
                v_ref = getattr(self, cond, None)
                if is_none(v_ref, with_bool=True):
                    return ''

        def build(v):
            if v is None:
                return 'NONE'
            elif isinstance(v, (list, np.ndarray)):
                if form:
                    v = [('{:' + form + '}').format(v_) for v_ in v]
                if self.d > 3:
                    v = f'[{v[0]}, {v[1]}, <...>, {v[-1]}]'
                return f'{v}'
            elif form:
                return ('{:' + form + '}').format(v)
            elif kind == 'bool':
                return 'YES' if v else 'no'
            elif kind == 'int':
                return f'{v}'
            elif kind == 'float':
                return f'{v:.6f}'
            else:
                return f'{v}'

        text = opt['desc']
        text += f' [{name}]' if with_name else ''
        text += ' ' * max(0, 40-len(text)) + ' : '
        text += build(v)
        if opt.get('list') and isinstance(v, (int, float, str)):
            text += f', ..., ' + build(v)

        if opt.get('info_add'):
            v = getattr(self, opt['info_add'], None)
            if v is not None:
                # TODO: do it more accurate
                text += f'   [real: {build(v)}]'

        return text + '\n'

    def init(self):
        self.err_list = []

        self.i = None
        self.x = None
        self.y = None

        self.i_max = None
        self.x_max = None
        self.y_max = None
        self.is_y_max_new = False

        self.i_min = None
        self.x_min = None
        self.y_min = None
        self.is_y_min_new = False

        self.y_list = []
        self.y_list_full = []

        self.m = 0
        self.m_cache = 0

        self.log_m_last = 0

        self.time_call = 0.
        self.timestamp_start = tpc()

        self.cache = {}

    def list_convert(self, x, kind='float', eps=1.E-16):
        """Convert list of (equal) values to one number and back."""
        if x is None:
            return None

        if kind == 'str':
            if isinstance(x, (int, float, str)):
                return np.array([str(x)]*self.d, dtype=object)
            for v in x:
                if v != x[0]:
                    return self.list_copy(x, 'str')
            return str(x[0]) if len(x) > 0 else None

        elif kind == 'int':
            if isinstance(x, (int, float)):
                return np.array([x]*self.d, dtype=int)
            for v in x:
                if np.abs(v - x[0]) > eps:
                    return self.list_copy(x, 'int')
            return int(x[0]) if len(x) > 0 else None

        elif kind == 'float':
            if isinstance(x, (int, float)):
                return np.array([x]*self.d, dtype=float)
            for v in x:
                if np.abs(v - x[0]) > eps:
                    return self.list_copy(x, 'float')
            return float(x[0]) if len(x) > 0 else None

        else:
            raise ValueError('Unsupported kind for list conversion')

    def list_copy(self, x, kind=None):
        """Copy list or array and return the new array."""
        if x is None:
            return None

        if not kind:
            x = np.asanyarray(x)
        elif kind == 'str':
            x = np.asanyarray(x, dtype=object)
        elif kind == 'int':
            x = np.asanyarray(x, dtype=int)
        elif kind == 'float':
            x = np.asanyarray(x, dtype=float)
        else:
            raise ValueError('Unsupported kind for list copy')

        return x.copy()

    def log_check(self):
        if not self.with_log:
            return False

        if self.log_cond == 'min':
            return self.is_y_min_new

        if self.log_cond == 'max':
            return self.is_y_max_new

        if self.log_cond in ['min-max', 'max-min']:
            return self.is_y_min_new or self.is_y_max_new

        if self.log_cond == 'step':
            return self.log_step and self.m - self.log_m_last >= self.log_step

    def parse_input(self, I=None, X=None):
        if I is None and X is None or I is not None and X is not None:
            msg = 'Can`t parse input. Invalid case'
            raise ValueError(msg)

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

    def plot(self, fpath=None):
        """Build the plot for benchmark."""
        raise NotImplementedError

    def prep(self):
        """A function with a specific benchmark preparation code."""
        if self.is_prep:
            raise ValueError('Benchmark is already prepared')

        self.check_args()
        self.check_err()
        self.prep_bm()

        self.is_prep = True
        return self

    def prep_bm(self):
        """A function with a specific benchmark preparation code (inner)."""
        return

    def process(self, I, X, y, ind_new=None, t=0., is_batch=False, skip=False):
        if skip:
            return y if is_batch else y[0]

        self.y_list.extend(list(y if ind_new is None else y[ind_new]))
        self.y_list_full.extend(list(y))

        self.m += len(y) if ind_new is None else len(ind_new)
        self.m_cache += 0 if ind_new is None else len(y) - len(ind_new)

        self.time_call += tpc() - t

        self.i = I[-1, :].copy() if I is not None else None
        self.x = X[-1, :].copy() if X is not None else None
        self.y = y[-1]

        self.is_y_max_new = False
        self.is_y_min_new = False

        ind = np.argmax(y)
        if self.y_max is None or self.y_max < y[ind]:
            self.i_max = I[ind, :].copy() if I is not None else None
            self.x_max = X[ind, :].copy() if X is not None else None
            self.y_max = y[ind]
            self.is_y_max_new = True

        ind = np.argmin(y)
        if self.y_min is None or self.y_min > y[ind]:
            self.i_min = I[ind, :].copy() if I is not None else None
            self.x_min = X[ind, :].copy() if X is not None else None
            self.y_min = y[ind]
            self.is_y_min_new = True

        if self.log_check():
            self.log(self.info_current())

        if self.cache_m_max and len(self.cache.keys()) > self.cache_m_max:
            self.cache = {}
            self.wrn('The maximum cache size has been exceeded. Cache cleared')

        return y if is_batch else y[0]

    def process_last(self, is_cache=False):
        if self.with_log:
            self.log(self.info_current('<<< DONE\n'))

        if self.budget_raise:
            m = self.budget_m_cache if is_cache else self.budget_m
            raise BmBudgetOverException(m, is_cache)

    def recover(self, i=None, best=True):
        """Restores some benchmark-specific values."""
        # TODO: check the role of this function
        raise NotImplementedError

    def render(self, fpath=None, i=None, best=True):
        """Render the solution for benchmark."""
        raise NotImplementedError

    def set_budget(self, m=None, m_cache=None, is_strict=True):
        """Set computation budget."""
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
        if err:
            self.err_list.append(err)

    def set_grid(self, a=None, b=None, sh_sc=100., sh=False, sh_out=False):
        """Set grid lower (a) and upper (b) limits for the function-like BM."""
        if (a is not None or b is not None) and not self.d:
            raise ValueError('Please, set dimension "d" before')

        self.a = teneva.grid_prep_opt(a, self.d)
        self.b = teneva.grid_prep_opt(b, self.d)

        if self.a is None and self.b is None:
            return

        if self.a is None or self.b is None:
            raise ValueError('Please, set both "a" and "b"')

        if len(self.a) != self.d:
            raise ValueError('Arg "a" has invalid length')

        if len(self.b) != self.d:
            raise ValueError('Arg "b" has invalid length')

        if sh:
            # Note that we use "constant" noise:
            rand = np.random.default_rng(42)
            a_sh = rand.uniform(0, (self.b-self.a) / sh_sc, size=self.d)
            b_sh = rand.uniform(0, (self.b-self.a) / sh_sc, size=self.d)

            self.a = self.a + a_sh * (-1. if sh_out else +1.)
            self.b = self.b - b_sh * (-1. if sh_out else +1.)

        for k in range(self.d):
            if self.a[k] >= self.b[k]:
                raise ValueError('Invalid grid limits (a >= b)')

    def set_grid_kind(self, kind='cheb'):
        """Set the kind of the grid ('cheb' or 'uni')."""
        if not kind in ['uni', 'cheb']:
            msg = f'Invalid kind of the grid (should be "uni" or "cheb")'
            raise ValueError(msg)

        self.grid_kind = kind

    def set_log(self, log=False, cond=None, step=1000, prec=3, prefix='bm',
                with_max=None, with_min=None, log_wrn=None):
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

        if cond is None:
            cond = 'max' if self.is_opti_max else 'min'
        if not cond in ['min', 'max', 'min-max', 'max-min', 'step']:
            raise ValueError(f'Invalid "cond" argument "{cond}"')

        self.log_cond = cond
        self.log_step = int(step) if step else None
        self.log_prec = int(prec)
        self.log_prefix = prefix

        if with_max is None:
            self.log_with_max = self.is_opti_max
        else:
            self.log_with_max = with_max

        if with_min is None:
            self.log_with_min = self.is_opti_min
        else:
            self.log_with_min = with_min

    def set_max(self, i=None, x=None, y=None):
        """Set exact (real) global maximum (index, point and related value)."""
        self.i_max_real = i
        self.x_max_real = x
        self.y_max_real = y

        if self.i_max_real is not None:
            if isinstance(self.i_max_real, (int, float)):
                self.i_max_real = self.list_convert(self.i_max_real, 'int')
            else:
                self.i_max_real = self.list_copy(self.i_max_real, 'int')
        if self.x_max_real is not None:
            if isinstance(self.x_max_real, (int, float)):
                self.x_max_real = self.list_convert(self.x_max_real, 'float')
            else:
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
            if isinstance(self.i_min_real, (int, float)):
                self.i_min_real = self.list_convert(self.i_min_real, 'int')
            else:
                self.i_min_real = self.list_copy(self.i_min_real, 'int')
        if self.x_min_real is not None:
            if isinstance(self.x_min_real, (int, float)):
                self.x_min_real = self.list_convert(self.x_min_real, 'float')
            else:
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

    def set_opts(self, **kwargs):
        """Set values for some of options specific to the benchmark."""
        for name, value in kwargs.items():
            if not name in self.opts_info.keys():
                raise ValueError(f'Option "{name}" does not exist')
            setattr(self, name, value)

    def set_opts_dflt(self):
        """Set default values for options specific to the benchmark."""
        for name, opt in self.opts_info.items():
            if not 'dflt' in opt:
                raise ValueError(f'Option "{name}" has not default value')
            if hasattr(self, name):
                raise ValueError(f'Invalid option name "{name}" (conflict)')
            setattr(self, name, opt['dflt'])

    def set_size(self, n=None):
        """Set sizes for all d-modes (n should be int or list)."""
        if n is not None and not self.d:
            raise ValueError('Please, set dimension "d" before')

        self.n = teneva.grid_prep_opt(n, self.d, int)

        if self.n is not None and len(self.n) != self.d:
            raise ValueError('Arg "n" has invalid length')

    def show(self, fpath=None, i=None, best=True):
        """Present the state of the benchmark (image, graph, etc.)."""
        raise NotImplementedError

    def target(self, x):
        """Function that computes value for a given point/index."""
        return self.target_batch(x.reshape(1, -1))[0]

    def target_batch(self, X):
        """Function that computes values for a given batch of points/indices."""
        return np.array([self.target(x) for x in X])

    def wrn(self, text):
        self.log_wrn('!!! BM-WARNING : ' + text)


class BmBudgetOverException(Exception):
    def __init__(self, m, is_cache=False):
        self.m = m
        self.is_cache = is_cache

        self.message = 'Computation budget '
        if self.is_cache:
            self.message += 'for cache '
        self.message += f'(m={self.m}) exceeded'

        super().__init__(self.message)
