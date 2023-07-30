# teneva_bm


## Description

Benchmarks library, based on the software product [teneva](https://github.com/AndreiChertkov/teneva), for testing multidimensional approximation and optimization methods. Our benchmarks include both multidimensional data arrays and discretized functions of many variables.


## Installation

> Current version "0.6.3".

The package can be installed via pip: `pip install teneva_bm` (it requires the [Python](https://www.python.org) programming language of the version 3.8 or 3.9). It can be also downloaded from the repository [teneva_bm](https://github.com/AndreiChertkov/teneva_bm) and installed by `python setup.py install` command from the root folder of the project.

> Required python packages (see `requirements.txt`) [matplotlib](https://matplotlib.org/) (3.7.0+) and [teneva](https://github.com/AndreiChertkov/teneva) (0.14.5+) will be automatically installed during the installation of the main software product.

Some benchmarks require additional installation of specialized libraries. The corresponding instructions are given in the description of each benchmark (see `DESC` string in the python files with benchmarks). Installation of all required libraries for all benchmarks can be done with the following commands:

- Collections `func`, `hs` and `various` do not require installation of additional libraries.

- Сollections `odeoc` and `qubo` require installation of the following libraries:
    ```bash
    pip install networkx==3.0 qubogen==0.1.1 gekko==1.0.6
    ```

- Сollection `agent` require a rather complicated installation process of the `gym` and `mujoco` frameworks and related packages, so we have prepared a special python installation script [install_mujoco.py](https://github.com/AndreiChertkov/teneva_bm/blob/main/install_mujoco.py). Detailed instructions for using the script are presented in the file header.

> To run benchmark optimization examples (see `demo_opti` folder), you should also install the [PROTES](https://github.com/anabatsh/PROTES) optimizer (`pip install protes==0.3.3`).


## Documentation and examples

All benchmarks inherit from the `Bm` base class (`teneva_bm/bm.py`) and are located in the subfolders (collections of benchmarks) of `teneva_bm` folder. The corresponding python files contain a detailed description of the benchmarks, as well as a scripts for a demo run at the end of the files. You can get detailed information on the created benchmark using the `info` class method:
```python
from teneva_bm import *
bm = BmQuboMaxcut()
bm.prep()
print(bm.info())
```

You can run demos for all benchmarks at once with the command `python demo.py` from the root folder of the project (you can also specify the name of the benchmark as a script argument to run the demo for only one benchmark, e.g., `python demo.py bm_qubo_knap_det`). You can also use a function from the `teneva_bm` package to run all or only one demo:
```python
from teneva_bm import teneva_bm_demo
teneva_bm_demo('bm_qubo_knap_det', with_info=True)
```

> We prepare some demo scripts with benchmark optimization examples in the `demo_opti` folder. To run these examples (e.g., `python demo_opti/base.py`), you need to install the [PROTES](https://github.com/anabatsh/PROTES) optimizer). We also present some examples in this [colab notebook](https://colab.research.google.com/drive/1z8LgqEARJziKub2dVB65CHkhcboc-fCH?usp=sharing).


## Available benchmarks

- `agent` - the collection of problems from [gym](https://www.gymlibrary.dev/) framework, including [mujoco agents](https://www.gymlibrary.dev/environments/mujoco/index.html) based on the physics engine [mujoco](https://mujoco.org/) for faciliatating research and development in robotics, biomechanics, graphics and animation. The collection includes the following benchmarks: `BmAgentAnt`, `BmAgentHuman`, `BmAgentHumanStand`, `BmAgentLake`, `BmAgentPendInv`, `BmAgentPendInvDouble`, `BmAgentSwimmer`.
    > Within the framework of this collection, explicit optimization of the entire set of actions (discrete or continuous) may be performed (if `direct` policy name is set) or discrete Toeplitz policy may be used (if `toeplitz` policy name is set; it is the default value); you can also set your own custom policy as an instance of the correct class (see `agent/policy.py` for details).

- `func` - a collection of analytic functions of a real multidimensional argument. The collection includes the following benchmarks: `BmFuncAckley`, `BmFuncAlpine`, `BmFuncDixon`, `BmFuncExp`, `BmFuncGriewank`, `BmFuncMichalewicz`, `BmFuncPiston` (only `d=7` is supported), `BmFuncQing`, `BmFuncRastrigin`, `BmFuncRosenbrock`, `BmFuncSchaffer`, `BmFuncSchwefel`.
    > For almost all functions, the exact global minimum ("continuous x point", not multi-index) is known (see `bm.x_min_real` and `bm.y_min_real`). For a number of functions (`BmFuncAlpine`, `BmFuncExp`, `BmFuncGriewank`, `BmFuncMichalewicz`, `BmFuncQing`, `BmFuncRastrigin`, `BmFuncRosenbrock`, `BmFuncSchwefel`), a `bm.build_cores()` method is available that returns an exact representation of the function on the discrete grid used in the benchmark in the tensor train (TT) format as a list of 3D TT-cores. Note also that we apply small random shift of the grid limits for all functions, to make the optimization problem more difficult (because many functions have a minimum at the center point of the domain).

- `hs` (draft!) - the [Hock & Schittkowski](http://apmonitor.com/wiki/index.php/Apps/HockSchittkowski) collection of benchmark functions, containing continuous analytic functions of small dimensions (2-5), some of which have given constraints. The collection includes the following benchmarks: `BmHsFunc001`, `BmHsFunc006`.

- `odeoc` - a collection of optimal control problems described by ordinary differential equations, some of the problems have explicit restrictions on the elements of the control vector. The collection includes the following benchmarks: `BmOdeocSimple`, `BmOdeocSimpleConstr`.

- `qubo` - a collection of quadratic unconstrained binary optimization (QUBO) problems; all benchmarks are discrete and have a mode size equals `2`. The collection includes the following benchmarks: `BmQuboKnapDet`, `BmQuboKnapQuad`, `BmQuboMaxcut`, `BmQuboMvc`.
    > The exact global minimum is known only for `BmQuboKnapDet` benchmark (note that this benchmark supports only dimensions `10`, `20`, `50`, `80` and `100`).

- `various` - a collection of heterogeneous benchmarks that are not suitable for any other collection (note that in this case, we do not use the name of the collection in the name of the benchmarks, unlike all other sets). The collection includes the following benchmarks: `BmMatmul`, `BmTopopt` (draft!), `BmWallSimple`.


## Usage

A typical scenario for working with a benchmark is as follows.

##### Benchmark initialization

First, we create an instance of the desired benchmark class and manually call the `prep` method (optionally, we can also print detailed information about the benchmark):
```python
import numpy as np
from teneva_bm import *

bm = BmFuncAckley()
bm.prep()
print(bm.info())
```

The class constructor of all benchmarks has the following optional arguments:

- `d` - dimension of the benchmark's input (non-negative integer). For some benchmarks, this number is hardcoded (or depends on other specified auxiliary arguments), if another value is explicitly passed, an error message will be generated (e.g., the dimension for benchmark `various.bm_matmul` is determined automatically by the values of auxiliary arguments `size`, `rank` and `only2`). By default, some correct value is used (specified in the benchmark description).

- `n` - number of possible discrete values for benchmark input variables, i.e., the mode size of the related tensor / multidimensional array (non-negative integer if all mode sizes are equal or a list of non-negative integers of the length `d`). For some benchmarks, this number is hardcoded (or depends on other specified auxiliary arguments), if another value is explicitly passed, an error message will be generated (e.g., in `qubo` collection all benchmarks should have `n = 2`). By default, some correct value is used (specified in the benchmark description).

- `name` - the display name of the benchmark (string). By default, the name corresponding to the file/class name (without `Bm` prefix) is used.

- `desc` - the description of the benchmark (string). By default, a detailed description of the benchmark is used, provided in the corresponding python file.

- `...other arguments...` - some benchmarks have additional optional arguments, which are described in the corresponding python files.

##### Setting advanced options

Before calling the `bm.prep()` method, you can set a number of additional benchmark options:

- `bm.set_seed(seed=42)` - with this function we can set a custom random seed. Note that we use `Random Generator` from numpy (i.e., `numpy.random.default_rng(seed)`) and for a fixed value of the seed, the behavior of the benchmark will always be the same, however, not all benchmarks depend on a seed.

- `bm.set_grid_kind(kind='cheb')` - by default, we use the Chebyshev grid (`kind = 'cheb'`) for benchmarks corresponds to a function of a continuous argument, but you can alternatively set it manually to use a uniform grid (`kind = 'uni'`).

- `bm.set_budget(m=None, m_cache=None, is_strict=True)` - optional method to set the computation buget `m`. If the number of requests to the benchmark (from calls to `bm.get` and `bm.get_poi` methods) exceeds the specified budget, then `None` will be returned. If the flag `is_strict` is disabled, then the request for the last batch will be allowed, after which the budget will be exceeded, otherwise this last batch will not be considered. Note that when the budget is exceeded, `None` will be returned both when requesting a single value and a batch of values. Also, in a similar way, you can set a limit on the use of the cache by `m_cache` parameter.

- `bm.set_constr(penalty=1.E+42, eps=1.E-16, with_amplitude=True)` - if the benchmark has a constraint, then using this function you can set a `penalty` (for the requested points that do not satisfy the constraint, the value `penalty * constraint_value` will be returned) and a `eps` (threshold value to check that the constraint has been fulfilled). Note that we set the constraints as a function (`bm.constr` / `bm.constr_batch`) that returns the value `constraint_value` (amplitude) of the constraint, and if the constraint is met, then the value must be non-positive, otherwise, the objective function is not calculated and a value proportional to the amplitude of the constraint is returned (if you disable flag `with_amplitude`, then just the value of the penalty will be returned). For the case of maximization task you should set negative `penalty` value.

- `bm.set_max(i=None, x=None, y=None)` - if necessary, you can manually set the multi-index, the corresponding continuous point (for benchmarks, which relate to functions of a continuous argument), and the corresponding value for the exact global maximum of the function. The corresponding values will be further available in the benchmark as `bm.i_max_real`, `bm.x_max_real` and `bm.y_max_real` respectively. When the benchmark is initialized, this function is called automatically if the optimum is known.

- `bm.set_min(i=None, x=None, y=None)` - the same as in the previous point, but for the global minimum.

- `bm.set_log(log=False, cond='min-max', step=1000, prefix='bm', with_min=True, with_max=True)` - when calling this function with the `True` argument `log`, the log will be printed while requests to benchmark. You may set the log codition `cond` (`min`, `max`, `min-max` or `step`; e.g., in the case `min` the log will be presented each time the `min` value is updated), the log step (for condition `step`) and a string `prefix` for the log. You can also disable the display of current minimum values (`with_min`) or maximum values (`with_max`) in the log string. Note that you can provide as `log` argument some print-like function, e.g., `log=print`, in this case, printing will occur not to the console, but to the corresponding function.

- `bm.set_cache(with_cache=False, cache=None, m_max=1.E+8)` - when calling this function with the `True` argument `with_cache`, the cache will be used, that is, all the values requested from the benchmark will be saved and when the same multi-indices are accessed again, the values will be retrieved from the cache instead of explicitly calculating the objective function. Additionally, you can optionally pass as an argument `cache` an already existing cache in the form of a dictionary (the keys are multi-indices in the form of tuples, and the values are the corresponding values of the objective function). We especially note that the cache is only used when querying benchmark values in discrete multi-indices; for requested continuous points, no cache will be used. It is also important to note that no cache will be used for matching multi-indices in the same requested batch of values. Optionally, you can set `m_max` argument that specifies the maximum cache size. If the size is exceeded, the cache will be cleared and a corresponding warning will be displayed to the log. Note that when the `bm.init` method is called, the cache is always reset to zero.

- `bm.set_opts(...)` - for some benchmarks, this function may be called to set additional benchmark-specific options (please see the description of arguments in the relevant benchmark code file).

> You can get all configuration options as a dictionary by the function `bm.get_config()`.

##### Computing benchmark values

Now the benchmark is ready to go, and we can calculate its value in any requested discrete multi-index (a real number will be returned) or a list of its values for any requested batch of discrete multi-indices (1D numpy array of real numbers will be returned):
```python
# Get value at multi-index i:
i = np.ones(bm.d)
print(bm[i]) # you can use the alias "bm.get(i)"

# Get values for batch of multi-indices I:
I = np.array([i, i+1, i+2])
print(bm[I]) # you can use the alias "bm.get(I)"
```
Note that the `get` method can be used instead of `[ ]` notation, for example, if it is necessary to pass somewhere a function that calculates benchmark values.

Since the considered benchmark (`BmFuncAckley`) corresponds to a function of a continuous argument, above we calculated the values for the discretization of the function on a automatically selected grid. Additionally, we can calculate values at continuous points by analogy:
```python
# Get value at point x:
x = np.ones(bm.d) * 0.42
print(bm(x)) # you can use the alias "bm.get_poi(x)"

# Get values for batch of points X:
X = np.array([x, x*0.3, x*1.1])
print(bm(X)) # you can use the alias "bm.get_poi(X)"
```

##### Dataset generation

For convenience, the benchmark also has functions that allow you to generate training and test data sets on a discrete grid:
```python
# Generate random train dataset (from LHS):
# I_trn is array of [500, bm.d] and y_trn is [500]
I_trn, y_trn = bm.build_trn(500)

# Generate random test dataset (from random choice):
# I_tst is array of [100, bm.d] and y_tst of [100]
I_tst, y_tst = bm.build_tst(100)
```

> Note that, by default, the spent computational budget (see `bm.set_budget` function) and time spent does not change when the test dataset is generated (this is controlled by the default `skip_process` flag value in `bm.build_tst(m=0, skip_process=True)`), but is consumed when generating the training dataset (i.e., `bm.build_trn(m=0, skip_process=False)`).

##### Request history analysis

During requests to the benchmark, that is, when calling functions `bm[]` (or `bm.get`), `bm()` (or `bm.get_poi`), `bm.build_trn` (if the flag `skip_process` is not set in the function arguments; it has a value `False` by default) and `bm.build_tst` (if `skip_process` is not set; it is `True` by default for this function), the following useful class parameters are updated:

- `bm.m` - the total number of performed calculations of the benchmark value (if a cache is used, then the values taken from the cache are not taken into account in this variable).

- `bm.m_cache` - the total number of cache hits performed instead of explicitly calculating benchmark values (if no cache is used, it is `0`).

- `bm.time` - total time in seconds spent on calculating the benchmark values (the time spent on cache accesses is also taken into account).

- `bm.time_full` - benchmark lifetime in seconds from the moment of initialization (i.e., the call to `bm.init` method).

- `bm.y_list` - a list of all sequentially calculated benchmark values (results of cache accesses are also added to the list).

- `bm.i_max`, `bm.x_max`, `bm.y_max` - a discrete multi-index, a continuous multi-dimensional point, and benchmark values corresponding to the maximum of all requested values. Note that for the case of a discrete function, the value of `x_max` will be `None`, and for the case of a continuous function, the values of `i_max` and `x_max` will correlate, while if requests were made for continuous points, then `x_max` will correspond to the exact position of the point, and `i_max` will be the nearest multi-index of the used discrete grid.

- `bm.i_min`, `bm.x_min`, `bm.y_min` - same as in the previous point, but for the minimum value.

- `bm.i`, `bm.x`, `bm.y` - the last requested multi-index / point and the related computed benchmark's value.

> The following function may be used to print the corresponding values: `print(bm.info_history())`. You can also get these values as a dictionary by the function `bm.get_history()`.

##### Notes

- For some benchmarks (e.g., for all benchmarks from `agent` collection) the method `show` (present the current state, the state for provided input or the final state for the best found input) and `render` (present the animation for the current solution, the solution from the provided input or for the solution from the best found input) are available.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)


## Contributors

- [Anastasia Batsheva](https://github.com/anabatsh)

We are happy to invite you to become a contributor, especially if you have interesting benchmarks ;) Please see detailed instructions for developers in `workflow.md`.


---


> ✭__🚂  The stars that you give to **teneva_bm**, motivate us to develop faster and add new interesting features to the code 😃
