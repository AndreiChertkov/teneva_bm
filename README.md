# teneva_bm


## Description

Benchmarks library, based on the software product [teneva](https://github.com/AndreiChertkov/teneva), for testing multidimensional approximation and optimization methods. Our benchmarks include both multidimensional data arrays and discretized functions of many variables.


## Installation

1. The package can be installed via pip (it requires the [Python](https://www.python.org) programming language of the version 3.8 or 3.9):
    ```bash
    pip install teneva_bm==0.8.4
    ```
    > The package can be also downloaded from the repository [teneva_bm](https://github.com/AndreiChertkov/teneva_bm) and be installed by `python setup.py install` command from the root folder of the project.

2. Some benchmarks require additional installation of specialized libraries. The corresponding instructions are given in the description of each benchmark. Fast installation of all required libraries for all benchmarks can be done with the script [install_all.py](https://github.com/AndreiChertkov/teneva_bm/blob/main/install_all.py) for existing [anaconda](https://www.anaconda.com) environment `ENV_NAME` (except the colab platform, where `env` flag should not be used):
    ```bash
    conda create --name ENV_NAME python=3.8 -y
    wget https://raw.githubusercontent.com/AndreiChertkov/teneva_bm/main/install_all.py
    python install_all.py --env ENV_NAME --silent
    ```
    > Please note that the collection `agent` requires a rather complicated installation process of the `gym` and `mujoco` frameworks and related packages, so the script `install_all.py` is rather complicated. You can find mode details for using this script in the header of the file. If you have problems downloading the script via wget, you can download it manually from the root folder of the repository [teneva_bm](https://github.com/AndreiChertkov/teneva_bm).

3. To run benchmark optimization examples (see `demo/opti_*.py` folder), you should also install the [PROTES](https://github.com/anabatsh/PROTES) optimizer:
    ```bash
    pip install protes==0.3.5
    ```


## Documentation and examples

All benchmarks inherit from the `Bm` base class (`teneva_bm/bm.py`) and are located in the subfolders (collections of benchmarks) of `teneva_bm` folder. The corresponding python files contain a detailed description of the benchmarks. You can get detailed information on the created benchmark using the `info` method:
```python
from teneva_bm import *
bm = BmQuboMaxcut()
bm.prep()
print(bm.info())
```

We prepare some demo scripts with benchmark usage and optimization examples in the `demo_opti` folder. We also present some examples in this [colab notebook](https://colab.research.google.com/drive/1z8LgqEARJziKub2dVB65CHkhcboc-fCH?usp=sharing).


## Available benchmarks

- `agent` - a collection of problems from [gym](https://www.gymlibrary.dev/) framework, including [mujoco agents](https://www.gymlibrary.dev/environments/mujoco/index.html) based on the physics engine [mujoco](https://mujoco.org/) for faciliatating research and development in robotics, biomechanics, graphics and animation. The collection includes the following benchmarks: `BmAgentAnt`, `BmAgentCheetah`, `BmAgentHuman`, `BmAgentHumanStand`, `BmAgentLake`, `BmAgentLander`, `BmAgentPendInv`, `BmAgentPendInvDouble`, `BmAgentReacher`, `BmAgentSwimmer`.
    > Within the framework of this collection, explicit optimization of the entire set of actions (discrete or continuous) may be performed (if `direct` policy name is set) or discrete Toeplitz policy may be used (if `toeplitz` policy name is set; it is the default value for almost all agents); you can also pass your own custom policy as an instance of the correct class (see `teneva_bm/agent/policy.py` for details).

- `decomp` - a collection of low-rank tensor networks. The collection includes the following benchmarks: `BmDecompHt`, `BmDecompPeps`, `BmDecompTt`.

- `func` - a collection of analytic functions of a real multidimensional argument having an arbitrary dimension (default is `7`). The collection includes the following benchmarks: `BmFuncAckley`, `BmFuncAlpine`, `BmFuncChung`, `BmFuncDixon`, `BmFuncExp`, `BmFuncGriewank`, `BmFuncMichalewicz`, `BmFuncPathological`, `BmFuncPinter`, `BmFuncPowell`, `BmFuncQing`, `BmFuncRastrigin`, `BmFuncRosenbrock`, `BmFuncSchaffer`, `BmFuncSchwefel`, `BmFuncSalomon`, `BmFuncSphere`, `BmFuncSquares`, `BmFuncTrid`, `BmFuncTrigonometric`, `BmFuncWavy`, `BmFuncYang`.
    > For almost all functions, the exact global minimum ("continuous x point", not multi-index) is known (see `bm.x_min_real` and `bm.y_min_real`). For a number of functions (`BmFuncAlpine`, `BmFuncExp`, `BmFuncGriewank`, `BmFuncMichalewicz`, `BmFuncQing`, `BmFuncRastrigin`, `BmFuncRosenbrock`, `BmFuncSchwefel`), a `bm.build_cores()` method is available that returns an exact representation of the function on the discrete grid in the tensor train (TT) format as a list of 3D TT-cores. Note also that we apply small shift of the grid limits for all functions, to make the optimization problem more difficult (because many functions have a minimum at the center point of the domain).

- `func_fix` - a collection of analytic functions of a real multidimensional argument having the fixed dimension. The collection includes the following benchmarks: `BmFuncFixBiggs` (`d = 5`), `BmFuncFixCantrell` (`d = 4`), `BmFuncFixColville` (`d = 4`), `BmFuncFixDolan` (`d = 5`), `BmFuncFixPiston` (`d=7`).

- `hs` - the [Hock & Schittkowski](http://apmonitor.com/wiki/index.php/Apps/HockSchittkowski) collection of benchmark functions, containing continuous analytic functions of small dimensions (`2-5` for almost all problems), some of which have given constraints (`= 0` and/or `> 0`). The collection includes the following benchmarks: `BmHsFunc001`, `BmHsFunc002`, ..., `BmHsFunc128` (except `BmHsFunc082`, `BmHsFunc094`, `BmHsFunc115`, `BmHsFunc121`, `BmHsFunc123`, `BmHsFunc125` which do not exist).

- `odeoc` - a collection of optimal control problems described by ordinary differential equations (ODEs), some of the problems have explicit restrictions on the elements of the control vector. The collection includes the following benchmarks: `BmOdeocSimple`, `BmOdeocSimpleConstr`.

- `qubo` - a collection of quadratic unconstrained binary optimization (QUBO) problems having an arbitrary dimension (default is `100`); all benchmarks are discrete and have a mode size equals `2`. The collection includes the following benchmarks: `BmQuboKnapQuad`, `BmQuboMaxcut`, `BmQuboMvc`.

- `qubo_fix` - a collection of quadratic unconstrained binary optimization (QUBO) problems having the fixed dimension. The collection includes the following benchmarks: `BmQuboKnap10` (`d=10`), `BmQuboKnap20` (`d=20`), `BmQuboKnap50` (`d=50`), `BmQuboKnap80` (`d=80`), `BmQuboKnap100` (`d=100`).

- `various` - a collection of heterogeneous benchmarks that are not suitable for any other collection (note that in this case, we do not use the name of the collection in the name of the benchmarks, unlike all other sets). The collection includes the following benchmarks: `BmMatmul`, `BmTopopt` (draft!), `BmWallSimple`.

> Note that you can use the function `teneva_bm_get` to obtain a list of all benchmark classes. Also, this function supports various filters (for example, call `teneva_bm_get(is_func=True, is_opti_max=False)` will return all benchmarks that are continuous functions and relates to minimization task).


## Usage

A typical scenario for working with any benchmark from our package is as follows.

##### Benchmark initialization

First, we create an instance of the desired benchmark class and manually call the `prep` method (optionally, we can also print detailed information about the benchmark):
```python
import numpy as np
from teneva_bm import *

bm = BmFuncAckley()
bm.prep()
print(bm.info())
```

> We especially note that all imported classes from `teneva_bm` have the `Bm` prefix, that is, using an asterisk when importing will not pollute the namespace.

The class constructor of all benchmarks has the following optional arguments:

- `d` - dimension of the benchmark's input (non-negative integer). For some benchmarks, this number is hardcoded (or depends on other specified auxiliary arguments), if another value is explicitly passed, an error message will be generated (e.g., the dimension for benchmark `various.bm_matmul` is determined automatically by the values of auxiliary arguments `size`, `rank` and `only2`). By default, some correct value is used (specified in the benchmark description).

- `n` - number of possible discrete values for benchmark input variables, i.e., the mode size of the related tensor / multidimensional array (non-negative integer if all mode sizes are equal or a list of non-negative integers of the length `d`). For some benchmarks, this number is hardcoded (or depends on other specified auxiliary arguments), if another value is explicitly passed, an error message will be generated (e.g., in `qubo` collection all benchmarks should have `n = 2`). By default, some correct value is used (specified in the benchmark description).

- `seed`- the random seed (default is `42`). Note that we use `Random Generator` from numpy (i.e., `numpy.random.default_rng(seed)`) and for a fixed value of the seed, the behavior of the benchmark will always be the same, however, not all benchmarks depend on a seed.

- `name`- the display name for benchmark. By default, the class name without `Bm` prefix is used.

- `...other arguments...` - some benchmarks have additional optional arguments, which are described in the corresponding python files (and in the info text).

##### Setting advanced options

Before calling the `bm.prep()` method, you can optionally set a number of additional benchmark options:

- `bm.set_budget(m=None, m_cache=None, is_strict=True)` - optional method to set the computation buget `m`. If the number of requests to the benchmark (from calls to `bm.get` and `bm.get_poi` methods) exceeds the specified budget, then `None` will be returned. If the flag `is_strict` is disabled, then the request for the last batch will be allowed, after which the budget will be exceeded, otherwise this last batch will not be considered. Note that when the budget is exceeded, `None` will be returned both when requesting a single value and a batch of values. Also, in a similar way, you can set a limit on the use of the cache by `m_cache` parameter.

- `bm.set_constr(penalty=1.E+42, eps=1.E-16, with_amplitude=True)` - if the benchmark has a constraint, then using this function you can set `penalty` (for the requested points that do not satisfy the constraint, the value `penalty * constraint_value` will be returned) and `eps` (threshold value to check that the constraint has been fulfilled). Note that we set the constraints as a function (`bm.constr` / `bm.constr_batch`) that returns the value `constraint_value` (amplitude) of the constraint, and if the constraint is met, then the value must be non-positive, otherwise, the objective function is not calculated and a value proportional to the amplitude of the constraint is returned (if you disable flag `with_amplitude`, then just the value of the penalty will be returned). Note that for the case of maximization task you should set negative `penalty` value.

- `bm.set_grid_kind(kind='cheb')` - by default, we use the Chebyshev grid (`kind = 'cheb'`) for benchmarks corresponds to a function of a continuous argument, but you can alternatively set it manually to use a uniform grid (`kind = 'uni'`).

- `bm.set_max(i=None, x=None, y=None)` - if necessary, you can manually set the multi-index, the corresponding continuous point (for benchmarks, which relate to functions of a continuous argument), and the corresponding value for the exact global maximum of the function. The corresponding values will be further available in the benchmark as `bm.i_max_real`, `bm.x_max_real` and `bm.y_max_real` respectively. When the benchmark is initialized, this function is called automatically if the optimum is known.

- `bm.set_min(i=None, x=None, y=None)` - the same as in the previous point, but for the global minimum.

- `bm.set_log(log=False, cond=None, step=1000, prefix='bm', with_min=None, with_max=None)` - when calling this function with the `True` argument `log`, the log will be printed while requests to benchmark. You may set the log codition `cond` (`min`, `max`, `max-min` or `step`; e.g., in the case `min` the log will be presented each time the `min` value is updated), the log step (for condition `step`) and a string `prefix` for the log. You can also disable the display of current minimum values (`with_min`) or maximum values (`with_max`) in the log string. Note that you can provide as `log` argument some print-like function, e.g., `log=print`, in this case, printing will occur not to the console, but to the corresponding function. If `cond`, `with_min` and `with_max` are not set, then the logs will correspond to an improvement in the optimization result, taking into account the type of benchmark (max/min).

- `bm.set_cache(with_cache=False, cache=None, m_max=1.E+8)` - when calling this function with the `True` argument `with_cache`, the cache will be used, that is, all the values requested from the benchmark will be saved and when the same multi-indices are accessed again, the values will be retrieved from the cache instead of explicitly calculating the objective function. Additionally, you can optionally pass as an argument `cache` an already existing cache in the form of a dictionary (the keys are multi-indices in the form of tuples, and the values are the corresponding values of the objective function). We especially note that the cache is only used when querying benchmark values in discrete multi-indices; for requested continuous points, no cache will be used. It is also important to note that no cache will be used for matching multi-indices in the same requested batch of values. Optionally, you can set `m_max` argument that specifies the maximum cache size. If the size is exceeded, the cache will be cleared and a corresponding warning will be displayed in the log. Note that when the `bm.init` method is called, the cache is always reset to zero.

- `bm.set_opts(...)` - for some benchmarks, this function may be called to set additional benchmark-specific options (please see the names and description of options in the info text). Note that the arguments to this function must be named (e.g., `opt_a=42`).

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

Since the considered benchmark (`BmFuncAckley`) corresponds to a function of a continuous argument, above we calculated the values for the discretization of the function on an automatically selected grid (see `bm.set_grid` method). Additionally, we can calculate values at continuous points by analogy:
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

> Note that, by default, the spent computational budget (see `bm.set_budget` function) and time spent does not change when the test dataset is generated (this is controlled by the default `skip_process` flag value in `bm.build_tst(m=0, seed=None, skip_process=True)`), but is consumed when generating the training dataset (i.e., `bm.build_trn(m=0, seed=None, skip_process=False)`). Also note that if the seed is not given as an argument to these functions, then the global seed of the benchmark will be used (in this case, the same data sets will be generated when the function is called again).

##### Request history analysis

During requests to the benchmark, that is, when calling functions `bm[]` (or `bm.get`), `bm()` (or `bm.get_poi`), `bm.build_trn` (if the flag `skip_process` is not set in the function arguments; it has a value `False` by default) and `bm.build_tst` (if `skip_process` is not set; it is `True` by default for this function), the following useful class parameters are updated:

- `bm.m` - the total number of performed calculations of the benchmark value (if a cache is used, then the values taken from the cache are not taken into account in this variable).

- `bm.m_cache` - the total number of cache hits performed instead of explicitly calculating benchmark values (if no cache is used, it is `0`).

- `bm.time_calc` - total time in seconds spent on calculating the benchmark values (the time spent on cache accesses is also taken into account).

- `bm.time_full` - benchmark lifetime in seconds from the moment of initialization (i.e., the call to `bm.init` method, which is called automatically when creating a benchmark or can then be called manually to reset the query history).

- `bm.y_list` - a list of all sequentially calculated benchmark values (results of cache accesses are not added to the list).

- `bm.y_list_full` - a list of all sequentially calculated benchmark values including the cache requests.

- `bm.i_max`, `bm.x_max`, `bm.y_max` - a discrete multi-index, a continuous multi-dimensional point, and benchmark's value corresponding to the maximum of all requested values. Note that for the case of a discrete function, the value of `x_max` will be `None`, and for the case of a continuous function, the values of `i_max` and `x_max` will correlate, while if requests were made for continuous points, then `x_max` will correspond to the exact position of the point, and `i_max` will be the nearest multi-index of the used discrete grid.

- `bm.i_min`, `bm.x_min`, `bm.y_min` - same as in the previous point, but for the minimum value.

- `bm.i`, `bm.x`, `bm.y` - the last requested multi-index / point and the related computed benchmark's value.

> The function `print(bm.info_history())` may be used to print the corresponding values from the history of requests.

##### Extracting information from the benchmark

For various purposes, including saving calculation results to a file, the following dictionaries can be used:

- `bm.args` - the main benchmark arguments that are set during initialization (dimension, mode size, random seed, etc.);

- `bm.opts` - additional benchmark options (if available; set using function `set_opts`);

- `bm.prps` - various properties of the benchmark (function limits, computational budget, etc.);

- `bm.hist` - values associated with the history of requests to the benchmark (list of requested values, found optimum, etc.).

> You can get all these dictionaries within a single higher level dictionary with corresponding key names via property `bm.dict`.

##### Notes

- For some benchmarks (e.g., for all benchmarks from `agent` collection) the method `show` (save the `png` image with the current state, the state for provided input or the final state for the best found input) and `render` (save the `mp4` animation for the current solution, the solution from the provided input or for the solution from the best found input) are available.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)


## Contributors

- [Anastasia Batsheva](https://github.com/anabatsh)
- [Artem Basharin](https://github.com/a-wernon)

We are happy to invite you to become a contributor, especially if you have interesting benchmarks ;) Please see detailed instructions for developers in [workflow.md](https://github.com/AndreiChertkov/teneva_bm/blob/main/workflow.md) file.


---


> ✭__🚂  The stars that you give to **teneva_bm**, motivate us to develop faster and add new interesting features to the code 😃
