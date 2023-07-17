# teneva_bm


## Description

Benchmarks library, based on the software product [teneva](https://github.com/AndreiChertkov/teneva), for tensor based multidimensional approximation and optimization methods. Our benchmarks include both multidimensional data arrays and discretized functions of many variables.


## Installation

> Current version "0.1.2".

The package can be installed via pip: `pip install teneva_bm` (it requires the [Python](https://www.python.org) programming language of the version >= 3.8). It can be also downloaded from the repository [teneva_bm](https://github.com/AndreiChertkov/teneva_bm) and installed by `python setup.py install` command from the root folder of the project.

> Required python packages [matplotlib](https://matplotlib.org/) (3.7.0+) and [teneva](https://github.com/AndreiChertkov/teneva) (0.14.2+) will be automatically installed during the installation of the main software product.

Some benchmarks require additional installation of specialized libraries. The corresponding instructions are given in the description of each benchmark (see `DESC` string in the corresponding python files). Installation of all required libraries for all benchmarks can be done with the following command:

```bash
pip install networkx==3.0 qubogen==0.1.1 gekko==1.0.6
```


## Documentation and examples

All benchmarks inherit from the `Bm` base class (`teneva_bm/bm.py`) and are located in the subfolders (collections of benchmarks) of `teneva_bm` folder. The corresponding python files contain a detailed description of the benchmarks, as well as a scripts for a demo run at the end of the files. You can run demos for all benchmarks at once with the command `python demo.py` from the root folder of the project (you can also specify the name of the benchmark as a script argument to run the demo for only one benchmark, for example, `python demo.py bm_qubo_knap_amba`).

A typical scenario for working with a benchmark is as follows:
```python
import numpy as np
from teneva_bm import *
np.random.seed(42)

# Prepare benchmark and print info:
bm = BmQuboMaxcut().prep()
print(bm.info())

# Get value at multi-index i:
i = np.ones(bm.d)
print(bm[i]) # you can use the alias "bm.get"

# Get values for batch of multi-indices I:
I = np.array([i, i, i])
print(bm[I]) # you can use the alias "bm.get"

# Generate random train dataset:
I_trn, y_trn = bm.build_trn(1000)
```


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)

Please see detailed instructions for developers in `workflow.md`.

---

> ✭__🚂  The stars that you give to **teneva_bm**, motivate us to develop faster and add new interesting features to the code 😃
