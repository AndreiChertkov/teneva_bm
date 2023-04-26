# teneva_bm


## Description

Benchmarks library, based on a software product [teneva](https://github.com/AndreiChertkov/teneva), for tensor based multidimensional approximation and optimization methods. Benchmarks include both multidimensional data arrays and discretized functions of many variables.


## Installation

> Current version "0.1.0".

The package can be installed via pip: `pip install teneva_bm` (it requires the [Python](https://www.python.org) programming language of the version >= 3.8). It can be also downloaded from the repository [teneva_bm](https://github.com/AndreiChertkov/teneva_bm) and installed by `python setup.py install` command from the root folder of the project.

> Required python packages [matplotlib](https://matplotlib.org/) (3.7.0+) and [teneva](https://github.com/AndreiChertkov/teneva) (0.14.0+) will be automatically installed during the installation of the main software product.

Some benchmarks require additional installation of specialized libraries. The corresponding instructions are given in the description of each benchmark (see `DESC` string in the corresponding python files). Installation of all required libraries for all benchmarks can be done with the following command:

```bash
pip install networkx==3.0 qubogen==0.1.1 gekko==1.0.6
```


## Documentation and examples

All benchmarks inherit from the `Bm` base class (`teneva_bm/bm.py`) and are located in the `teneva_bm` folder. The corresponding python files contain a detailed description of the benchmarks, as well as a script for a demo run at the end of the file. You can run demos for all benchmarks at once with the command `python demo.py` from the root folder of the project.

A typical scenario for working with a benchmark is as follows:
```python
import numpy as np
from teneva_bm import *
np.random.seed(42)

# Prepare benchmark and print info:
bm = BmQuboMaxCut().prep()
print(bm.info())

# Get value at multi-index i:
i = np.ones(bm.d)
print(bm[i])

# Get values for batch of multi-indices I:
I = np.array([i, i, i])
print(bm[I])

# Generate random train dataset:
I_trn, y_trn = bm.build_trn(1000)
```


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)

> ✭__🚂  The stars that you give to **teneva_bm**, motivate us to develop faster and add new interesting features to the code 😃
