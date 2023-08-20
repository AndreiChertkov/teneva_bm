# workflow

> Workflow instructions for `teneva_bm` developers.


## How to install the current local version

1. Install [anaconda](https://www.anaconda.com) package manager with [python](https://www.python.org) (version 3.8);

2. Create a virtual environment:
    ```bash
    conda create --name teneva_bm python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate teneva_bm
    ```

4. Install special dependencies (for developers only):
    ```bash
    pip install jupyterlab twine
    ```

5. Install `teneva_bm` from the source:
    ```bash
    python setup.py install
    ```

6. Install dependencies for all benchmarks:
    ```bash
    python install_all.py --env teneva_bm --silent
    ```

7. Install dependency to run benchmark optimization examples:
    ```bash
    pip install protes==0.3.5
    ```

8. Reinstall `teneva_bm` from the source (after updates of the code):
    ```bash
    clear && pip uninstall teneva_bm -y && python setup.py install
    ```

9. Run all the tests:
    ```bash
    clear && python test_all.py && python test_ref.py
    ```
    > Note that, unfortunately, for some benchmarks (especially for the `agent` collection), the calculation result depends on the computing device used, so tests for checking the match in the reference value, i.e. `test_ref.py`, may fail.

10. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name teneva_bm --all -y
    ```


## How to add a new benchmark

1. Create python script in the appropriate subfolder of `teneva_bm` folder with the name like `bm_<subfolder>_<name>.py`, where `<subfolder>` is a name of the collection (e.g., `func`, `qubo`) and `<name>` is a lowercase name of the benchmark (e.g., `ackley`, `mvc`).
    > Note that we do not use the `<subfolder>` in the name of benchmarks (only) for collection `various`.

2. Prepare a benchmark class `Bm<Subfolder><Name>` (class names should be in the camel case notation) inheriting from base class `Bm` in the created python file. Please, note:
    - We **should** rewrite the method `target` and / or `target_batch` of the parent class (calculating a benchmark's value for a given multidimensional index or point);
    - We **should** rewrite the property `is_func` or `is_tens` (flag indicating whether the benchmark is a continuous or discrete function);
    - We **should** rewrite the property `ref`, which returns the reference (random) multi-index and related value of the benchmark;
    - If the objective function has a constraint, we should specify the function `constr` and / or `constr_batch`, also we should specify the value `True` for the property `with_constr`;
    - Method `cores` can be specified to generate an exact tensor train (TT) representation of the benchmark, in which case the property `with_cores` should be set to `True`;
    - If the benchmark has auxiliary main arguments (which are specified when the class is initialized), then they should be described using property `args_info`. Please see the base class `Bm` as an example (note that the default arguments are dimension `d`, mode size `n` and random seed `seed`);
    - If the benchmark has auxiliary options, then they can be described using property `opts_info`. Please see benchmark `BmFuncAckley` as an example;
    - If the benchmark is for a maximization (rather than minimization) problem, then please rewrite property `is_opti_max` with a return value of `True`;
    - Also, if necessary, rewrite property `identity`, which returns a list of argument names (from among those used when initializing the class) that fully define the benchmark (by default, this is the dimension `d` and the mode size `n`; for agents, this is the number of agent steps, the type of policy used, etc.). The corresponding values will be used when saving the calculation results to a file, etc.

3. Add import line of the new benchmark into `__init__.py` file in the collection's folder and also append it into the function `teneva_bm_get_<subfolder>` in this file (additions are recommended to be done in alphabetical order).

4. Run tests for all benchmarks (note that we should reinstall our library from the source to try the new benchmark):
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python test_all.py && python test_ref.py
    ```
    > Note that if you have not changed the core of the library, then you can also run (try) the new benchmark as a normal python file, i.e., `python bm_<subfolder>_<name>.py` (without reinstalling the library).

5. Add a description of the new benchmark to the section `Available benchmarks` of the `README.md` file (additions are recommended to be done in alphabetical order).

6. Use the new benchmark locally until the next library version update on pypi.

> Please use underscore prefixes for all new class instance variables and functions (e.g., `_env`) to avoid the name conflict with the base class `Bm`. However, underscores do not need to be used for the benchmarks' arguments and options, i.e., the parameters in `__init__` (see also `args_info`) and `bm.set_opts` (see also `opts_info`) method, but please make sure their names do not conflict with the base class variable names.


## How to update the base class Bm

Modifying this class may break the functionality of all benchmarks, so please do it with care!


## How to update the package version

1. Reinstall the package locally and run the tests for all benchmarks:
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python test_all.py && python test_ref.py
    ```

2. Reinstall the package locally and run the demo scripts:
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python demo/base_func.py && python demo/base_agent.py && python demo/opti_base.py
    ```

3. Update the package version (like `0.8.X`) in `teneva_bm/__init__.py` and `README.md` files, where `X` is a new subversion minor number (if major number changes, then update it also here and in the next point);

4. Do commit like `Update version (0.8.X)` and push;

5. Upload the new version to `pypi` (login: AndreiChertkov):
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

6. Reinstall the package from `pypi` and check that the installed version is new:
    ```bash
    pip uninstall teneva_bm -y && pip install --no-cache-dir --upgrade teneva_bm
    ```

7. Update the version in [colab notebook](https://colab.research.google.com/drive/1z8LgqEARJziKub2dVB65CHkhcboc-fCH?usp=sharing) and run all its cells.
