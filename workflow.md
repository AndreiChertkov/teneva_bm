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
    pip install protes==0.3.4
    ```

8. Reinstall `teneva_bm` from the source (after updates of the code):
    ```bash
    clear && pip uninstall teneva_bm -y && python setup.py install
    ```

9. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name teneva_bm --all -y
    ```


## How to add a new benchmark

1. Create python script in the appropriate subfolder of `teneva_bm` folder with the name like `bm_<subfolder>_<name>.py`, where `<subfolder>` is a name of the collection (e.g., `func`, `qubo`) and `<name>` is a lowercase name of the benchmark (e.g., `ackley`, `knap_det`).

2. Prepare a benchmark class `Bm<Subfolder><Name>` (class names should be in the camel case notation) in the created python file and then write a demo example of its usage (initialization, get method, training dataset generation, etc.; please, do it by analogy with other benchmarks) in the bottom section after `if __name__ == '__main__':`. Please, note:
    - We should necessarily rewrite the method `target` and / or `target_batch` of the parent class (calculating a benchmark's value for a given multidimensional index or point);
    - We should necessarily rewrite the property `is_func` or `is_tens` (flag indicating whether the benchmark is a continuous or discrete function);
    - If the objective function has constraint, we should specify the function `constr` and / or `constr_batch`, also we should specify the value `True` for property `with_constr`;
    - Method `cores` can be specified to generate an exact tensor train (TT) representation of the benchmark, in which case the property `with_cores` should be set to `True`.

3. Run the demo example for the new benchmark (note that we should reinstall our library from the source to try the new benchmark):
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python demo.py bm_<subfolder>_<name>
    ```
    > This script will run the demo (from the section `if __name__ == '__main__':`) for the benchmark specified as an argument. If the argument is not provided, then the examples for all benchmarks from all collections will be run sequentially. Note that if you have not changed the core of the library, then you can run the new benchmark as a normal python file, i.e., `python bm_<subfolder>_<name>.py`.

4. Add a description of the new benchmark to section `Available benchmarks`  of the `README.md` file.

5. Use the new benchmark locally until the next library version update.

> Please use underscore prefixes for all new class instance variables (except the names of the benchmarks' options, i.e., the args in `__init__` or `bm.set_opts` method; for options, please make sure their names do not conflict with base class variable names) and functions created in the benchmark (e.g., `_env`) so that there is no name conflict with the base class `Bm`.


## How to update the base class Bm

Modifying this class may break the functionality of all benchmarks, so please do so with care!


## How to update the package version

1. Reinstall the package locally and run the demo script for all benchmarks:
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python demo.py
    ```

2. Update the package version (like `0.7.X`) in `teneva_bm/__init__.py` and `README.md` files, where `X` is a new subversion number;

3. Do commit like `Update version (0.7.X)` and push;

4. Upload the new version to `pypi` (login: AndreiChertkov):
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

5. Reinstall the package from `pypi` and check that installed version is new:
    ```bash
    pip uninstall teneva_bm -y && pip install --no-cache-dir --upgrade teneva_bm
    ```
