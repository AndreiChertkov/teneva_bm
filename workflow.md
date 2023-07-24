# workflow

> Workflow instructions for `teneva_bm` developers


## How to install the current local version

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager)

2. Create and activate a virtual environment:
    ```bash
    conda create --name teneva_bm python=3.8 -y && conda activate teneva_bm
    ```

3. Install special dependencies (for developers only):
    ```bash
    pip install jupyterlab twine
    ```

4. Install `teneva_bm` from the source:
    ```bash
    python setup.py install
    ```

5. Install dependencies for all benchmarks (see instructions in the section `Installation` in `README.md` file)

6. Reinstall `teneva_bm` from the source (after updates of the code):
    ```bash
    clear && pip uninstall teneva_bm -y && python setup.py install
    ```

7. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name teneva_bm --all -y
    ```


## How to add a new benchmark

1. Create python script in the appropriate subfolder of `teneva_bm` folder with the name like `bm_<subfolder>_<name>.py`, where `<subfolder>` is a name of the collection (e.g., `func`, `qubo`) and `<name>` is a lowercase name of the benchmark (e.g., `ackley`, `knap_det`)

2. Prepare a benchmark class `Bm<Subfolder><Name>` (class names should be in the camel case notation) in the created python file and then write a demo example of its usage (initialization, get method, training dataset generation, etc.; please, do it by analogy with other benchmarks) in the bottom section after `if __name__ == '__main__':`. Please, note:
    - We should necessarily rewrite the method `_f` and / or `_f_batch` of the parent class (calculating a benchmark value for a given multidimensional index or point)
    - We should necessarily rewrite the property `is_func` or `is_tens` (flag indicating whether the benchmark is a continuous or discrete function)
    - If the objective function has constraint, we should specify the function `_c` and / or `_c_batch`, also we should specify the value `True` for property `with_constr`
    - Method `_cores` can be specified to generate an exact TT-representation of the benchmark, in which case the property `with_cores` should be set to `True`

3. Run the demo example for the new benchmark (note that we should reinstall our library from the source to try the new benchmark):
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python demo.py bm_<subfolder>_<name>
    ```
    > This script will run the demo (from the section `if __name__ == '__main__':`) for the benchmark specified as an argument. If the argument is not provided, then the examples for all benchmarks from all collections will be run sequentially

4. Add a description of the new benchmark to section `Available benchmarks`  of file `README.md`

5. Update the package version

> Please use prefixes for all class instance variables entered in the benchmark so that there is no name conflict with the base class Bm, e.g., `opt_` (in `set_opts` methods) and `bm_` (for other custom variables)


## How to update the base class Bm

Modifying this class may break the functionality of all benchmarks, so please do so with care!


## How to update the package version

1. Reinstall the package and run the demo script for all benchmarks:
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python demo.py
    ```

2. Update version (like `0.3.X`) in `teneva_bm/__init__.py` and `README.md` files, where `X` is a new subversion number

3. Do commit like `Update version (0.3.X)` and push

4. Upload the new version to `pypi` (login: AndreiChertkov):
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

5. Reinstall the package from `pypi` and check that installed version is new:
    ```bash
    pip uninstall teneva_bm -y && pip install --no-cache-dir --upgrade teneva_bm
    ```
