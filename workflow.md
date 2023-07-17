# workflow

> Workflow instructions for `teneva_bm` developers.


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

4. Install teneva_bm from the source:
    ```bash
    python setup.py install
    ```

5. Install dependencies for all benchmarks (see instructions in README.md)

6. Reinstall teneva_bm from the source after updates of the code:
    ```bash
    clear && pip uninstall teneva_bm -y && python setup.py install
    ```

7. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name teneva_bm --all -y
    ```


## How to add a new benchmark

1. Create python script in the appropriate subfolder of `teneva_bm` folder with the name like `bm_<subfolder>_<name>.py`, where `<subfolder>` is a name of the collection (e.g., `func`, `oc`, `qubo`, etc.) and `<name>` is a lowercase name of the benchmark.

2. Prepare a benchmark class `Bm<Subfolder><Name>` (class names are in camel case notation) in the created python file and then write a demo example of its usage in the bottom section `if __name__ == '__main__':`, by analogy with other benchmarks.

3. Run the demo example for the new benchmark (note that we should reinstall our library from the source to try the new benchmark):
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python demo.py bm_<subfolder>_<name>
    ```
    > This script will run the demo (from section `if __name__ == '__main__':`) for the benchmark specified as an argument. If the argument is not provided, then the examples for all benchmarks will be run sequentially.


## How to update the package version

1. Reinstall the package and run the demo script for all benchmarks:
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python demo.py
    ```

2. Update version (like `0.1.X`) in `teneva_bm/__init__.py` and `README.md`

3. Do commit `Update version (0.1.X)` and push

4. Upload new version to `pypi` (login: AndreiChertkov):
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

5. Reinstall and check that installed version is new:
    ```bash
    pip install --no-cache-dir --upgrade teneva_bm
    ```
