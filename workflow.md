# workflow

> Workflow instructions for `teneva_bm` developers.


## How to install the current local version

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager)

2. Create a virtual environment:
    ```bash
    conda create --name teneva_bm python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate teneva_bm
    ```

4. Install special dependencies (for developers):
    ```bash
    pip install sphinx twine jupyterlab
    ```

5. Install teneva_bm:
    ```bash
    python setup.py install
    ```

6. Install dependencies for all benchmarks (see README.md)

7. Reinstall teneva_bm (after updates of the code):
    ```bash
    clear && pip uninstall teneva_bm -y && python setup.py install
    ```

8. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name teneva_bm --all -y
    ```


## How to update the package version

1. Reinstall the package and run the demo script:
    ```bash
    pip uninstall teneva_bm -y && python setup.py install && clear && python demo.py
    ```

2. Update version (like `0.1.X`) in `teneva_bm/__init__.py` and `README.md`

3. Do commit `Update version (0.1.X)` and push

4. Upload new version to `pypi` (login: AndreiChertkov)
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

5. Reinstall and check that installed version is new
    ```bash
    pip install --no-cache-dir --upgrade teneva_bm
    ```
