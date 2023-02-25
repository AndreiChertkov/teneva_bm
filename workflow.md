# workflow: teneva_bm


## How to install the current local version

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

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

5. Install basic dependencies:
    ```bash
    pip install numpy==1.22.1 scipy==1.8.1 torch==1.13.1 "jax[cpu]"==0.4.3
    ```

6. Install dependencies for all benchmarks:
    ```bash
    pip install TODO
    ```

7. Install `teneva_bm`:
    ```bash
    python setup.py install
    ```

8. Reinstall `teneva_bm` (after updates of the code):
    ```bash
    clear && pip uninstall teneva_bm -y && python setup.py install
    ```

9. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name teneva_bm --all -y
    ```


## How to update the package version

1. Update version (like `0.1.X`) in the file `teneva_bm/__init__.py`

    > For breaking changes we should increase the major index (`1`), for non-breaking changes we should increase the minor index (`X`)

2. Do commit `Update version (0.1.X)` and push

3. Upload new version to `pypi` (login: AndreiChertkov; passw: xxx)
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

4. Reinstall
    ```bash
    pip install --no-cache-dir --upgrade teneva_bm
    ```
