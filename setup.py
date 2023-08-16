import os
import re
from setuptools import setup


def find_packages(package, basepath):
    packages = [package]
    for name in os.listdir(basepath):
        path = os.path.join(basepath, name)
        if not os.path.isdir(path):
            continue
        packages.extend(find_packages('%s.%s'%(package, name), path))
    return packages


here = os.path.abspath(os.path.dirname(__file__))


desc = 'Benchmarks library, based on the package teneva, for testing multivariate approximation and optimization methods'
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    desc_long = f.read()


with open(os.path.join(here, 'teneva_bm/__init__.py'), encoding='utf-8') as f:
    text = f.read()
    version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", text, re.M)
    version = version.group(1)


with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().split('\n')
    requirements = [r for r in requirements if len(r) >= 3]


setup_args = dict(
    name='teneva_bm',
    version=version,
    description=desc,
    long_description=desc_long,
    long_description_content_type='text/markdown',
    author='Andrei Chertkov',
    author_email='andre.chertkov@gmail.com',
    url='https://github.com/AndreiChertkov/teneva_bm',
    classifiers=[
        'Development Status :: 4 - Beta', # 3 - Alpha, 5 - Production/Stable
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='benchmarks approximation optimization multidimensional array multivariate function low-rank representation tensor train format TT-decomposition',
    packages=find_packages('teneva_bm', './teneva_bm/'),
    python_requires='>=3.8',
    project_urls={
        'Source': 'https://github.com/AndreiChertkov/teneva_bm',
    },
    license='MIT',
    license_files =('LICENSE.txt',),
)


if __name__ == '__main__':
    setup(
        **setup_args,
        install_requires=requirements,
        include_package_data=True)
