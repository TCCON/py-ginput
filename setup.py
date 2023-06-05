import os
from setuptools import setup, find_packages

_mydir = os.path.dirname(__file__)

setup(
    name='GInput',
    desciption='Python code that creates the .mod and .vmr files used in GGG',
    author='Joshua Laughner, Sebastien Roche, Matthaeus Kiel',
    author_email='jlaugh@caltech.edu',
    version='1.1.5e',  # make sure stays in sync with the version in ginput/__init__.py
    url='',
    install_requires=[
        'astropy>=3.1.2',
        'cfunits>=3.0.0',
        'ephem>=3.7.6.0',
        'h5py>=2.9.0',
        'jplephem>=2.9',
        'matplotlib>=3.0.3',
        'netcdf4>=1.4.2',
        'pandas>=0.24.2',
        'pydap>=3.2.2',
        'requests>=2.14.2',
        'scipy>=1.2.1',
        'sgp4>=1.4',
        'skyfield>=1.10',
        'xarray>=0.12.1'
    ],
    packages=find_packages(),
)
