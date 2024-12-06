import os
from setuptools import setup, find_packages

_mydir = os.path.dirname(__file__)
# Take the contents of the README file as the long description for PyPI
with open(os.path.join(_mydir, 'README.md')) as f:
    _readme = f.read()

setup(
    name='ginput',
    description='Python code that creates the .mod and .vmr files used in GGG',
    long_description=_readme,
    long_description_content_type='text/markdown',
    author='Joshua Laughner, Sebastien Roche, Matthaeus Kiel',
    author_email='jlaugh@caltech.edu',
    version='1.3.0',  # make sure stays in sync with the version in ginput/__init__.py
    url='',
    install_requires=[
        'astropy>=3.1.2',
        'cfunits>=3.3.2',
        'ephem>=3.7.6.0',
        'h5py>=2.9.0',
        'jplephem>=2.9',
        'matplotlib>=3.0.3',
        'netCDF4>=1.4.2',
        'pandas>=0.24.2',
        'pydap>=3.2.2',
        'python-dateutils>=2.8.2',
        'requests>=2.14.2',
        'scipy>=1.2.1',
        'sgp4>=1.4',
        'skyfield>=1.10',
        'xarray>=0.12.1',
    ],
    packages=find_packages(),
    include_package_data=True,
)
