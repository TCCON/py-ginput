# Installation and first steps

## Standard installation

First, download or clone `ginput` to your computer and `cd` into the top directory of the repo (the one with the README file).
If you have GNU Make and `conda` installed on your computer, then running `make install` will create the default conda environment
for the `ginput` dependencies, install them, and create the `run_ginput.py` script you can use to run any part of `ginput`.
Alternatively, if you do not have GNU Make or another program that can run Makefiles, you can call the `install.sh` script
yourself and provide the environment name to use as the only argument.
The following command will mimic the standard installation:

```
./install.sh ginput-auto-default
```

### How to get conda

We now recommend [Miniforge](https://github.com/conda-forge/miniforge) as the method to install conda, as Anaconda has been vigorous
about enforcing license requirements for their Python distributions.
Other Python installers that provide conda, including the standard Anaconda package and Miniconda, should work as well.

## Manual installation

If you need to use an alternate package manager (such as `mamba` or `micromamba`), you can use the provided `environment-py310.yml`
file to create and environment yourself.
For instance, to use micromamba to create an environment in the top directory of the repo, you can do:

```
micromamba create --prefix ./.mambaenv --file environment-py310.yml
```

Once this completes, activate the new environment and run the `install-runscript.sh` script.
This will create the `run_ginput.py` script that you can use to run any part of `ginput`.
It will insert a shebang at the top of that script that points to the Python executable for the
environment active when you run `install-runscript.sh`, so that you _do not_ need to activate this
environment to use ginput - simply calling `run_ginput.py` will ensure that environment is used.

## First steps before running

If you want to use `ginput` to create `.vmr` files, then you will need to first run the `update_fo2`
subcommand as in:

```
./run_ginput.py update_fo2
```

This will download data to calculate the annual mean oxygen dry mole fraction.
Starting with ginput version 1.3.0, this value will be included in the `.vmr` file headers,
so that GGG2020.1 and later can account for the time trend of oxygen in the atmosphere.
