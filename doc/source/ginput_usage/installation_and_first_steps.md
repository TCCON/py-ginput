# Installation and first steps

## Install and use as a global tool

The recommended way to install is using [pixi](https://pixi.sh/latest/):

```
pixi global install ginput
```

If you prefer `uv`, it can be installed with:

```
uv tool install ginput
```

The `ginput_cli` command will then be available globally.


## Install as a dependency


### Using pixi

If you use [pixi](https://pixi.sh/latest/), it can be added to your
environment with:

```
pixi add ginput
```

You can then run it from the environment's directory with

```
pixi run ginput_cli
```

### Using conda or similar

If you use the conda directly, and assuming you have the target environment active, install with:

```
conda install -c conda-forge ginput
```

This should also work using `mamba` or `micromamba` instead of `conda`.

To run, you would activate the environment and call `ginput_cli`.
Alternatively, use the `run` subcommand of `conda`, `mamba`, or `micromamba`,
point to the correct environment with the `-n` or `-p` flags, and give `ginput_cli`
as the command to execute.


### Using uv

If you use `uv`, then you can add it to an existing project with:

```
uv add ginput
```

Similar to `pixi`, you can then run the CLI with `uv run ginput_cli` from the project directory.


### Using pip

If you prefer `pip`, first, activate the virtual environment that you want to
install into.
Then, run:

```
pip install ginput
```

To use the CLI, either activate the environment and call `ginput_cli` or use the absolute path,
e.g. `/home/user/ginput-env/.venv/bin/ginput_cli`.

## Install from source

We support installation with [pixi](https://pixi.sh/latest/), [uv](https://docs.astral.sh/uv/), or
`pip`.
`pixi` is recommended, as it handles dependencies like the netCDF and HDF5 libraries best.

Using any of these tools, first download or clone `ginput` to your computer and `cd` into the
top directory of the repo (the one with the README file).
To install with `pixi`, do:

```
pixi install --locked
```

To install with `uv`, do:

```
uv sync
```

To install with `pip`, activate the destination virtual environment, then install with:

```
pip install --editable .
```

All of these methods will create the `ginput_cli` entry script in the environment's `bin` directory.
As above, if using `pixi` or `uv`, you can run from the ginput directory with `pixi run ginput_cli`
or `uv run ginput_cli`. 
If you installed with `pip`, you can activate the environment and call `ginput_cli` directly.
For all methods, you may also point to the absolute path of the `ginput_cli` script to run it.


## First steps before running

If you want to use `ginput` to create `.vmr` files, then you will need to first run the `update_fo2`
subcommand as in:

```
ginput_cli update_fo2
```

This will download data to calculate the annual mean oxygen dry mole fraction.
Starting with ginput version 1.3.0, this value will be included in the `.vmr` file headers,
so that GGG2020.1 and later can account for the time trend of oxygen in the atmosphere.
