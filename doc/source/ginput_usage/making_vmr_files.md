(usage-vmr)=
# Making VMR files

This section assumes that you have completed {ref}`usage-mod`.
Like that section, this will assume you want to use GEOS FP files, since those files are available without a data subscription.
However, please note that GEOS FP is _not_ the standard met source for TCCON and COCCON priors, and therefore `.mod`, `.vmr`, and `.map` file generated from GEOS FP are not suitable for standard data production for those networks.

## Creating .vmr files for dates

As with the previous steps, we will use the `run_ginput.py` script to call ginput.

To generate the actual `.vmr` files, we use the `vmr` subcommand.
We'll assume that we are going to generate the `.vmr` files corresponding to the `.mod` files we made in {ref}`usage-mod`, so:

- the files were output under `MOD_FILE_DIR`,
- we are generating the files for 1 Jan 2018,
- the site latitude and longitude are 36.604 N and 97.486 W,
- we used GEOS FP met data as input, and 
- we want to write the `.vmr` files to `VMR_FILE_DIR`.

Given these assumptions, our command is:

```
$ ./run_ginput.py vmr \
    --product fp \
    --lon -97.486 \
    --lat 36.604 \
    --base-vmr-file ginput/testing/test_input_data/summer_35N.vmr \
    --integral-file ginput/testing/test_input_data/ap_51_level_0_to_70km.gnd \
    --save-path VMR_FILE_DIR \
    --auto-update-fo2-file \
    20180101 \
    MOD_FILE_DIR/fp/xx/vertical
```

```{note}
Remember, if you installed with conda or pip, you probably want to use `ginput_cli`
instead of `./run_ginput.py`. 
```

```{note}
ginput v1.3 added a requirement for O2 mole fraction data when generating the `.vmr` files.
For v1.3, you would have to execute the command `./run_ginput update_fo2` once before creating any `.vmr` files.
From v1.3.1 on, you can use the `--auto-update-fo2-file` flag for the `vmr` subcommand as shown below.
If you are using a version of ginput before 1.3, you will not have the `update_fo2` subcommand nor the `--auto-update-fo2-file` flag, and ginput will not add the O2 mole fractions to the `.vmr` file headers.
```

This part of ginput requires several look up tables to be pre-calculated.
The first time you run it, it will calculate them and save them as netCDF files.
On successive runs, it will load the netCDF files unless any of the inputs or code needed to make those files have changed since the last run.
Thus, your first run will be slow, but later runs will be significantly faster.

Some things to take note of:

1. We do not need to specify altitude here, only latitude and longitude.
2. We do need to indicate that we used GEOS FP, now with the `--product` option instead of `--mode`. However, we do not need to indicate that we used `eta` level files, so unlike the previous step, the value here is only "fp", not "fp-eta".
3. We point to two files under `ginput/testing/test_input_data`. If you have GGG installed, it is recommended to point to the versions of these files that come with GGG. `summer_35N.vmr` is under `$GGGPATH/vmrs/gnd` and `ap_51_level_0_to_70km.gnd` is under `$GGGPATH/levels`.

Like with the model files, we will get a tree output directory like so:

```
VMR_FILE_DIR
└── fp
    └── xx
        └── vmrs-vertical
            ├── JL1_2018010100Z_37N_097W.vmr
            ├── JL1_2018010103Z_37N_097W.vmr
            ├── JL1_2018010106Z_37N_097W.vmr
            ├── JL1_2018010109Z_37N_097W.vmr
            ├── JL1_2018010112Z_37N_097W.vmr
            ├── JL1_2018010115Z_37N_097W.vmr
            ├── JL1_2018010118Z_37N_097W.vmr
            └── JL1_2018010121Z_37N_097W.vmr
```

See {ref}`usage-mod` for an explanation of the meaning of each level.

## Producing .vmr files for a runlog

As with the `.mod` files, ginput can generate the specific `.vmr` files needed for a GGG runlog.
(If you don't know what a runlog is, you can skip this section.)
The following example shows how to create the `.vmr` files for the Park Falls benchmark runlog included with GGG.
Like with the `.mod` files, this example uses GEOS FP-IT data, because GEOS FP does not cover the year needed by this runlog.

```
$ ./run_ginput.py rlvmr \
    --base-vmr-file ginput/testing/test_input_data/summer_35N.vmr \
    --integral-file ginput/testing/test_input_data/ap_51_level_0_to_70km.gnd \
    --product fpit \
    --mod-root-dir MOD_FILE_DIR \
    --save-path VMR_FILE_DIR \
    --auto-update-fo2-file \
    $gggpath/runlogs/gnd/pa_ggg_benchmark.grl
```

Most of the options are the same as the example above where we specific the dates to generate.
However, notice that instead of giving the full path to the `.mod` files as the second positional argument, we use the `--mod-root-dir` option and point it to the top directory that the `.mod` files were output to.
This directory must have the structure `met/site/vertical`.
For our example, it would be:

```
MOD_FILE_DIR
└── fpit
    └── pa
        └── vertical
            ├── FPIT_2004072121Z_46N_090W.mod
            ├── FPIT_2004072200Z_46N_090W.mod
            └── FPIT_2004122215Z_46N_090W.mod
```

The reason we recommend using `--mod-root-dir` in this case is that, if your runlog contains multiple sites, this allows ginput to automatically select the correct subdirectory, assuming that your `.mod` files are organized in the above structure.
On the other hand, if all of your `.mod` files are in one directory (even from different sites), then passing the path to that directory as the second positional argument and omitting the `--mod-root-dir` option instead will handle that organization.
