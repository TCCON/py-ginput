(usage-vmr)=
# Making VMR files

This section assumes that you have completed {ref}`usage-mod`.
Like that section, this will assume you want to use GEOS FP files, since those files are available without a data subscription.
However, please note that GEOS FP is _not_ the standard met source for TCCON and COCCON priors, and therefore `.mod`, `.vmr`, and `.map` file generated from GEOS FP are not suitable for standard data production for those networks.

## Creating .vmr files for dates

As with the previous steps, we will use the `run_ginput.py` script to call ginput.
Before we can generate our `.vmr` files, we need to download one other set of input data, needed to calculate the global mean O2 mole fraction.
To do this, simply run `./run_ginput.py update_fo2`; this will download the necessary data and write a file containing the mean O2 mole fractions calculated at `ginput/data/o2_mean_dmf.dat`.
This command only needs run about once per year to update the data, but since it's pretty quick, running it more ofter (once a month or once a week even) is not a problem.

```{note}
The `update_fo2` subcommand was added in ginput v1.3.
If you are using an older version of ginput, you will not have that subcommand, and ginput will not add the O2 mole fractions to the `.vmr` file headers.
```

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
    20180101 \
    MOD_FILE_DIR/fp/xx/vertical
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
