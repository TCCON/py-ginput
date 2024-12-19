(usage-mod)=
# Making model files

This section assumes that you have downloaded at least some GEOS files following the instructions in {ref}`usage-dl-met`.
Like that section, this will assume you want to use GEOS FP files, since those files are available without a data subscription.
However, please note that GEOS FP is _not_ the standard met source for TCCON and COCCON priors, and therefore `.mod`, `.vmr`, and `.map` file generated from GEOS FP are not suitable for standard data production for those networks.

## Creating .mod files for dates

Most users will want to generate `.mod` files for specific date ranges.
You must have the required GEOS data downloaded for those dates, see {ref}`usage-dl-met` for how to do so.

As with downloading met files, we will use the `run_ginput.py` script to call ginput.
In this case, the subcommand we want is `mod`, since we are generating nonstandard priors from GEOS FP.
(If we were using GEOS FP-IT, we could use the `tccon-mod` subcommand instead, which sets some of the options for us.)

Like in the {ref}`usage-dl-met` section, we'll assume that our GEOS met data is under the `GEOS_MET_DIR` directory and the chemistry GEOS files are under `GEOS_CHEM_DIR`.
We'll now add the assumption that we want to output the model files to `MOD_FILE_DIR`.
Additionally, we'll assume that we want to generate data for a custom location at 36.604 N, 97.486 W which is 200 m (0.2 km) above sea level.
With those assumptions, we can generate model files like so:

```
$ ./run_ginput mod \
    --alt 0.2 \
    --lon -97.486 \
    --lat 36.604 \
    --chem-path GEOS_CHEM_DIR \
    --include-chem \
    --save-path MOD_FILE_DIR \
    --mode fp-eta \
    20180101 \
    GEOS_MET_DIR
```

This will take a few minutes to run.
Once complete, you will see a directory tree in your `GEOS_MET_DIR` like so:

```
GEOS_MET_DIR
└── fp
    └── xx
        └── vertical
            ├── FP_2018010100Z_37N_097W.mod
            ├── FP_2018010103Z_37N_097W.mod
            ├── FP_2018010106Z_37N_097W.mod
            ├── FP_2018010109Z_37N_097W.mod
            ├── FP_2018010112Z_37N_097W.mod
            ├── FP_2018010115Z_37N_097W.mod
            ├── FP_2018010118Z_37N_097W.mod
            └── FP_2018010121Z_37N_097W.mod
```

The top subdirectory reflects the input met product used.
Since we had `--mode fp-eta`, our top directory is `fp`.

The next level down is the site ID.
This will be two letters, and since we did a custom location (and not a standard TCCON site), this will always be `xx`.

The third level will always be `vertical`.
This reflects that the profiles are for a vertical location directly over the given lat/lon, rather than along a slant path towards the sun.
(Support for slant paths is not currently maintained; future support for that is being considered.)

### Common variations on this command

If you want to generate files for a whole range of dates, you would change the second to last line to include two dates separated by a dash.
For example, to run for all of January 2018, you would give 20180101-20180201 as that first positional argument.
Note that the end date is exclusive, so it would not generate Feb 1st in this example.

If you put your GEOS meteorology and chemistry files in the same directory, such that `GEOS_MET_DIR/Nv` has both the eta-level met and chem files, then you can omit the `--chem-path GEOS_CHEM_DIR` option.

## Producing model files for a runlog

If you are a GGG user and have a runlog for which you need the `.mod` files for, ginput provides a command to create only the `.mod` files needed by that runlog.
(If you don't know what a runlog is, then just skip this section.)
This still requires you have the GEOS files downloaded; {ref}`usage-dl-met` includes a section on how to download files needed for a runlog.

The following example shows how you would generate the `.mod` files for the Park Falls benchmark runlog that is included with GGG.
Since that runlog has spectra from 2004, we cannot use GEOS FP data for it, so this example shows the use of GEOS FP-IT:

```
$ ./run_ginput.py rlmod \
    --include-chem \
    --chem-path GEOS_CHEM_DIR \
    --mode fpit-eta \
    --save-path MOD_FILE_DIR \
    $GGGPATH/runlogs/gnd/pa_ggg_benchmark.grl \
    GEOS_MET_DIR
```

Note that we do not specify latitude, longitude, altitude, site IDs, or dates.
All of that information is automatically inferred from the runlog.
