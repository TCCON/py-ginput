(usage-dl-met)=
# Downloading met files

To generate the `.mod` and `.vmr` files for GGG2020, `ginput` needs to have 3 types of GEOS files available:

1. 2D assimilated met variables: these are files with the identifier `inst3_2d_asm_Nx` that contain variables like `TROPPB` (i.e. blended tropopause pressure estimate)
2. 3D assimilated met variables: these are files with the identifier `inst3_3d_asm_Nv` that contain variables like `T` (temperature) and `EPV` (Ertel's Potential Vorticity) on hybrid model levels.
3. 3D chemistry simulation variables: these are files with the identifier `inst3_3d_chm_Nv` that contain, among others, the `CO` variable on hybrid model levels.

Ginput provides the ability to download these files from 3 different streams:

1. GEOS FP-IT: this is a product provided for instrument teams by Goddard. It is the data stream used for GGG2020, but requires a data subscription.
2. GEOS FP: this is a freely available simulation product from Goddard. It is available at higher spatial resolution than FP-IT but does not go as far back in time.
3. GEOS IT: this is the new instrument teams product, released by Goddard in 2023. It also requires a data subscription. GGG2020 will switch to this when FP-IT is discontinued.

```{note}
The examples in the rest of this page will show downloading GEOS FP data, since that is the only one of these three that does not require a data subscription.
However, if you are using ginput to generate priors for TCCON or COCCON retrievals, it is required that you use priors generated with GEOS FP-IT or GEOS IT, as those are the standard met types used by both networks.
For information on how to obtain these priors, see:

- [General information](https://tccon-wiki.caltech.edu/Main/ObtainingGinputData)
- [For PROFFAST users](https://tccon-wiki.caltech.edu/Main/EM27ProffastSupport)
```

Downloading is done with the `getg5` and `get-rl-g5` subcommands to `run_ginput.py`. As with all subcommands, passing the `--help` flag will print out the most up-to-date command line argument information. 

## Expected file structure

If you are downloading this data to use as input to `ginput`, then the GEOS data must be stored in a particular file structure as follows:

```
GEOS_ROOT_DIR
├── Nv
└── Nx
```

`GEOS_ROOT_DIR` may be any path. However, under that path you must have two subdirectories: `Nv` and `Nx`. 
The `Nv` directory must contain the 3D files, and the `Nx` directory the 2D files. 
The meteorology and chemistry 3D files can both go in the `Nv` directory, or you could use an alternate directory structure:

```
GEOS_CHEM_DIR
└── Nv
GEOS_MET_DIR
├── Nv
└── Nx
```

Here, `GEOS_CHEM_DIR` and `GEOS_MET_DIR` can again be any path. 
They do *not* need to be siblings to each other; `GEOS_CHEM_DIR` could be `/data/chem` and `GEOS_MET_DIR` could be `/mnt/goddard/geos/fpit` for example.
However, they must have the correct `Nv` and `Nx` subdirectories.
As before, 3D files go in the `Nv` subdirectory and 2D files in the `Nx` subdirectory.

```{note}
The download command described below will place the data automatically in the correct `Nv` or `Nx` subdirectories. You only need to do this manually if
you are downloading GEOS data yourself.
```

## Downloading data for dates

If you want to download GEOS data for a single date or range of dates, the `getg5` subcommand will do that. 
It accepts a date or date range as its positional argument.
The `--path` argument indicates where to download the data to.
(GEOS data volume adds up quickly, so make sure this is right!)
The `--mode`, `--filetypes` (short: `-t`), and `--levels` (short: `-l`) options set which data stream (FP, FP-IT, or IT), which kind of data (meteorology or chemistry) and which set of vertical levels (3D hybrid or 2D) to download files for.
If you wanted to download all the files needed to generate (nonstandard) GGG2020 priors from GEOS FP data, for 1 Jan 2018, you would run this command three times as follows:

```
$ ./run_ginput.py getg5 --mode FP --filetypes met --levels eta --path GEOS_MET_DIR 20180101
$ ./run_ginput.py getg5 --mode FP --filetypes met --levels surf --path GEOS_MET_DIR 20180101
$ ./run_ginput.py getg5 --mode FP --filetypes chm --levels eta --path GEOS_CHEM_DIR 20180101
```

`GEOS_MET_DIR` and `GEOS_CHEM_DIR` can be whatever path you wish. 
Note that in the third command, the argument to `--filetypes` is "chm" (no "e").
If you wished to download the GEOS FP-IT data used for the standard priors, you would change the "FP" argument of `--mode` to "FPIT" (note that this would require a data subscription from NASA Goddard).
If you wished to download data for all of January 2018, you would make the positional argument "20180101-20180201". 
Note that in this case, the second date in the range is *exclusive*, that is, this command would download up to and including 31 Jan 2018 but not 1 Feb 2018.

```{note}
If you omit both of the technically optional arguments `--filetypes` and `--levels`, `ginput` would download the expected 2D met files as well as 3D *fixed pressure levels* files, instead of the hybrid level files expected.
This is because the fixed pressure level files were expected to be the preferred ones during primary `ginput` development, and the decision to change to the hybrid level files came late enough that we chose not to upset the default in a breaking change.
Therefore, if you want the met files that GGG2020 uses, you must give the `--filetypes` and `--levels` options as shown above.
```

## Downloading data for a runlog

```{note}
This section only applies to GGG users, and of those, users who want to generate custom priors themselves.
If you do not use GGG (i.e. if you are using `ginput` to generate priors for satellite missions), this section will not be useful to you.
Likewise, if you are someone who needs to do standard GGG2020 processing of TCCON or EM27 data, you should use the `.mod` and `.vmr` files generated at Caltech as described [here](https://tccon-wiki.caltech.edu/Main/ObtainingGinputData).
```

If you want to download GEOS data necessary to generate the `.mod` and `.vmr` files to process spectra defined in a runlog, the `get-rl-g5` subcommand will automatically download the GEOS files for the dates included in a runlog.
Suppose you have a runlog, `$GGGPATH/runlogs/gnd/xx20200101_20200201.grl`. 
To download the GEOS files needed to make nonstandard `.mod` and `.vmr` files for this runlog, you would again need three commands:

```
$ ./run_ginput get-rl-g5 --mode FP --filetypes met --levels eta --path GEOS_MET_DIR $GGGPATH/runlogs/gnd/xx20200101_20200201.grl
$ ./run_ginput get-rl-g5 --mode FP --filetypes met --levels surf --path GEOS_MET_DIR $GGGPATH/runlogs/gnd/xx20200101_20200201.grl
$ ./run_ginput get-rl-g5 --mode FP --filetypes chm --levels eta --path GEOS_MET_DIR $GGGPATH/runlogs/gnd/xx20200101_20200201.grl
```

By default, this will download the met data required for the entire runlog. 
To limit the dates, use the `--first-date` and `--last-date` options.
For example, if you had a runlog for which you had previously obtained the GEOS data, and that you added new spectra to, you could use `--first-date` to indicate the earliest date for which new GEOS files need downloaded.
These take dates in YYYY-MM-DD format.
