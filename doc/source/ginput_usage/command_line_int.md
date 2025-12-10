# Ginput command line interface

If you installed `ginput` using:

* the `make install` command,
* the `install.sh` script, or
* you ran the `install-runscript.sh` script with the proper Conda environment active,

then there will be a `run_ginput.py` script in the root directory of the `ginput` repo. This is the main entry point to run `ginput` from the command line. To see all the possible options, call it as `./run_ginput.py --help` from the root directory of the `ginput` repo. 

Starting from v1.6, installing with `pip` or the various `conda`-type installers will also create a `ginput_cli` script in your environment's `bin` directory.
This means that as long as you either (a) have that environment active or (b) linked this `ginput_cli` script to somewhere on your regular `PATH`,
you can call `ginput_cli` from anywhere to run ginput.

```{note}
`ginput_cli` and `./run_ginput.py` are completely interchangeable.
The rest of the documentation uses the latter in examples since, historically, most users have used that entry point.
However, you can replace `./run_ginput.py` with `ginput_cli` in _any_ example in the rest of this documentation, and
it will work the same!
```

This prints out all the possible subcommands:

```
$ ./run_ginput.py --help

usage: run_ginput.py [-h] {oco,acos,geocarb,update_hourly,mod,tccon-mod,rlmod,tccon-rlmod,vmr,rlvmr,getg5,get-rl-g5,map,sample-model} ...

Call various pieces of ginput

positional arguments:
  {oco,acos,geocarb,update_hourly,mod,tccon-mod,rlmod,tccon-rlmod,vmr,rlvmr,getg5,get-rl-g5,map,sample-model}
                        The following subcommands execute different parts of ginput
    oco                 Generate .h5 file for input into the OCO algorithm
    acos                Generate .h5 file for input into the GOSAT algorithm
    geocarb             Generate .h5 file for input into the GeoCarb algorithm
    update_hourly       Update monthly input CO2 files with new NOAA hourly data
    mod                 Generate .mod (model) files for GGG
    tccon-mod           Generate .mod (model) files appropriate for use with TCCON GGG2020 retrievals.
    rlmod               Generate .mod (model) files for spectra enumerated in a runlog
    tccon-rlmod         Generate .mod (model) files appropriate for use with TCCON GGG2020 retrievals for spectra enumerated in a runlog.
    vmr                 Generate full .vmr files for GGG
    rlvmr               Generate .vmr files from a runlog
    getg5               Download GEOS5 FP or FP-IT data
    get-rl-g5           Download GEOS5 FP or FP-IT data for spectra in a runlog
    map                 Generate .map (a priori) files.
```

In the above example, `run_ginput.py` is called as `./run_ginput.py` and *not* `python run_ginput.py`. This is because when `run_ginput.py` is created it is given a [shebang](https://en.wikipedia.org/wiki/Shebang_(Unix)) that points to the Python executable in the correct Conda environment. That's why you can run it as `./run_ginput.py` without activating the Conda environment that `ginput` was installed it first, whereas if you tried to call it as `python run_ginput.py` you *would* have to activate the right environment.

```{note}
The command line help is more likely to be up-to-date than this page. If you notice a discrepancy between the command line help and this page, please [open an issue](https://github.com/TCCON/py-ginput/issues).
```

Each subcommand represents one action that `ginput` can perform. You can get command line help for any subcommand, by calling the subcommand with the `--help` argument, e.g.:

```
$ ./run_ginput getg5 --help

usage: run_ginput.py getg5 [-h] [--mode {FP,FPIT,GEOSIT}] [--path PATH] [-t {met,chm}] [-l {surf,p,eta}] [-g {L,C}] date_range

Download GEOSFP or GEOSFP-IT reanalysis met data

positional arguments:
  date_range            The range of dates to get, in YYYYMMDD-YYYYMMDD format. The second date may be omitted, in which case the end date will be one day after the first date. The end date is not included in the
                        range.

options:
  -h, --help            show this help message and exit
  --mode {FP,FPIT,GEOSIT}
                        Which GEOS product to get. The default is FP. Note that to retrieve FP-IT data requires a subscription with NASA (https://gmao.gsfc.nasa.gov/GMAO_products/)
  --path PATH           Where to download the GEOS data to, "." by default. Data will be placed in Np, Nv, and Nx subdirectories automatically created in this directory.
  -t {met,chm}, --filetypes {met,chm}
                        Which file types to download. Works in conjunction with --levels to determine which files to download.
  -l {surf,p,eta}, --levels {surf,p,eta}
                        Which level type to download. Note that only "eta" levels are available for the "chm" file type.
  -g {L,C}, --gridtypes {L,C}
                        used to specify the grid type when downloading GEOS-IT files, L for lat-lon and C for cubed-sphere

If both --filetypes and --levels are omitted, then the legacy behavior is to download met data for the surface and on fixed pressure levels. However, if one is given, then both must be given.
```

These subcommands can be divided into several groups.


## Downloading met data

The `getg5` and `get-rl-g5` subcommands help to download the GEOS-5 meteorology data the `ginput` ingests to produce met and a priori trace gas profile files. `getg5` allows you to download files for specific dates, while `get-rl-g5` allows users running GGG to download the met files required for a GGG [runlog](https://tccon-wiki.caltech.edu/Main/RunLog) (login required).

For more information, see {ref}`usage-dl-met`.

## Generating model files

In GGG, the meteorology data must be converted to {file}`.mod` files, which store the met variable profiles in a simple text format that GGG can read. There are four subcommands for generating these file:

* `tccon-mod` - generate these files for date ranges using standard settings to generate TCCON-GGG2020-style priors.
* `mod` - generate these files for date ranges with more flexibility in options.
* `tccon-rlmod` - similar to `tccon-mod`, but generates the model files needed by a GGG runlog instead of for a given set of dates.
* `rlmod` - similar to `mod`, but generates the model files needed by a GGG runlog instead of for a given set of dates.

For more information, see {ref}`usage-mod`.

## Generating VMR files

In GGG, a priori trace gas profiles must be written to {file}`.vmr` files, which like the {file}`.mod` files, store these profiles in a simple text format for GGG to read. This step requires that the {file}`.mod` files for the times you are generating files for exist. There are two subcommands related to generating {file}`.vmr` files:

* `vmr` - generate these files for a specific date range
* `rlvmr` - generate these files for the times needed by a GGG runlog instead of a given set of dates.

For more information, see {ref}`usage-vmr`.

## Generating .map files

In contrast to the {file}`.mod` and {file}`.vmr` files, {file}`.map` files are not inputs to GGG but were traditionally output by the post processing. These contain the key trace gas a priori profiles used by GGG. Because some groups rely on these files, `ginput` includes the capability to generate them without needing to run the `gsetup` and `write_aux` programs within GGG. There is only one subcommand related to writing these files, `map`. 

See {ref}`usage-map` for more information.

## Updating input NOAA data

`ginput` relies on timeseries of CO2, N2O, and CH4 from NOAA observatories in Mauna Loa, Hawaii, USA and American Samoa to capture the atmospheric growth of these gases. The repo includes flask files for these gases from 2018 and has the capability to extrapolate forward in time; however, extrapolation will have larger errors than using more recent flask or in situ data. This tool allows users to update files using more rapidly delivered data from NOAA. 

See {ref}`usage-update-noaa` for more information.

```{warning}
The version of the CO2 flask data included in ginput v1.1.7 is still on the X2007 scale. If you need to switch to the X2019 scale, see {ref}`this section <usage-noaa-more-recent>` in the {ref}`usage-update-noaa` page.
```
