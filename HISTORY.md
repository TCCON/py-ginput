# Ginput Version History

**A note on versioning:** because of the nature of this code, `ginput` follows
a different set of criteria for major/minor/patch versions:

- The major version number will be incremented when the output from `ginput`
  changes so much that retrievals using the new priors are likely to produce
  different results than those using the previous version's priors.
- The minor version number will be incremented when a new feature is added
  or the API (Python internal or command line) changes. Thus, users should
  expect that upgrading from, e.g., v1.3.x to v1.4.0 may require updates to
  how they call `ginput`.
- The patch version will be incremented when changes do not break the API
  or alter existing output variables are made.

We recognize that this is non-standard, and that breaking API changes should
result in a major version increase. However, our experience has been that small
improvements to improve the scientific capabilities of this code sometimes require
an update to the API because the new best default behavior requires additional
user input.

## 1.6.0

This release includes the following changes:

### The default method for interpolating & extrapolating NOAA data changed.

Previously, the MLO and SMO timeseries were averaged together first (dropping
any months where one site was missing data), then any missing months within
the timeseries were filled in by simple time-linear interpolation and any
necessary extrapolation at the beginning and end was done with a combination
of a fit to the secular trend and imposition of an average seasonal cycle.

This, however, performs poorly when either site has a large gap - the linear
interpolation means that a gap of more than a few months will fail to capture
a seasonal cycle, biasing the overall trend high or low depending on where the
gap starts and ends in the cycle. 1.6.0 introduces a new approach by default,
where MLO and SMO are filled in and extended separately _and_ use the trend +
seasonal cycle logic to interpolate all gaps.

Because the sites are now gap filled and extended separately, this will result
in changes even when using MLO and SMO data that do not have gaps. All of the
command line interface calls use this new method. For simplicity, there is
no way to revert to the old method from the command line. However, the
`MloSmoTraceGasRecord` and its child classes _do_ have a new keyword
(`use_pre1p6_interpolation`) that can be set to `True` to switch them back
to the old interpolation/extrapolation method. Likewise, the `acos_interface_main`
function in the `priors.acos_interface` module has a `pre_1p6_interp` keyword
that, when set to `True`, reverts to the old method. If you truly need to
use version 1.6+ but retain the old method, please write custom wrapper code
to call the necessary routines with the custom gas records or options for
`acos_interface_main`.

### Improved CLI access

Ginput traditionally creates a `run_ginput.py` script when installed manually.
We have now added a console script entry point so that, when installed with
`pip`, the `ginput_cli` script will be added to your current environment as
well. This simplifies the install process.

The `main()` function is also now found in `ginput.__main__` and can accept
a list of command line arguments. This means that you can mimic CLI calls
from within Python more easily.

### Prototype netCDF tool

This version adds a prototype subcommand (`tar2nc`) that converts standard
site tarballs of `.mod` and `.vmr` files to a single netCDF file. It is
still being developed, and may not work for all versions of `.mod` and
`.vmr` files.

## 1.5.1

This release aims to fix cases where the `git` or `hg` commands are not available,
causing the writers for some files to fail.

## 1.5.0

This release includes several disparate updates:

- Updated usage of `pandas` and `scipy` to remove calls to deprecated functions
  or using deprecated keywords/time interval formats.
  This also allows `ginput` to work under Python 3.12. (Credit: @rocheseb)
    - Because this raised the minimum versions of a few dependencies, we have
      incremented the minor version number to indicate that users may need to
      rebuild their environments.
- The test cases now use `pytest` instead of the built-in unit testing framework.
    - The tests have now been reorganized to reside in the top level of the repo;
      the `ginput/testing` subdirectory is deprecated. It will be full removed once
      the last utility commands are migrated.
    - This was done to make it easier to run tests while creating a conda-forge
      package.


## 1.4.4

This release updates the `fo2_prep` program to v1.0.1, which handles a change
to the Scripps O2/N2 data file format that occurred in 2025.

It also fixes an edge case in the satellite priors code where no sounding had
valid input met data. Before, it would crash before completely writing the output
file. Now, that no longer occurs.

## 1.4.3

This release just fixes the allowed scipy version, otherwise it is identical
to 1.4.2.

## 1.4.2

This release addresses two issues.

First, it now errors by default if input in situ data includes negative values,
assuming that these are fill values which should have been replaced by NaNs.

Second, it fixes a bug in error handling in the satellite priors code.
Previously, `ginput` would crash if a sounding resulted in an error that
did not have a string as its first argument; it will now properly convert
that argument to a string.


## 1.4.1

This release fixes a bug when creating satellite prior files with a specific
O2 dry mole fraction file as input and adds more flexibility to the creation
of the O2 dry air mole fraction files:

- When passing an O2 file to the `oco` or `acos` subcommands with the `--fo2-file`
  option, it is now correctly passed down into the functions that calculate the
  priors. Because these function instantiate an instance of the O2 record class,
  `ginput` would crash if you specified `--fo2-file` without having the default
  O2 file in the `ginput` data directory. This is now fixed.
- The `update_fo2` subcommand now accepts a `--dest-file` argument to specify where
  to write the new/updated O2 file, rather than assuming that it should overwrite
  the existing O2 file.
- The `update_fo2` subcommand now has a `--no-download` flag to disable automatic
  download of the NOAA and Scripps input files required for the O2 DMF calculation.
  In that case, the location of these files can be specified through the `--download-dir`
  option, which can accept either a path to a directory containing the files
  (which must have the expected names) or a JSON file specifying where to find
  each of the 4 required files.
- The `update_fo2` subcommand now has a `--extrap-to-year` option, which will
  cause the output f(O2) file to be extrapolated to the given year. This allows
  less dependence on the upstream data files being kept up-to-date.

Other changes:

- A new one time script is included to pad NOAA hourly data with fill values to the
  end of the current year, in case the available hourly files stop early.
- NOAA data through 2024 is included in the repo for future-proofing.

## 1.4.0

This release adds mean O2 dry air mole fractions into the satellite .h5 files.
There will be one value per sounding in the "o2_record/o2_global_dmf" variable
and a granule mean in the "o2_record/granule_mean_o2_global_dmf" variable.
Note that the latter is a scalar; if reading with the `h5py` Python package,
you will need to slice it with an empty tuple rather than a colon to get the 
value. This change means that running the satellite priors module requires that the
input data for the O2 DMF calculation be available or it be allowed to automatically
download and prepare those inputs. Because this is a change to the command line API,
we have incremented the minor version number.

The `update_fo2` subcommand also now has more options to control where the various
data files are written, if needed.

Additionally, the time needed to calculate the stratospheric lookup tables for N2O
and CH4 has been reduced by a factor of 3, with no change to the output values. This
was accomplished by minimizing duplicate calculations in the inner-most loop for those
LUTs.

## 1.3.1

This release improves the ergonomics of including the time-varying O2 mole fraction
by providing an option to the `O2MeanMoleFractionRecord` and a command line flag for
the `vmr` and `rlvmr` subcommands to automatically download the necessary data and
create/update the O2 mole fraction data file when creating `.vmr` files.

## 1.3.0

This release primarily adds the ability to calculate a time-varying O2 mole fraction,
which gets added to the `.vmr` file header. To support this, it includes a new program
that downloads Scripps O2/N2 data and NOAA global average CO2 data and calculates the
yearly O2 mole fraction from them. This file is not included with `ginput`, but can be
obtained by running the `update_fo2` subcommand of `run_ginput.py`. 

Other changes:

- `GeosSource` and `GeosVersion` classes moved from the `mod_constants` module to the
  new `versioning` module (under `common_utils`).
- Some type hints in `get_NOAA_flask_data` updated to be backwards compatible to at least
  Python 3.7.
- Small update to `mod_maker` to handle taking chemistry variables from a different version
  of the GEOS files than the met variables. This will support possible TCCON reprocessing
  with GEOS FP-IT met and GEOS IT CO.
- GEOS file information now includes checksums

## 1.2.1

Specific changes:

 - Improved tracking of GEOS source; each GEOS file contributing to the `.mod` file
   has its version written in the `.mod` file header.

## 1.2.0

This release fully handles generating priors from GEOS IT met and chemistry data.
Specific changes:

- Tropospheric scaling for CO set to 1 (i.e. no scaling) when using GEOS IT chemistry files
- A new module, `ginput.priors.automation`, and the corresponding subcommand (`auto`) 
  constructed as an interface for a system that automatically runs `ginput`.
- Functions to get information about the most recent commit for logging/file metadata have
  been updated to work with Git and Mercurial.
- More flexibility to print download URLs for met files to stdout instead of to a file
- New subprogram, `getnoaa`, to update the NOAA data used as input (pull request #6 by
  rocheseb).
- Additional catch to handle missing `udunits2` system dependency when importing the `writers`
  module (pull request #8 by chris-msat).

This is a minor version bump for two reasons:

1. The `.mod` file format has changed slightly; there is a new header row that indicates
   which GEOS product the CO in the `.mod` file came from. This is important because the
   GEOS IT CO fields require no scaling in the troposphere, whereas the GEOS FP-IT CO fields
   do, therefore `ginput` must know which GEOS product provided the CO values in the `.mod`
   file (which is never scaled) when generating the `.vmr` file (which is scaled as needed).
2. There is a new feature, the `automation` module and subcommand in `run_ginput.py`. This is
   not something users will generally need to use; it is solely intended as an interface point
   between the Rust version of the Caltech automatic priors generator and the Python ginput.

## v1.1.8

Another minor update to address issues arising from running with GEOS IT.

1. Download URLs for GEOS IT updated to latest product.
2. Solves an issue running the satellite interface (`oco`, `gosat`, or `geocarb` subcommands)
   with 3 GEOS IT input. The interpolators created for the GEOS IT files are large enough that
   three cannot be passed between threads in Python 3.6 due to a limit on the number of bytes
   that the Python 3.6 multiprocessing module can pickle. This is fixed by Python 3.10 at the
   latest, but getting Python 3.10 and required numerical dependencies to reproduce the Python
   3.6 results to numerical precision was not possible. Therefore, as a workaround, if the 
   satellite interface detects that it is running on Python 3.9 or earlier, it will pickle the
   interpolators as separate files and load them back in from the threads when `--nprocs` is not
   0.

There are two other aspects to this release:

1. This is the first release that can be run on Python 3.10 and has the changes needed to run
  the satellite interface with GEOS IT files. v1.1.7 didn't have those GEOS IT changes and
  v1.1.5d was not compatible with Python 3.6.
2. The unit testing code now ignores the GINPUT_VERSION value in the `.vmr` file headers; this
  saves us from needing to update the test input files with each version if there should not
  be changes in the output.

## v1.1.7

Installing with pip also includes the ginput/data folder without needing to use the editable mode

## v1.1.5f

**`mlo_smo_prep` version 1.1.1**

This version is a small bugfix to allow `mlo_smo_prep` to handle updating the SMO monthly average
file when there is no SMO data. Previous versions would crash because the function that checked
that all the necessary GEOS files were present did not handle an empty dataframe of NOAA data
and could not calculate the min and max datetimes required.

## v1.1.5e

During testing with GEOS IT files, we found that cases in which 3 EqL interpolators needed passed
between threads crashed due to a limit on the maximum size of data which can be passed between
threads in Python 3.6. Because upgrading to Python 3.10 will fix that issue but definitely introduce
numerical differences, this version implements a workaround in which the EqL interpolators are
saved to disk as pickle files and read in from the threads, bypassing the inter-thread object
size limit.

## v1.1.5d

**`mlo_smo_prep` version 1.1.0**

Minor, backwards compatible, update to allow the `update_hourly` subcommand to accept hourly files from alternate NOAA sites.

This change also stems from the lack of NOAA hourly data from MLO after the Mauna Loa eruption
at the end of Nov 2022. NOAA set up temporary measurements on Mauna Kea until the Mauna Loa
observatory can be reopened. This data comes with the site ID "MKO". Previously, the `update_hourly`
command would not allow either the hourly or monthly input files to contain site IDs other than
"MLO" or "SMO" as a protection against accidentally passing the wrong file for the wrong site.

To support MKO data, plus any potential future site shifts, this version adds two new command line
options to the `update_hourly` subcommand:

* `--allow-alt-noaa-site`: when this flag is passed, the hourly file is allowed to have a site ID
  that does not match the expected "MLO" or "SMO". That site ID will be recorded as the site ID 
  for the new months in the output monthly file. An error will still be raised if the input hourly
  file contains multiple site IDs.
* `--site-id-override`: allows the caller to pass a site ID to use in the output monthly file
  *instead* of the site ID(s) found in the input hourly file. When given, the hourly file *may*
  have multiple site IDs; they will be ignored and the site ID passed to this option will be 
  used instead.

The site IDs in the input monthly file are still checked, but will no longer raise an error in
any case. Instead either a warning or informational message will be logged if the site ID(s) in
the input file are do not match "MLO"/"SMO" or the override site ID. Whether a warning or 
informational message is printed depends on whether `--allow-alt-noaa-site` is absent or present.
Make this check a warning rather than a hard error was done because once a monthly file uses an
alternate site once, it will always have multiple site IDs going forward, which would require
passing `--allow-alt-noaa-site` every time, even after the hourly file reverts back to the
expected site (MLO or SMO).

Like v1.1.5b and v1.1.5c, this version number is outside the standard semantic versioning pattern,
as it was a fix that needed to be applied to the version of `ginput` used for OCO-2/3 B11 processing.

## v1.1.5c

**`mlo_smo_prep` version 1.0.2**

Minor patch to fix unexpected crash in `update_hourly` subcommand when NOAA hourly data is all fills.

In Dec 2022, the NOAA hourly data from Mauna Loa was all flagged. This caused a crash
when running the `update_hourly` subcommand because it expects there to be at least
some valid data during the preliminary filtering process. The fix was straightforward,
as if there is no valid data, the preliminary filtering cannot filter out any more
data and so can return early. This produces a NaN in the monthly average output file
as expected.

Like v1.1.5b, this version number is outside the standard semantic versioning pattern,
as it was a fix that needed to be applied to the version of `ginput` used for OCO-2/3
B11 processing.

## v1.1.5b

**acos_interface version 1.2.3**

Minor patch to support GEOS-IT file naming conventions when generating OCO-2/3 priors.

This version number is outside the standard semantic versioning pattern, as the request
was to make the fix without changing any science output. To ensure that was the case, 
this patch was applied off of the 1.1.5 version of ginput used in OCO-2/3 B11 rather
than the later 1.1.7 version. It will be merged into the main branch in a later version.

## v1.1.5

**acos_interface version 1.2.2**

Two fixes:

1. HDO priors are now calculated as an absolute value to avoid introducing negative
   DMFs when the H2O DMF is too small.
2. The MLO-SMO derived priors now include an option to turn off the altitude grid adjustment
   that was introduced during development while using the fixed pressure level GEOS FP-IT files.
   The satellite interface turns off that grid adjustment in all cases. This was implemented 
   due to the discovery of an edge case in OCO-2 granule `211102032132s` where this grid adjustment
   erroneously moved the bottom altitude layer to altitude 0.

## v1.1.4

Bugfix in the update-hourly subcommand for SMO files; versions 1.1.0 to 1.1.3
have an error in the wind filtering that effectively accepts data in the wind
sector 0 to 180 rather than 330 to 160 deg CW from north. This has only a small
impact on the monthly average CO2 DMFs for SMO (< 0.02 ppm max) and a very small
impact on the priors (< 0.001 ppm with my test OCO-2 and GOSAT granules).

## v1.1.3

**acos_interface version 1.2.1**

Per SDOS request, `acos_interface.py` modified to limit MLO/SMO extrapolation to 
2 year + 1 month from the data date, rather than execution date.

## v1.1.2

Additional bugfixes to `update_mlo_smo` program, as well as a small fix to the main
priors code.

`update_mlo_smo`:

* Now uses the dataset creation time listed in the NOAA hourly file header to 
  determine what the last actually useful data row is. 
* Fixed behavior that caused it to break on hourly files that do not list data
  after the creation date.
* Can specify what the last month that should be added to the monthly average file
  is; by default, it is the last month (calculated from the date the program is run).

Main priors code:

* When checking if the MLO/SMO input files reach late enough to satisfy the truncation
  requirement, months with NaNs in the input files now count towards this criterion.

## v1.1.1

Two small bugfixes to the `update_mlo_smo` program:

1. When updating the monthly CO2 files, fill values at the end of the NOAA hourly
   file (present if they produce a file for the rest of the current year) are ignored
   so that NaNs are not introduced at the end of the monthly file.
2. Added a flag when updating the SMO monthly file to allow missing GEOS surface files.
   By default, an error is raised if any are missing; this flag allows that check to 
   be bypassed.

## v1.1.0

**acos_interface version 1.2**

Primary change: the MLO/SMO data ingested can optionally be truncated at a certain
date and forced to be extrapolated after that. The `acos_interface` now uses that
option by default to truncate the data to two months before the latest date in the
input met file.

In conjunction, a new module (and subcommand) has been added to prepare NOAA hourly
in situ CO2 data from MLO & SMO into monthly average files. This can permit more
frequent update of the MLO/SMO data to avoid falling out of sync with the real CO2
trend.

Fixed incorrect calculation of oversaturated H2O VMRs in `mod_maker`.

Two small quality-of-life improvements to `mod_maker`:

1. Some of the messages printed updated to be clear they are not time predictions
2. --flat-outdir option added

## v1.0.10

.run_ginput_template.py fixed - was incorrectly passing gosat parser
inplace of the geocarb parser.

## v1.0.9

Update TCCON site definitions: Harwell abbreviation changed to "hw" and
Nicosia added.

## v1.0.8

Added geocarb option to the satellite interface.

## v1.0.7

This version should introduce no scientific changes. It adds some functionality
to the command line interface to generate GEOS-FP derived files, fixes some small
edge-case bugs, and adds manpage documentation.

## v1.0.6

HF priors now have values < 0.1 ppt set to 0.1 ppt. This allows proper computation of
HF averaging kernels, which requires non-zero VMRs.

## v1.0.5

Updated Caltech and Dryden lat/lon in `tccon_sites.py` to optimize their profiles.
For Caltech, the main goal was to get the surface altitude in GEOS closer to the
instrument altitude. For Dryden, the goal was to reduce the influence of LA on the
CO profiles. Wollongong's position was also reevaluated with the `mod_maker` interpolation
bugfix, and kept at its previous value.

Also modified `tccon_priors` to write the tropospheric effective latitude and 
midtropospheric potential temperature to the .vmr file headers.

## v1.0.4

ACE-derived lookup tables for N2O, CH4, and HF regenerated a second time using
the 72-level terrain following GEOS-FPIT files. In v1.0.3, they still used the
42-level fixed-pressure GEOS files when deriving the CLaMS age for all the ACE
profiles.

## v1.0.3

ACE-derived lookup tables for N2O, CH4, and HF regenerated with lat_lon_interp
bugs fixed. LUTs still derived using fixed-pressure-level GEOS files.

## v1.0.2

Second bug fix in mod_maker.lat_lon_interp. Latitude and longitude coordinates
were backwards in interp2d; lon needed to go first because in interp2d the first
coordinate is the coordinate for the columns of the data array. This was not 
introduced in v1.0.1, there were two separate problems.

## v1.0.1

Bug fix in mod_maker.lat_lon_interp. Second row of the data array in the interp2d
call was backwards (lon2 was in the first column; should have been in the second.)

## v1.0.0

First full release of Ginput. 
