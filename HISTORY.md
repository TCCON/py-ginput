# Ginput Version History

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
