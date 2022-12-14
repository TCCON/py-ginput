# Ginput Version History

## v1.1.7

Installing with pip also includes the ginput/data folder without needing to use the editable mode

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
