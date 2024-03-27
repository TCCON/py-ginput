(sat-interface)=
# Satellite priors interface

## Overview

The {py:mod}`~ginput.priors.acos_interface` module handles generating priors for satellite retrievals.
Currently, it supports OCO-2/3, GOSAT, and GeoCARB.
Satellite priors use a different driver than the standard TCCON priors, because TCCON priors are generally needed for only a few locations but for all times of day, whereas satellites need priors for a large number of locations but each at only a single time.
Therefore, some data can be loaded once and reused for generating the satellite priors at different locations more efficiently than it can for all the different times of day needed for TCCON.

The entry point to the {py:mod}`~ginput.priors.acos_interface` module is the `acos_interface_main` function. 
The general steps it takes are:

1. Load met data from a source file containing met data resampled at the desired locations and map the variables to ones recognized by the rest of `ginput`.
2. Generate equivalent latitude (EqL) interpolators that cover the time range represented in the met file
3. Use those interpolators to calculate EqL profiles for each satellite sounding
4. Calculate trace gas profiles for each sounding using the EqL plus met profiles
5. Save the trace gas profiles plus some ancillary data to an HDF5 file

If a satellite provided EqL profiles in its met files, then steps 2 and 3 could be skipped. 
However, EqL can be a bit tricky to calculate and is not included in many meteorology reanalysis products, so we provide the capability to calculate it in `ginput`.

## Loading met data

How met data is stored tends to vary a lot from satellite to satellite, so each satellite supported by `ginput` has its own `read_*_resampled_met` function, and which one is used depends on which satellite instrument is requested via the `instrument` argument to the main function. These `read_*_resampled_met` function can use a helper function, `read_resampled_met`, which knows how to read variables from HDF5 files and map them to variables `ginput` recognizes, given a dictionary mapping HDF5 paths to `ginput` variables. The required `ginput` variables are:

- `pv`: profile of Ertel's potential vorticity, in units of K m+2 kg-1 s-1.
- `temperature`: profile of temperature, in Kelvin
- `pressure`: profile of pressure, in hPa (converted from Pa by `read_resampled_met`)
- `date_strings`: the date/time of the sounding in UTC or GPS time (for `ginput`, the sub-minute difference between them is unlikely to have a big impact).

```{note}
`read_resampled_met` assumes that the datetime strings have the format "yyyy-mm-ddTHH:MM:SS.dddZ", e.g. 2018-01-02T12:30:45.300Z for 12:30:45.300 UTC on 2 Jan 2018. If your met file stores datetimes in another format, then either `read_resampled_met` will need to be updated to support that, or you will need to create a custom version of `read_resampled_met` for your instrument.
```

- `altitude`: height profile in kilometers above sea level (converted from meters by `read_resampled_met`)
- `latitude`: latitude of the sounding, with south represented as negative
- `longitude`: longitude of the sounding, with west represented as negative
- `trop_pressure`: pressure at the tropopause, in hPa (converted from Pa by `read_resampled_met`)
- `trop_temperature`: temperature at the tropopause, in Kelvin
- `surf_gph`: despite the name, this is surface altitude in meters (converted from meters by `read_resampled_met`)
- Optionally, `co` if you require CO profiles. This must point to profiles of CO dry mole fraction, in units of mol/mol.

`read_resampled_met` returns a dictionary with values for all of the above variables, plus several which it calculates:

- `theta`: potential temperature profiles in Kelvin
- `dates`: Python {py:class}`~datetime.datetime` instances corresponding to the date strings
- `datenums`: the number of seconds since midnight UTC 1 Jan 1970 corresponding to the date strings
- `surf_alt`: this is a special case that replaces `surf_gph` in the output dictionary. It is the surface altitude in kilometers.

`read_resampled_met` returns the met arrays in their native shape in the HDF5 file, however the rest of the interface expects the variables first two dimensions to represent different soundings (i.e. along and across track, the exact definition does not matter, as it just iterates over both to compute the profiles). 2D variables such as `surf_alt` need only those two, 3D profile variables must have the levels as the third dimension. Additionally, the levels must be ordered space-to-surface.

```{note}
If you need to implement a new reader for met data in a different format than a typical ACOS-like met file, you may do best to implement a `read_*_resampled_met` function that loads this data directly, rather than trying to make `read_resampled_met` handle your specific format. In this case, take careful note of the expected units of each quantity in the returned dictionary.
```

## Calculating equivalent latitudes

Calculating equivalent latitude (EqL) requires testing how the potential vorticity (PV) and potential temperature (PT) of a given profile compare to those quantities over the remainder of the globe. This is accomplished in `ginput` by constructing a 2D interpolator of EqL vs. PV and PT from GEOS met files passed in to the main driver function. Each profile's PV and PT are then used to find the EqL from the interpolators for the two GEOS files that bound that profile in time. The final EqL is then an average of those two profiles, weighted by the temporal proximity of the sounding to those GEOS files. That is, a sounding at 13:00 UTC would be bounded by the 12:00 and 15:00 GEOS files, and the EqL from the 12:00 file would be weighted twice as strongly as the EqL from the 15:00 file.

```{note}
Constructing the interpolators is actually somewhat time consuming. Since it takes the same amount of time to construct one interpolator whether it is used to get EqL for one profile or thousands of profiles, the satellite interface is built to construct the interpolators once and use them for all profiles.
```

EqL can be calculated in serial or parallel, depending on the value of the `nprocs` parameter. In `compute_sounding_equivalent_latitudes`, it dispatches to `_eqlat_serial` or `_eqlat_parallel`, which both iterate over all the soundings and call `_eqlat_helper` on each one. The main difference is whether the iteration occurs via a simple `for` loop or a {py:mod}`multiprocessing` construct. (The original reason for having separate function was to test the parallel version against the serial one, to ensure the EqL profiles were mapped to soundings correctly. In production, use of the parallelization is highly encouraged.)

There is one aspect of running in parallel to note: {py:mod}`multiprocessing` uses pickles to send data between processes. For Python versions up to at least 3.6, the maximum size of pickles was limited to the number of bytes that could be recorded in a 32-bit integer. Because of how the interpolators are constructed, if the range of PV or PT became very large, the interpolators would exceed this size limit and could not be passed between processes. This limitation was addressed between Python 3.7 and 3.10.

As a workaround, if `ginput` detects that it is running on Python 3.9 or earlier (since Python 3.10 is the version we used in testing where we know this issue is fixed), it will automatically write the interpolators out to pickle files on disk, pass the paths to the child processes, and reload the pickles from disk in the child processes. It takes care to ensure that the pickle file names are unique to each instance of `ginput` running and to clean up the files when done or if a fatal error occurs. This logic is handled by {py:class}`~ginput.priors.acos_interface._eqlat_pickle_manager`.

## Calculating trace gas profiles

Calculating the trace gas profiles is in many ways similar to calculating EqL. Similarly to the EqL interpolators, the trace gas profiles require instances of {py:class}`~ginput.priors.tccon_priors.TraceGasRecord` subclasses which can be time consuming to instantiate. To address this, the needed instance is instantiated once in the main function, then it calls either `_prior_serial` or `_prior_parallel`, which handle iteration akin to how `_eqlat_serial` and `_eqlat_parallel` do for EqL. Both of these functions call `_prior_helper`, which itself calls out to {py:func}`~ginput.priors.tccon_priors.generate_single_tccon_prior` to create the trace gas profile derived from the met data for one sounding. If multiple gases are required, the main function loops over them, instantiating the needed {py:class}`~ginput.priors.tccon_priors.TraceGasRecord` for each gas and creating the array of profiles.

## Saving trace gas profiles

The trace gas profiles are written to HDF5 files. These files are created in two steps. First, {py:func}`~ginput.priors.acos_interface.init_prior_h5` creates the output file and assigns certain global attributes. Then, for each gas, {py.func}`~ginput.priors.acos_interface.write_prior_h5` is called to create and write to the output variables. When multiple gases are required by the `instrument`, each one is written to a group that include the gas's name. When only a single gas is needed, it is just written out to a group called "priors".

Additionally, for gases that ingest a timeseries of NOAA data as their basis, that timeseries (extended as needed by `ginput`) is written out to a separate group. As with the prior profiles, the group name contains the gas name if the `instrument` required >1 gas.

## Special considerations for reproducibility

`ginput` uses NOAA surface data from Mauna Loa/Mauna Kea and American Samoa as the basis (directly or indirectly) for CO2, N2O, CH4, and HF priors. `ginput` comes with an old set of monthly average flask data for CO2, N2O, and CH4 (HF does not have its own NOAA data; it is derived from CH4) that end in 2018. It is capable of extrapolating these timeseries into the future; however, it has to assume that there will be no unusual events that would disrupt the trend inferred from the last 5 or 10 years. Such events include El Nino years, which introduce step changes in at least the CO2 trend.

To avoid unecessary extrapolation, some users receive more frequent updates to the NOAA data and use that as the input, rather than the out of date data contained with `ginput`. The "Updating NOAA input data" section of this documentation describes how to handle that update, as well as some of the concerns about changes between early, not fully quality controlled data, and the proper, fully QC'd data. Our concern here is how that updating process could impact the ability to regenerate identical priors before and after an input data update.

The issue stems from `ginput`'s need for data in the future from the date for which it is generating priors for. This is because it handles deseasonalization with a rolling average, plus is assumes that the change in gas mixing ratios in the northern hemispheres precedes the corresponding change measured at the tropical NOAA sites.  Combined, those changes mean `ginput` could need data as much as a year in the future: priors for 1 June 2018 may need data from May even June 2019.

Consider what this means if you want to generate priors for 1 June 2018 in June 2018. The latest NOAA data you could possibly have is for May 2017, so `ginput` would need to extrapolate for 12 months. That works fine, however, now consider what happens if you need to regenerate those same priors in December 2018. Depending on how frequently you update the NOAA data, you could have up to November 2018, meaning that for those 1 June 2018 priors, `ginput` now only needs to extrapolate 6 months, as now June through November 2018 have proper data. That proper data will almost certainly be different from what `ginput` extrapolated.

To allow users to ensure consistent priors no matter when they are generated, the satellite interface provides the `truncate_mlo_smo_by` input. This allows users to specify a month relative to the maximum date in the met file. If the NOAA data does not cover at least up to that month, an error is raised. Otherwise, only data up to and including that month is actually used; months after that are always extrapolated even if they have data in the NOAA files.

To give a concrete example, consider the 1 June 2018 priors again. If we run in June with `truncate_mlo_smo_by` set to 2, then the end month for the NOAA data will be set to April 2018. Assuming we have that month, `ginput` will run and extrapolate the NOAA data for every month starting from May 2018. If we then run the 1 June 2018 priors in December 2018 (stilll with `truncate_mlo_smo_by` set to 2), then again because the maximum date *in the met file* should be June 2018, the NOAA data will be truncated after April 2018, even though we have 6 additional months of data now.

## Error handling

Error handling is done with the {py:class}`~ginput.priors.acos_interface.ErrorHandler` class. This helps to catch errors that should not crash the entire program, but instead just put fill values for that sounding's prior profiles. Any such cases are wrapped in a `try-except` block, and handled errors are passed to the `ErrorHandler.handle_err` method along with the indices of the current sounding, the error code, and an array of status flags (one per sounding). The `ErrorHandler` looks up the numeric error code corresponding to the string error code given and writes that code to the flag array.

```{note}
If you give the `ErrorHandler` an invalid string error code, it will cause a total crash, so don't mistype! The allowed error codes are the keys of the `_err_codes` attribute of the `ErrorHandler` class.
```

## ACOS assumptions
The satellite interface was built to support satellites that used ACOS-like retrievals, including OCO -2 and -3, GOSAT ACOS, and GeoCard. Several assumptions about ACOS data formats are currently embedded into this interface:

- **Fill values:** The interface assumes floating point values less than `-9e5`, integer values of -999999, and string values of "N/A" represent fills.
- **Fill values in time fields:** Any datetime before 1993 is considered a fill value or indicative of a bad sounding; this stems from the definition of TAI93 time used in many satellite products. Since that starts at 1993, if a negative fill value ended up in a TAI93 time field, it produces a time before 1993.
- **Time strings:** As mentioned in the met section, time strings are assumed to be of the form "yyyy-mm-ddTHH:MM:SS.dddZ"
- **Met file structure:** If using `read_resampled_met` to help handle reading in met data, note that it can currently only handle HDF5 files which are organized within top level groups. That is, you might have a group "Meteorology" which contains the temperature, pressure, etc. and another group "SoundingGeometry" which has the latitude, longitude, etc, but these groups cannot have subgroups, nor can the variables be directly in the root of the HDF5 file. If your met file does not follow this organization, either `read_resampled_met` will need to be modified or you will need to write a custom top-level reader to call from the main driver.