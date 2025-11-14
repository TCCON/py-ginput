Ginput: ACOS Interface
======================

.. Sphinx uses second level sections as the headers in man pages, the first level is skipped

Synopsis
--------

run_ginput.py oco | acos | geocarb [ --use-trop-eqlat ] [ --cache-strat-lut ] [ --raise-errors ] [ -v | --verbose ] [ -q | --quiet ]
                                   [ --mlo-co2-file CO2_FILE ] [ --smo-co2-file CO2_FILE ] 
                                   [ --truncate-mlo-smo-by NMONTHS] [--no-truncate-mlo-smo]
                                   [ -n | --nprocs NPROCS ] [ --raise-errors ]
                                   GEOS_FILES   MET_FILE   OUTPUT_FILE


Description
-----------

The oco, acos, or geocarb subcommands to run_ginput.py generate an HDF5 file containing CO2 (and for geocarb, CH4 and CO) priors
OCO-2/3, GOSAT, or GeoCarb level 2 retrievals. It requires access to both the met variables resampled to the satellite sounding
locations/times and the original GEOS-FP(IT) files in order to compute equivalent latitude for the stratosphere. It is configured
to allow for parallelization over soundings if desired to speed up processing. This can sometimes cause problems with OpenMP threads,
see Notes, below.


Arguments
---------

**GEOS_FILES** 
    A required positional argument that is a comma separated list of GEOS FP or FPIT files that cover the times of the soundings. 
    For example, if the soundings span 0100Z to 0200Z on 2018-01-01, then the GEOS files for 2018-01-01 0000Z and 0300Z must
    be listed. If the soundings span 0230Z to 0330Z, then the GEOS files for 2018-01-01 0000Z, 0300Z, and 0600Z must be listed.

**MET_FILE**
    A required positional argument that is the path to the HDF5 met resampler file containing the meteorology data for these sounding.

**OUTPUT_FILE**
    A required positional argument that is the filename to give the output HDF5 file containing the prior profiles and any additional 
    variables. Note that this path will be overwritten without any warning.

**--use-trop-eqlat**
    Turn on using a theta-derived equivalent latitude in the troposphere.

**--cache-strat-lut**
    Give this flag to turn on the ability of the code to cache the stratospheric CO2 and CH4 lookup tables rather than recalculating 
    them each time this program is launched. Even when cached, the table will be recalculated if the code detects that the dependencies 
    of the table have changed. This can speed up the code because the calculating the CH4 lookup table especially takes significant time,
    but it requires that the code be able to write to its own installation directory, so is disabled by default.

**--mlo-co2-file CO2_FILE**, **--smo-co2-file CO2_FILE**
    These arguments allow you to specify which file the Mauna Loa (mlo) and American Samoa (smo) NOAA monthly average flask data are
    read from. If not specified, the default files included with ginput are read (./ginput/data/{ML,SMO}_monthly_obs_{co2,ch4}.txt).
    Note that for OCO and GOSAT these may be specified normally, i.e. /data/priors/ml_monthly_obs_co2.txt. However, if producing
    GeoCARB priors, these paths `must` include the substring `{gas}` which will be replaced with "co2" or "ch4", depending on which
    gas's priors are being produced.

**--truncate-mlo-smo-by NMONTHS**
    To enforce consistent priors generation when using MLO/SMO input files that are updating frequently, the MLO/SMO data can be truncated
    at a specific date, such that any future re-runs of the priors code with MLO/SMO files that have additional data still produce the same
    priors. The default behavior is to use MLO/SMO data up to and including the month for which priors are being generated. Setting this 
    option to a value >0 will move the last required month back in time. For example, using `--truncate-mlo-smo-by 1` when producing 
    priors for a granule in May 2017 will require MLO/SMO data up through April 2017.

**--no-truncate-mlo-smo**
    Setting this flag disables the MLO/SMO truncation; instead all available MLO/SMO data will be used and whatever extrapolation is needed 
    will be done. This also disables the check that MLO/SMO data includes a certain minimum date.

**--fo2-file FO2_FILE**
    This option accepts a path to the tabular file created by the `update_fo2` subcommand, which contains the annual mean O2 dry mole
    fraction. If not given, then the default file in the `ginput` data directory will be used and must be sufficiently up to date.

**--auto-update-fo2-file**
    This flag will automatically try to update the `FO2_FILE` to include the latest NOAA and Scripps data. It does NOT extrapolate
    past the end of that data, so if some of that data is too out-of-date, you will need to create an O2 mole fraction file with
    the `update_fo2` subcommand using its ability to extrapolate to more recent years, and pass the resulting file in using the
    `--fo2-file` option.

**-n, --nprocs**
    Number of processes to use in parallel when computing the priors. The default is to run in serial. Passing a number >=1 will use
    that many parallel processes. See note below about potential interaction with numpy threads.

**--raise-errors**
    Adding this flag will cause any error to be raised like a normal Python error instead of potentially being suppressed and just flagged
    in the output file.

**-v, --verbose**
    Increase logging verbosity. May be specified multiple time to further increase it.

**-q, --quiet**
    Silence most logging output. Critical messages will still be displayed, and there may be some messages not handled by the logging
    system that will not be silenced.

Notes
-----

OpenMP threads
**************

numpy (one of the Python packages used by ginput) can use multiprocessing threads to paralleize array calculations. This can sometimes
lead to errors similar to "pthread_create: Resource temporarily unavailable", especially if running with --nprocs > 1. Usually the 
problem is that each ginput process has its own numpy which is trying to use as many threads as possible. This can be avoided by
setting the environmental variable OMP_NUM_THREADS either globally (in your .bashrc/.cshrc) or when executing run_ginput.py by::

    OMP_NUM_THREADS=1 ./run_ginput.py ...

Note that numpy can use various C/Fortran backends, so depending on which one is used, you may need to use a different variable.
To identify which C/Fortran library numpy is linked to:

1. Activate whatever Python environment will run ginput (if applicable)
2. Run **python -c 'import numpy; print(numpy.__file__)'** from the command line. This will print the path to that environment's
   numpy installation (properly, the main .py file in it). 
3. Go to the numpy installation directory's **core** subfolder. In there, you should find two **multiarray*.so** files. Run **ldd**
   on the one that is _not_ the "tests" file. Look for the BLAS or MKL library. That should be the one that recognizes the
   environmental variable for the number of threads.


Error handling
**************

The ACOS interface is set up to catch `most` errors that only affect a single sounding and log them instead of letting them crash
the whole program. By default, a short version of the error will be written to the log. To print the full error traceback to the
log, increase the verbosity to maximum (**-vvvv**). This will still flag it rather than crashing, but provides more information
as to the cause. To raise errors normally (crashing on the first error), use the **--raise-errors** flag.
