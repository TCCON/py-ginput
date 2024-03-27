(usage-update-noaa)=
# Updating NOAA input data

`ginput` relies on NOAA measurements of CO2, CH4, and N2O for the baseline atmospheric mixing ratios. The repository include the necessary files with data up to 2018 (as of v1.1.7) and can extrapolate to later years as needed. However, using more recent data will provide more accurate priors (that better capture the atmospheric growth rate of these species). One option is for users to download new monthly average flask files and use them as described {ref}`below <usage-noaa-more-recent>`. The flask files are updated by NOAA about once per year. 

A second option is to use more rapidly updated (about once per month) in situ data. This requires additional steps to convert the rapid data to monthly averages in the same format as the flask data. This is done with the `update_hourly` subcommand of `run_ginput.py` and will be described in detail {ref}`below <usage-update-hourly>`.



(usage-noaa-more-recent)=
## Using more recent flask data

The repo includes NOAA flask data from 2018 and earlier in the {file}`ginput/data` directory, so that it can run with the user only responsible for obtaining the GEOS met files used for input. When `ginput` needs CO2, N2O, or CH4 beyond the end of 2018, it will extrapolate them. However, as usual, extrapolation provides poorer results that using the most recent data possible.


`ginput` natively reads the monthly average flask files available from [NOAA GML](https://gml.noaa.gov/dv/data/index.php?frequency=Monthly%2BAverages&type=Flask). To download more recent files from that link, select the gas(es) of interest in the "Parameter" column (carbon dioxide, nitrous oxide, or methane), then in the "Site" column select "MLO" (Mauna Loa) and "SMO" (American Samoa). For each site, this should provide exactly one file for download in the table at the bottom of the page. Save both the MLO and SMO files to your system.


A utility program is included in ginput/download/ to directly download the latest NOAA monthly averaged flask data, it is available via `run_ginput.py`. See usage info with:

`./run_ginput.py getnoaa -h`

### Generating .vmr files

In ginput v1.1.7, it is only possible to pass alternative NOAA files via the command line for the satellite subcommands (`oco`, `gosat`, `geocarb`). Because of the need to potentially pass multiple gases' files for regular {file}`.vmr` files, supporting this for the `vmr` and `rlvmr` subcommands requires additional development, but is planned for a future ginput version.

As a workaround for the time being, you can replace the flask files in the `ginput/data` subdirectory. These files are:

* {file}`ML_monthly_obs_co2.txt`, {file}`ML_monthly_obs_n2o.txt`, {file}`ML_monthly_obs_ch4.txt` for Mauna Loa files
* {file}`SMO_monthly_obs_co2.txt`, {file}`SMO_monthly_obs_n2o.txt`, {file}`SMO_monthly_obs_ch4.txt` for American Samoa files

```{note}
The update files must use EXACTLY the same names. Also, it is best to replace both files for a given gas at the same time so that they have the same end date.
```

As long as you have installed `ginput` in the recommended way (using `make install`) or have otherwise installed it into the environment in develop/editable mode (i.e. with `python setup.py develop` or `pip install -e .`), this is all you need to do. If you have installed it into the {file}`site-packages` directory of your environment (using `python setup.py install` or `pip install` for example), then you must re-run the same installation command you used before to update the data in the installation directory.

### Generating satellite priors

The `oco`, `gosat`, and `geocarb` subcommands that generate priors for those satellites accept paths to MLO/SMO monthly average files. For `oco` and `gosat`, this is straightforward as they only need CO2 files. If you have your updated MLO and SMO files for CO2 at {file}`/data/mlo_new_co2.txt` and {file}`/data/smo_new_co2.txt`, your call to `run_ginput.py` would look like:

```
$ ./run_ginput.py oco \
    --mlo-co2-file /data/mlo_new_co2.txt \
    --smo-co2-file /data/smo_new_co2.txt \
    # other additional arguments
```

Since `geocarb` generates CO2, CH4, and CO profiles, it uses these same command line options to specify the paths for both CO2 and CH4. To do so, your files must follow two rules:

* The file names must include the gas in lower case (i.e. "co2" or "ch4", not "CO2" or "CH4")
* The *only* difference in the paths to each file for the same NOAA site (MLO or SMO) must be the gas name. That is, if {file}`/data/mlo_new_co2.txt` is your CO2 Mauna Loa file, then your CH4 Mauna Loa file must be {file}`/data/mlo_new_ch4.txt`. Likewise, for American Samoa if the CO2 file is {file}`/data/smo_new_co2.txt`, the CH4 file must be {file}`/data/smo_new_ch4.txt`.

Then when passing the paths to `run_ginput.py`, they must include `{gas}` in the path as a placeholder for the gas name. Using the example paths in the second bullet above, this would look like:

```
$ ./run_ginput.py geocarb \
    --mlo-co2-file '/data/mlo_new_{gas}.txt' \
    --smo-co2-file '/data/smo_new_{gas}.txt' \
    # remaining arguments...
```

Note that the option flags (`--mlo-co2-file` and `--smo-co2-file`) do have "co2" in them even though they represent multiple gases; this is a holdover from the OCO/GOSAT commands. Also note that in the example the paths are enclosed in single quotes; this is a good idea to ensure your shell does not try to [expand the curly braces](https://www.linux.com/topic/desktop/all-about-curly-braces-bash/). The quotes may not be necessary, but are good insurance.

(usage-update-hourly)=
## Rapid updates with hourly data

If more frequent updates to the MLO/SMO record are desired, you will need to reach out to the NOAA GML group and request access to the rapid update in situ data. (I recommend looking at the contact information in the flask files for who to contact for hourly in situ analyzer data.) There are two elements of complexity in this approach:

1. The rapid update files are in a different format and time resolution than `ginput` expects.
2. The rapid data is preliminary, so later filtering may cause (minor) changes in the data.

(rapid-update-aside)=
```{admonition} Aside: why does the filtering matter?
In short, consistency. Imagine that you were generating priors to use for an operational algorithm, it was currently October, and you downloaded the rapid September data to use to generate some priors. Then, six months later, you want to regenerate the same priors but with rapid data downloaded in March. Further imagine that the later rapid data filtered out a few points in the September data, changing the September monthly average slightly. This means that you would not be able to regenerate exactly the same priors using the March version of the data as you did with the October version.

For operational products, this could be a problem, so `ginput` was designed with a mechanism to avoid that happening, and make it possible to get the latest rapid in situ data but keep the original data for months previously downloaded. Further down, we will discuss different workflows to update using rapid data depending on whether or not it is important for you to be able to exactly reproduce priors generated with an earlier version of the rapid data.
```

`ginput` includes a tool to address both of these, accessed via the `update_hourly` subcommand. It reads hourly files from NOAA in situ analyzers and does monthly averaging and *appends* the new monthly averages to an existing monthly averaging file. An example of the expected format for the hourly files is:

```
# data_fields: site year month day hour value std_dev n flag intake_ht inst
MLO 2010 01 01 00    387.820      0.060   1 .U.      40.00    LI1-4  
MLO 2010 01 01 01    387.660      0.040   1 .U.      40.00    LI1-4  
MLO 2010 01 01 02    387.630      0.050   1 .U.      40.00    LI1-4  
```

(This is the last line of the header and first 3 data rows.) Crucially, ginput needs data further back in time than these data are available, so these data must be combined with the flask measurements.

The `update_hourly` subcommand to `run_ginput.py` takes four arguments:

1. Which site (Mauna Loa = mlo, America Samoa = smo) is being processed
2. An existing monthly average file to append new data to
3. An hourly file to generate new monthly averages from
4. Where to output the extended monthly file

When called the `update_hourly` program will read in the hourly data for all months **after** the last month in the original monthly file, use that hourly data to compute new monthly averages, append those new monthly averages to those found in the original monthly file, and write out the combined data. To give an example, consider an example where we start with two files:

* {file}`mlo_monthly_flask.txt`, which has monthly averages up to and including December 2021
* {file}`mlo_hourly_data.txt`, which has hourly data starting in 2010 and going through October 2022.

Then if we ran the command:

```
$ ./run_ginput.py update_hourly \
    mlo \
    mlo_monthly_flask.txt \
    mlo_hourly_data.txt \
    extended_mlo_monthly.txt
```

the output file, {file}`extended_mlo_monthly.txt` would have the same monthly data as {file}`mlo_monthly_flask.txt` through December 2021. It would also have monthly averages for January through October 2022 calculated from the data in {file}`mlo_hourly_data.txt`. Even though the hourly file had earlier data, that data is ignored because the original monly file *already* had monthly averages for years prior to 2022.

When updating the SMO files, there is one other required input: the `--geos-2d-file-list` option. This must point to a file that lists all the 2D (Nx) assimilated GEOS-FP or GEOS FP-IT met files for the time period you need to calculate monthly averages for. For example, if you had {file}`smo_monthly.txt` which contained monthly averages through September 2022, and {file}`new_smo_hourly.txt` that has hourly data for October 2022, you would run:

```
$ ./run_ginput.py update_hourly \
    smo \
    smo_monthly.txt \
    new_smo_hourly.txt \
    extended_smo_monthly.txt \
    --geos-2d-file-list oct2022-geos.txt
```

where, assuming your 2D assimilation field GEOS files are kept in `/geos-fpit/met/Nx`,  {file}`oct2022-geos.txt` contains:

```
/geos-fpit/met/Nx/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20221001_0000.V01.nc4
/geos-fpit/met/Nx/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20221001_0003.V01.nc4
/geos-fpit/met/Nx/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20221001_0006.V01.nc4
/geos-fpit/met/Nx/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20221001_0009.V01.nc4

... 240 files omitted for brevity ...

/geos-fpit/met/Nx/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20221031_0012.V01.nc4
/geos-fpit/met/Nx/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20221031_0015.V01.nc4
/geos-fpit/met/Nx/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20221031_0018.V01.nc4
/geos-fpit/met/Nx/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20221031_0021.V01.nc4
/geos-fpit/met/Nx/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20221101_0000.V01.nc4
```

Note that you must include the first file from the next month (here the 00:00 Z file from 1 Nov 2022).

Now that we've covered the basics of how the `update_hourly` command works, let's look at two ways you can use it in practice.

### Workflow option 1: always use latest hourly data

This option is the easiest if it is not crucial that you be able to get exactly the same priors should you need to go back and regenerate priors for a previous date (see the {ref}`note above <rapid-update-aside>`). In this case, it is simplest to always use the flask data as your input monthly averages and the latest hourly data for the time period covered by it. Let's walk through an example for CO2 to make this clear.

As I am writing this it is November 2022, so let's assume that you obtained the latest hourly data from NOAA that extends through October 2022. The CO2 hourly data only goes back to 2010 (as far as I know), so we will need to use the NOAA flask data to extend further back in time. You would download the lastest MLO and SMO monthly average flask data from [here](https://gml.noaa.gov/dv/data/index.php?frequency=Monthly%2BAverages&type=Flask) (as of November 2022). Let's assume that you've done that, and so for MLO you have the flask file {file}`co2_mlo_surface-flask_1_ccgg_month.txt` and an hourly in situ file which you named `mlo_insitu_hourly_2022-10.txt`. 

```{note}
Even if you don't plan to generate priors for any time before 2010, it's still important to have the input NOAA data go as far back in time as possible. Parts of the `ginput` algorithm rely on averaging in much older air (to account for mixing in the stratosphere), so the further back in time your input data goes, the less `ginput` has to extrapolate and the more accurate your results will be.
```

Next, we want to truncate the flask data to remove data after the start of the hourly in situ data. Assuming that the hourly data starts in January 2010, this would mean editing your copy of {file}`co2_mlo_surface-flask_1_ccgg_month.txt` to remove all lines with data from January 2010 or later. Once done, you would call:

```
$ ./run_ginput.py update_hourly \
    mlo \
    co2_mlo_surface-flask_1_ccgg_month.txt \
    mlo_insitu_hourly_2022-10.txt \
    co2_mlo_combined_2022-10.txt
```

This will generate a new file, {file}`co2_mlo_combined_2022-10.txt` that has the flask data up to December 2009 and monthly averages calculated from the in situ hourly data for January 2010 through October 2022. You would repeat this step for the SMO files, just with the `--geos-2d-file-list` argument added as described above.

Now, let's imagine that it is December 2022, and you've downloaded the latest hourly data which now extends an additional month into November 2022, in a file called {file}`mlo_insitu_hourly_2022-11.txt`. To generate your new combined file, you simply repeat the above command, but with the new hourly file as the third argument, i.e.:

```
$ ./run_ginput.py update_hourly \
    mlo \
    co2_mlo_surface-flask_1_ccgg_month.txt \
    mlo_insitu_hourly_2022-11.txt \
    co2_mlo_combined_2022-11.txt
```

The new output file, {file}`co2_mlo_combined_2022-11.txt` will now have monthly averages through November 2022; however, some earlier months in 2022 may be very slightly different than in the previous output file ({file}`co2_mlo_combined_2022-10.txt`).



### Workflow option 2: update previous monthly file

This option is a little bit more complicated than Option 1, but allows you complete control over whether already-created monthly averages can be replaced. As in Option 1, you will still need to generate an initial monthly average file from flask data. As in the Option 1 example, let's assume that it is November 2022 and you have the latest hourly data from NOAA that extends through October 2022. Again, the CO2 hourly data only goes back to 2010, so we first obtain the CO2 flask data for MLO from [here](https://gml.noaa.gov/dv/data/index.php?frequency=Monthly%2BAverages&type=Flask) (as of November 2022). Again, let's assume that you've done that, and so for MLO you have the flask file {file}`co2_mlo_surface-flask_1_ccgg_month.txt` and an hourly in situ file which you named `mlo_insitu_hourly_2022-10.txt`.

As in Option 1, we'll generate our initial monthly average file by editing {file}`co2_mlo_surface-flask_1_ccgg_month.txt` to remove all lines with data from January 2010 or later and then calling:

```
$ ./run_ginput.py update_hourly \
    mlo \
    co2_mlo_surface-flask_1_ccgg_month.txt \
    mlo_insitu_hourly_2022-10.txt \
    co2_mlo_combined_2022-10.txt
```

This will generate a new file, {file}`co2_mlo_combined_2022-10.txt` which includes the flask data from {file}`co2_mlo_surface-flask_1_ccgg_month.txt` and monthly averages calculated from the hourly data in {file}`mlo_insitu_hourly_2022-10.txt`. So far, everything we've done has been identical to Option 1. 

Where this option differs is when the next month's hourly data becomes available. Let's imagine it is now December 2022, and you've downloaded the latest hourly data (which now extends into November 2022) in a file {file}`mlo_insitu_hourly_2022-11.txt`. In this option, we only want to add the November 2022 monthly average to our monthly average file, *not* regenerate all the monthly averages from 2010 on (as we did in Option 1). The command we use this time is:

```
$ ./run_ginput.py update_hourly \
    mlo \
    co2_mlo_combined_2022-10.txt \
    mlo_insitu_hourly_2022-11.txt \
    co2_mlo_combined_2022-11.txt
```

The difference is on the third line - instead of passing the *flask* file again (as we did in Option 1), we passed in our previous month's *output* file. The new output file will have monthly averages through November 2022, however unlike Option 1, all the monthly averages that we'd previous calculated from the October hourly file remain unchanged. Going forward, for each new month, you would pass the previous month's output file as the new input monthly average file.

## Other command line arguments

This section will describe why you might use the various optional command line arguments. Defer to the command line help in the event of any conflicts of information between that help text and this page.

* `-l`, `--last-month` - use this to limit how much data from the hourly file is appended to the monthly file. You may need to do this if you're using an old hourly file. For example, if your hourly file only has up to August 2022 complete, but you are running this program in November 2022, you would get an error about September and October missing. Limiting the output to August avoids that. You might also use this if you want control over when new hourly data gets added to the monthly data.
* `--allow-missing-hourly-times` - use this if the hourly file is missing one or more hours for a month that needs averaged, and you have confirmed that that is okay. Recommend you use this with `--last-month` to ensure only complete months get added.
* `--allow-missing-creation-date` - this program has some checks to ensure it doesn't try to use hourly data from a month that is incomplete because data for that month is still being taken. It relies on the header of the hourly file containing a line with "description_creation-time" in it to help with this check; if that line is missing, you will get an error. You can use this `--allow-missing-creation-date` flag to bypass looking for that header line, but be warned that doing so means you will could get monthly averages generated from incomplete months.
* `--no-limit-by-avail-data` - the default behavior of this program is to only use hourly data up to the start of the month the hourly file was created in. That is, if the creation time of the file (as specified in its header) is in November 2022, then this program will only output monthly averages up to October 2022. This (a) protects against generating a monthly average from an incomplete set of hourly data and (b) protects against using data that hasn't been at least preliminarily QA/QC'd by NOAA. This flag is available in case you receive a file from NOAA at the end of a month, for example, and are told it is okay to use to generate a monthly average. In general though, you should not use this flag.
* `--save-missing-geos-to` - when using the `--geos-2d-file-list` argument (required for SMO files), this program will check that all the expected 2D GEOS files are present (one every three hours for the month(s) monthly averages are being generated for). If any are missing, you can use this argument to write a list of missing files to a text file in order to help track them down.
* `allow-missing-geos-files` - also when using the `--geos-2d-file-list` argument, this flag allows you to bypass the check that all of the expected 2D files are listed. This is in place in case any 2D files are actually missing (e.g. Goddard failed to generate them, for some reason). Only use this flag in such a case.
* `--clobber` - use this to indicate that you want to allow the program to overwrite the output file, if it exists. The default is not to overwrite anything to prevent loss of data.
