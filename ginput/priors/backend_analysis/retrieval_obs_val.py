from collections import OrderedDict
import datetime as dt
from glob import glob

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import os
import re
import shutil

from ... import __version__
from ...common_utils import mod_utils, sat_utils, readers, writers
from ...mod_maker import mod_maker, tccon_sites
from .. import tccon_priors
from . import backend_utils as butils


_default_min_req_top_alt = 8.0


class VmrMergeError(Exception):
    pass


class AtmStitchError(Exception):
    pass


def generate_obspack_base_vmrs(obspack_dir, zgrid, std_vmr_file, save_dir, geos_dir, chm_dir=None, make_mod=True,
                               make_vmrs=True, overwrite=True):
    """
    Create the prior-only .vmr files for the times and locations of the .atm files in the given obspack directory

    :param obspack_dir: a path to a directory containing .atm files
    :type obspack_dir: str

    :param zgrid: any definition of a vertical grid understood by :func:`tccon_priors.generate_single_tccon_prior`

    :param save_dir: where to save the .mod and .vmr files produced. .mod files will be saved in an "fpit" subdirectory
     tree, .vmr files in "vmrs"
    :type save_dir: str

    :param geos_dir: path to the GEOS-FPIT data, must have Nv and Nx subdirectories.
    :type geos_dir: str

    :param chm_dir: if the chemistry GEOS-FPIT files are not in the same directory as the met files, then this must be
     the path to that directory. That directory must have a "Nv" subdirectory.
    :type chm_dir: str

    :param overwrite: controls whether to overwrite existing files. Only currently implemented for .mod files
    :type overwrite: bool

    :return: None. Writes .mod and .vmr files.
    """
    # For each .atm file, we will need to generate the corresponding .mod and .vmr files
    # We will then find the aircraft ceiling in the .atm file and extract the aircraft profile from below that. It will
    #   need binned/interpolated to the fixed altitude grid.
    # Finally we replace the data above the aircraft ceiling with the priors from the .vmr files. Then write those
    #   combined profiles to new .vmr files.
    obspack_files = list_obspack_files(obspack_dir)
    obspack_locations = construct_profile_locs(obspack_files)
    if make_mod:
        make_mod_files(obspack_locations=obspack_locations, save_dir=save_dir, geos_dir=geos_dir, chm_dir=chm_dir,
                       overwrite=overwrite)
    if make_vmrs:
        make_vmr_files(obspack_locations=obspack_locations, save_root_dir=save_dir, zgrid=zgrid, std_vmr_file=std_vmr_file)


def make_mod_files(obspack_locations, save_dir, geos_dir, chm_dir=None, overwrite=True):
    for date_range, loc_info in obspack_locations.items():
        loc_lon = loc_info['lon']
        loc_lat = loc_info['lat']
        loc_alt = [0.0 for x in loc_lon]
        loc_abbrev = loc_info['abbrev']

        if not overwrite:
            if len(loc_lat) != len(loc_lon) or len(loc_lat) != len(loc_abbrev):
                raise NotImplementedError('lon, lat, and abbrevs all assumed to be the same length')
            dates = pd.date_range(date_range[0], date_range[1], inclusive='left', freq='3h')
            exists = []
            for d in dates:
                for lon, lat, abbrev in zip(loc_lon, loc_lat, loc_abbrev):
                    mod_file = mod_utils.mod_file_name_for_priors(d, lat, lon, round_latlon=False)
                    exists.append(os.path.exists(os.path.join(save_dir, 'fpit', abbrev, 'vertical', mod_file)))
            if all(exists):
                print('.mod files for {}-{} already exist, not remaking'.format(*date_range))
                continue
            elif any(exists):
                print('{}/{} .mod files for {}-{} already exist, not remaking'.format(sum(exists), len(exists), *date_range))
                continue
            else:
                print('Must make .mod files for {}-{}'.format(*date_range))

        mod_maker.driver(date_range=date_range, met_path=geos_dir, chem_path=chm_dir, save_path=save_dir,
                         keep_latlon_prec=True, lon=loc_lon, lat=loc_lat, alt=loc_alt, site_abbrv=loc_abbrev,
                         mode='fpit-eta', include_chm=True)


def make_vmr_files(obspack_locations, save_root_dir, zgrid=None, std_vmr_file=None):
    vmr_save_dir = os.path.join(save_root_dir, 'vmrs')
    for date_range, loc_info in obspack_locations.items():
        loc_lon = loc_info['lon']
        loc_lat = loc_info['lat']
        loc_abbrev = loc_info['abbrev']

        tccon_priors.cl_driver(date_range=date_range, mod_root_dir=save_root_dir, save_dir=vmr_save_dir, zgrid=zgrid,
                               site_lat=loc_lat, site_lon=loc_lon, site_abbrev=loc_abbrev, keep_latlon_prec=True,
                               std_vmr_file=std_vmr_file)


def generate_obspack_modified_vmrs(obspack_dir, vmr_dir, save_dir, combine_method='weighted_bin',
                                   adjust_to_overworld=False, min_req_prof_alt=_default_min_req_top_alt,
                                   mod_dir=None, use_prior_fxn=None, filter_obs_fxn=None):
    """
    Generate .vmr files with the obspack data and prior profiles stiched together

    :param obspack_dir: the directory containing the .atm files
    :type obspack_dir: str

    :param vmr_dir: the directory containing the regular .vmr files, produced by :func:`make_vmr_files`.
    :type vmr_dir: str

    :param save_dir: the directory to save the stitched .vmr files to. Note: the files will have the same names as their
     un-stitched counterparts, so ``vmr_dir`` and ``save_dir`` may not be the same directory.
    :type save_dir: str

    :param combine_method: how the higher vertical resolution observational data is to be combined with the priors. If
     this is "interp" then the obs. data is just linearly interpolated to the prior profile altitudes. If this is
     "weighted_bin", then each level in the prior profile (below the obs. ceiling) is made a weighted sum of the obs.
     data, where the weights linearly decrease between the prior altitude level and the levels above and below it.
     "none" will not insert any obs. profile, it just saves the time-weighted .vmr prior profile.
    :type combine_method: str

    :param adjust_to_overworld: if True, the profile will be extended to the overworld as follows: if the obs. ceiling
     is above ``min_req_prof_alt``, then the top obs. bin will be extended to the tropopause. The concentration will
     then do a linear interpolation in theta between the tropopause and 380 K. If the top obs. bin is above 380 K, then
     nothing is changed. If the top obs. bin is below ``min_req_prof_alt``, then the prior is inserted above that level,
     no special treatment is used.
    :type adjust_to_overworld: bool

    :param min_req_prof_alt: the minimum altitude that the top alt. bin of the observations must be above for the
     special logic triggered by ``adjust_to_overworld`` to be used.
    :type min_req_prof_alt: float

    :param use_prior_fxn: a function that takes four inputs (altitudes of the .vmr prior levels, the prior profile in
     DMF, the combined obs+prior profile, and the .atm file name) and returns a modified combined profile with some
     levels replaced with the prior. It is called at the end of the interpolation/binning functions.
    :type use_prior_fxn: callable

    :param filter_obs_fxn: a function that takes five inputs (observation altitudes, observation concentrations, the 
     prior altitudes, the prior concentrations, and the .atm file name) and returns modified obs. altitudes and 
     concentrations. This is to provide a way to filter/replace bad data in the raw observations. It is called 
     immediately after the obs. data is loaded in the interpolation/binning functions, before any other action is taken.
    :type filter_obs_fxn: callable

    :return: none, writes new .vmr files, which also contain additional header information about the .atm files.
    """
    if os.path.samefile(vmr_dir, save_dir):
        raise ValueError('The vmr_dir and save_dir inputs may not point to the same directory')

    obspack_files = list_obspack_files(obspack_dir, include_floor_time=True)
    vmr_files = list_vmr_files(vmr_dir)
    if adjust_to_overworld:
        mod_files = list_mod_files(mod_dir)
    else:
        mod_files = None

    for obskey, obsfiles in obspack_files.items():
        prev_time, next_time, atm_date, prof_lon, prof_lat = obskey

        # For each gas we need to bin the aircraft data to the same vertical grid as the .vmr files (optionally do we
        # want to add a level right at the surface?) then replace those levels in the .vmr profiles and write new
        # .vmr files. We should include the .atm file names in the .vmr header, also which gases have obs data and the
        # obs ceiling.
        extra_header_info = OrderedDict()
        extra_header_info['observed_species'] = ''  # want first in the .vmr file, will fill in later
        extra_header_info['atm_files'] = ','.join(os.path.basename(f) for f in obsfiles)
        extra_header_info['combine_method'] = combine_method
        # will add the ceilings for each gas in the loop, in case they differ

        # The first step is to weight the vmr profiles to the time of the .atm file. Remember we're using the floor time
        # because we assume that temporal variation will be most important at the surface. Need to adjust end time
        # because it's 3 hours past the actual next file to work with mod_maker's end time being exclusive.
        wt = sat_utils.time_weight_from_datetime(atm_date, prev_time, next_time - dt.timedelta(hours=3))
        # Get the two .vmr files that bracket the observation
        matched_vmr_files = match_atm_vmr(obskey, vmr_files)
        vmrdat = weighted_avg_vmr_files(matched_vmr_files[0], matched_vmr_files[1], wt)
        vmr_trop_alt = vmrdat['scalar']['ZTROP_VMR']
        vmrz = vmrdat['profile'].pop('Altitude')
        
        if adjust_to_overworld:
            matched_mod_files = match_atm_vmr(obskey, mod_files)
            moddat = weighted_avg_mod_files(matched_mod_files[0], matched_mod_files[1], wt)
            # Must have the model data on the same altitude grid as the priors.
            moddat = mod_utils.interp_to_zgrid(moddat['profile'], vmrz)
            prof_theta = moddat['PT']
        else:
            prof_theta = None


        observed_species = []
        for gas_file in obsfiles:
            gas_name = _get_atm_gas(gas_file)
            observed_species.append(gas_name.upper())
            vmr_prof = vmrdat['profile'][gas_name.upper()]

            if combine_method == 'interp':
                combo_prof, obs_ceiling, adj_flag = interp_obs_to_vmr_alts(gas_file, vmrz, vmr_prof, vmr_theta=prof_theta,
                                                                           vmr_trop_alt=vmr_trop_alt,
                                                                           force_prior_fxn=use_prior_fxn,
                                                                           filter_obs_fxn=filter_obs_fxn,
                                                                           adjust_to_overworld=adjust_to_overworld,
                                                                           min_req_top_alt=min_req_prof_alt)
            elif combine_method == 'weighted_bin':
                combo_prof, obs_ceiling, adj_flag = weighted_bin_obs_to_vmr_alts(gas_file, vmrz, vmr_prof, vmr_theta=prof_theta,
                                                                                 vmr_trop_alt=vmr_trop_alt,
                                                                                 force_prior_fxn=use_prior_fxn,
                                                                                 filter_obs_fxn=filter_obs_fxn,
                                                                                 adjust_to_overworld=adjust_to_overworld,
                                                                                 min_req_top_alt=min_req_prof_alt)
            elif combine_method == 'none':
                combo_prof = vmr_prof
                obs_ceiling = np.nan
                adj_flag = 0
            else:
                raise ValueError('{} is not one of the allowed combine_method values'.format(combine_method))

            vmrdat['profile'][gas_name.upper()] = combo_prof
            extra_header_info['{}_ceiling'.format(gas_name.upper())] = '{:.3f} km'.format(obs_ceiling)
            extra_header_info['{}_adjusted'.format(gas_name.upper())] = '{:d}'.format(adj_flag)

        extra_header_info['observed_species'] = ','.join(observed_species)

        prof_lon = mod_utils.format_lon(prof_lon)
        prof_lat = mod_utils.format_lat(prof_lat)
        vmr_name = mod_utils.vmr_file_name(atm_date, lon=prof_lon, lat=prof_lat, date_fmt='%Y%m%d_%H%M',
                                           keep_latlon_prec=True, in_utc=True)
        vmr_name = os.path.join(save_dir, vmr_name)
        writers.write_vmr_file(vmr_name, tropopause_alt=vmr_trop_alt, profile_date=atm_date,
                                                   profile_lat=prof_lat, profile_alt=vmrz, profile_gases=vmrdat['profile'],
                                                   extra_header_info=extra_header_info)


def merge_vmr_files(vmr_in_dir, vmr_out_dir):
    def get_bdy_geos_times(ts):
        ts = pd.Timestamp(ts.year, ts.month, ts.day, ts.hour // 3 * 3)
        start = ts - pd.Timedelta(hours=3)
        end = ts + pd.Timedelta(hours=6)
        return start, end

    def merge_vmr_files(main_file, *additional_files):
        print('Merging {} into {}'.format(', '.join([os.path.basename(f) for f in additional_files]), os.path.basename(main_file)))
        main_vmr = readers.read_vmr_file(main_file, lowercase_names=False)
        for other_file in additional_files:
            other_vmr = readers.read_vmr_file(other_file, lowercase_names=False)
            other_species = other_vmr['scalar']['observed_species'].split(',')
            main_vmr['scalar']['observed_species'] += ',' + other_vmr['scalar']['observed_species']
            main_vmr['scalar']['atm_files'] += ',' + other_vmr['scalar']['atm_files']
            for k, v in other_vmr['scalar'].items():
                # need to copy the new species ceiling and adjusted flag values
                if k not in main_vmr['scalar']:
                    main_vmr['scalar'][k] = v
            for gas in other_species:
                gas = gas
                main_vmr['profile'][gas] = other_vmr['profile'][gas]

        # need to extract some of the values to pass back to the vmr writing function
        main_vmr['scalar'].pop('GINPUT_VERSION')
        ztrop = main_vmr['scalar'].pop('ZTROP_VMR')
        date_vmr = mod_utils.decimal_year_to_date(main_vmr['scalar'].pop('DATE_VMR'))
        lat_vmr = main_vmr['scalar'].pop('LAT_VMR')
        z_vmr = main_vmr['profile'].pop('Altitude')
        new_name = os.path.join(vmr_out_dir, os.path.basename(main_file))
        writers.write_vmr_file(new_name, tropopause_alt=ztrop, profile_date=date_vmr, profile_lat=lat_vmr,
                                                   profile_alt=z_vmr, profile_gases=main_vmr['profile'],
                                                   extra_header_info=main_vmr['scalar'])

    vmr_files = sorted(glob(os.path.join(vmr_in_dir, '*.vmr')))
    # Loop through the vmr files. For each file, record its date and species from observation. Then go back through and
    # merge files that are close enough in time to be considered simultaneous and which have different species. If we
    # have two files that have the same species and going forward or backward two GEOS time steps would overlap, then
    # error, because we can guarantee that the retrieval would use the right prior for all measurements we associate
    # with that observation, i.e. if one is at 1400 and the other 1700, then the measurements at 1600-1630 that go with
    # the 1700 observation would use the 1400 prior, because it should be linked to the 15Z file.
    vmr_info = dict()
    all_species = set()
    for f in vmr_files:
        vmrs = readers.read_vmr_file(f)
        vmr_date = mod_utils.find_datetime_substring(f, out_type=pd.Timestamp)
        vmr_species = vmrs['scalar']['observed_species'].split(',')
        site = mod_utils.find_lat_substring(f) + '_' + mod_utils.find_lon_substring(f)
        vmr_info[vmr_date] = {'file': f, 'species': vmr_species, 'site': site}
        all_species = all_species.union(vmr_species)

    # Convert into a dataframe - that way we can iterate over the lines
    df_dict = {gas: 0 for gas in all_species}
    df_dict['site'] = ''
    df_dict['file'] = ''
    df_dates = list(vmr_info.keys())
    vmr_df = pd.DataFrame(df_dict, index=df_dates)
    for k, info in vmr_info.items():
        vmr_df.loc[k, 'file'] = info['file']
        vmr_df.loc[k, 'site'] = info['site']
        for gas in info['species']:
            vmr_df.loc[k, gas] = 1

    # Now loop over each line, if there are no files within two geos timesteps, just copy the .vmr file to the output
    # directory. If there are two that don't share any species, merge. If there are two that share species, error.
    for site, site_df in vmr_df.groupby('site'):
        for time in site_df.index:
            start, stop = get_bdy_geos_times(time)
            sub_df = site_df.loc[slice(start, stop), :]
            if sub_df.shape[0] == 1:
                shutil.copy2(sub_df['file'].item(), vmr_out_dir)
            elif (sub_df.loc[:,all_species].sum(axis=0) > 1).any():
                raise VmrMergeError('Multiple .vmr files with the same species found in the time range {} to {} at {}:\n'
                                    ' * {}\nArchive the .atm file for one of the .vmr files, deleted the stitched .vmr '
                                    'files, and rerun.'.format(
                    start, stop, site.replace('_', ', '), '\n * '.join(sub_df['file'].to_numpy())
                ))
            else:
                merge_vmr_files(*sub_df['file'].to_numpy())


def add_strat_to_atm_files(obspack_in_dir, obspack_out_dir, vmr_dir):
    def get_scale_factor(atm_colname):
        scale_dict = {'ppm': 1e6, 'ppmv': 1e6, 'ppb': 1e9, 'ppbv': 1e9}
        unit = atm_colname.split('_')[-1]
        return scale_dict[unit]

    def write_atm_file_simple(out_file, data_df, header_dict):
        # TODO: unify with aircraft_preprocessing writer
        with open(out_file, 'w') as wobj:
            header_dict = header_dict.copy()
            header_dict['ginput_version'] = __version__
            for key, value in header_dict.items():
                if key == 'description':
                    wobj.write(value + '\n')
                else:
                    wobj.write('{:30} {}\n'.format(key+':', value))
            wobj.write('-' * 100 + '\n')
            wobj.write(','.join(data_df.columns) + '\n')
            for index, row in data_df.iterrows():
                # can't use to_csv b/c want each column to have a specific format
                fmt = '{:.0f},{:.5g},{:.5g},{:.2f}'
                if row.size > 4:
                    # .atm files always have an H2O column but only have a fifth column
                    # if there's a trace gas measurement.
                    fmt += ',{:.2f}'
                wobj.write(fmt.format(*row.to_numpy()) + '\n')

    # Loop through the .vmr files, for each one, read the corresponding .atm files, interpolate the corresponding gas to
    # the .atm grid, then save the new atm file in the output directory.
    vmr_files = glob(os.path.join(vmr_dir, '*.vmr'))
    completed_atm_files = set()
    for this_vmr_file in vmr_files:
        vmrdat = readers.read_vmr_file(this_vmr_file)
        vmrz = vmrdat['profile']['altitude'] * 1000  # atm file altitudes are in meters
        species = vmrdat['scalar']['observed_species'].split(',')
        atm_files = vmrdat['scalar']['atm_files'].split(',')
        for gas, f in zip(species, atm_files):
            gas = gas.lower()
            if f in completed_atm_files:
                raise AtmStitchError('{} seems to be referenced by multiple .vmr files'.format(f))
            completed_atm_files.add(f)
            f_full = os.path.join(obspack_in_dir, f)
            atmdat, atmhead = butils.read_atm_file(f_full, keep_header_strings=True)
            atmz = atmdat.iloc[:, 0]
            atm_ceiling = float(atmhead['aircraft_ceiling_m' if 'aircraft_ceiling_m' in atmhead else 'ceiling_m'])
            xx_replace = atmz > atm_ceiling

            # interpolate the prior to the stratospheric levels
            data_col = atmdat.columns[-1]
            vmr_scale = get_scale_factor(data_col)
            atmdat.loc[xx_replace, data_col] = np.interp(atmz[xx_replace], vmrz, vmr_scale*vmrdat['profile'][gas])
            write_atm_file_simple(os.path.join(obspack_out_dir, f), atmdat, atmhead)


def plot_vmr_comparison(obspack_dir, vmr_dirs, save_file, plot_if_not_measured=True, max_alt=100.0, log_scale_profs=None):
    """
    Create a .pdf file comparing observed profiles with multiple .vmr directories

    :param vmr_dirs: a dictionary of directories containing .vmr files. Each directory must have the same .vmr files.
     The keys will be used as the legend names for the profiles read from those .vmr files.
    :type vmr_dirs: dict

    :param save_file: the path to save the .pdf of the profiles as.
    :type save_file: str

    :param plot_if_not_measured: set to ``False`` to omit panels for gas profile that don't have observational data.
     Otherwise the .vmr profiles are plotted regardless.
    :type plot_if_not_measured: bool

    :param log_scale_profs: a collection of gas names to use log scale on the x-axis instead of linear. Names must be
     upper cases.
    :type log_scale_profs: list(str)

    :return: none, writes a .pdf file. Each page will have up to four plots, one each for CO2, N2O, CH4, and CO. If
     that profile was not measured and ``plot_if_not_measured`` is ``False``, the corresponding panel will be omitted.
    """
    # Loop through the modified .vmr files. For each one, read in the .atm files that correspond to it, load them.
    # Use PdfPages (https://matplotlib.org/3.1.1/gallery/misc/multipage_pdf.html) to put one set of plots per page,
    # always arrange:
    #   CO2 N2O
    #   CH4 CO
    # On each, plot the actual observed profile plus the profiles defined by the vmr_dirs dict
    gas_order = ('CO2', 'N2O', 'CH4', 'CO', 'H2O')
    gas_scaling = {'ppm': 1e6, 'ppb': 1e9, '%': 1e2}
    gas_units = {'CO2': 'ppm', 'N2O': 'ppb', 'CH4': 'ppb', 'CO': 'ppb', 'H2O': '%'}
    vmr_color = ('b', 'r', 'g')
    vmr_marker = ('x', '+', '*')
    x_scales = {k: 'linear' for k in gas_order}
    if log_scale_profs is not None:
        for k in log_scale_profs:
            x_scales[k] = 'log'

    first_key = list(vmr_dirs.keys())[0]
    vmr_files = list_vmr_files(vmr_dirs[first_key])

    pbar = mod_utils.ProgressBar(len(vmr_files), style='counter', suffix='profiles completed')
    with PdfPages(save_file) as pdf:
        for fidx, fname in enumerate(vmr_files.values()):
            pbar.print_bar(fidx)
            this_atm_file = None
            basename = os.path.basename(fname)
            vmr_info = readers.read_vmr_file(fname)

            matched_atm_files = _organize_atm_files_by_species(vmr_info['scalar']['atm_files'].split(','))

            fig = plt.figure()
            for iax, gas in enumerate(gas_order, 1):
                if gas not in matched_atm_files and not plot_if_not_measured:
                    continue

                unit = gas_units[gas]
                scale = gas_scaling[unit]

                ax = fig.add_subplot(3, 2, iax)
                if gas in matched_atm_files:
                    this_atm_file = os.path.join(obspack_dir, matched_atm_files[gas])
                    obsz, obsprof, obsfloor, obsceil = _load_obs_profile(this_atm_file, limit_below_ceil=False)
                    zzobs = (obsz <= max_alt) & (obsz <= obsceil)
                    ax.plot(obsprof[zzobs]*scale, obsz[zzobs], color='k', marker='o', label='Observed')
                    zzold = (obsz <= max_alt) & (obsz > obsceil)
                    ax.plot(obsprof[zzold]*scale, obsz[zzold], color='gray', marker='o', label='GGG2014 prior')
                    ax.axhline(obsceil, 0.1, 0.9, color='k', linestyle=':', label='Obs. ceiling')

                for i, (label, vdir) in enumerate(vmr_dirs.items()):
                    vmrdat = readers.read_vmr_file(os.path.join(vdir, basename), lowercase_names=True)
                    vmrz, vmrprof = vmrdat['profile']['altitude'], vmrdat['profile'][gas.lower()]
                    zz = vmrz <= max_alt
                    ax.plot(vmrprof[zz]*scale, vmrz[zz], color=vmr_color[i], marker=vmr_marker[i], label=label)

                tropz = vmrdat['scalar']['ztrop_vmr']  # should be the same in all files
                ax.axhline(tropz, 0.1, 0.9, color='orange', linestyle='--', label='Tropopause')

                ax.set_xlabel(r'[{}] ({})'.format(gas.upper(), unit))
                ax.set_xscale(x_scales[gas.upper()])
                ax.set_ylabel('Altitude (km)')
                ax.legend()
                ax.grid()
                ax.set_title(gas.upper())

            # We can use any of the .atm files read in for this profile. this_atm_file gets reset to None
            # at the beginning of the loop over vmr_files so we're sure to not use an .atm file from a 
            # previous .vmr file
            _, atm_header = butils.read_atm_file(this_atm_file)

            fig.set_size_inches(16, 16)
            fig.suptitle('{date} - {tccon} ({lon}, {lat})'.format(
                date=get_atm_date(atm_header), tccon=atm_header['TCCON_site_name'],
                lon=mod_utils.format_lon(atm_header['TCCON_site_longitude_E'], prec=2),
                lat=mod_utils.format_lat(atm_header['TCCON_site_latitude_N'], prec=2)
            ))

            pdf.savefig(fig)
            plt.close(fig)
    pbar.finish()


def list_obspack_files(obspack_dir, include_floor_time=False):
    """
    Create a dictionary of obspack files

    :param obspack_dir: the directory to find the .atm files
    :type obspack_dir: str

    :return: a dictionary with keys (start_geos_time, stop_geos_time, lon_string, lat_string) and the values are lists
     of files. This format allows there to be different gases for different date/locations.
    """
    def keys_close(k, d):
        for dk in d:
            if abs(dk[2] - k[2]) < dt.timedelta(minutes=5):
                return dk

        return False

    if include_floor_time:
        key_fxn = lambda f: _make_atm_key(f, True)
        key_test_fxn = keys_close
    else:
        key_fxn = _make_atm_key
        key_test_fxn=None
    return _make_file_dict(obspack_dir, '.atm', key_fxn, key_test_fxn)


def list_vmr_files(vmr_dir):
    """
    Create a dictionary of .vmr files

    :param vmr_dir: the directory to find the .vmr files
    :type vmr_dir: str

    :return: a dictionary with keys (date_time, lon_string, lat_string) and the values are the corresponding .vmr file.
    """
    file_dict = _make_file_dict(vmr_dir, '.vmr', _make_vmr_key)
    for k, v in file_dict.items():
        if len(v) != 1:
            raise NotImplementedError('>1 .vmr file found for a given datetime/lat/lon')
        file_dict[k] = v[0]

    return file_dict


def list_mod_files(mod_dir):
    """
    Create a dictionary of .mod files

    :param mod_dir: the directory containing the individual site directories.
    :type mod_dir: str

    :return: a dictionary with keys (date_time, lon_string, lat_string) and the values are the corresponding .mod file.
    """
    file_dict = dict()
    for site_dir in mod_utils.iter_mod_dirs(mod_dir, path='vertical'):
        site_file_dict = _make_file_dict(site_dir, '.mod', _make_vmr_key)
        for k, v in site_file_dict.items():
            if k in file_dict:
                raise KeyError('Multiple .mod files have the same key')
            elif len(v) != 1:
                raise NotImplementedError('>1 .mod file found for a given datetime/lat/lon')
            file_dict[k] = v[0]

    return file_dict


def _make_file_dict(file_dir, file_extension, key_fxn, key_test_fxn=None):
    """
    Create a dictionary of files with keys describing identifying information about them.

    :param file_dir: directory containing the files
    :param file_extension: extension of the files. May include or omit the .
    :param key_fxn: a function that, given a file name, returns the key to use for it.
    :return: the dictionary of files. Each value will be a list.
    """
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    if key_test_fxn is None:
        def key_test_fxn(k, d):
            return k if k in d else False

    files = sorted(glob(os.path.join(file_dir, '*{}'.format(file_extension))))
    files_dict = dict()
    pbar = mod_utils.ProgressBar(len(files), style='counter', prefix='Parsing {} file'.format(file_extension))
    for i, f in enumerate(files):
        pbar.print_bar(i)
        key = key_fxn(f)
        extant_key = key_test_fxn(key, files_dict)
        if extant_key:
            files_dict[extant_key].append(f)
        else:
            files_dict[key] = [f]

    pbar.finish()

    return files_dict


def match_atm_vmr(atm_key, vmr_files):
    """
    Find the .vmr files that bracket a given .atm file in time

    :param atm_key: the key from the dictionary of .atm files. Should contain the preceeding and following GEOS times,
     floor_time, longitude, and latitude in that order.
    :type atm_key: tuple

    :param vmr_files: the dictionary of .vmr files, keyed with (datetime, lon, lat) tuples
    :type vmr_files: dict

    :return: the two matching .vmr files, in order, the one before and the one after
    :rtype: str, str
    """
    vmr_key_1 = (atm_key[0], atm_key[3], atm_key[4])
    # the atm keys' second value is the end date of the mod_maker date range, which is exclusive
    # so it's an extra 3 hours in the future
    vmr_key_2 = (atm_key[1] - dt.timedelta(hours=3), atm_key[3], atm_key[4]) 
    return vmr_files[vmr_key_1], vmr_files[vmr_key_2]


def construct_profile_locs(obspack_dict):
    """
    Construct a dictionary that groups profiles by time

    :param obspack_dict: the dictionary of files returned by :func:`list_obspack_files`.
    :type obspack_dict: dict

    :return: a dictionary with the GEOS start and stop times as keys in a tuple. Each value will be a dictionary with
     keys "lon", "lat", and "abbrev" that will be lists of longitudes and latitudes (as floats) that fall in the range
     of GEOS times and the matched TCCON abbreviations.
    :rtype: dict
    """
    loc_dict = dict()
    for key, files in obspack_dict.items():
        start, stop, lonstr, latstr = key
        new_key = (start, stop)
        if new_key not in loc_dict:
            loc_dict[new_key] = {'lon': [], 'lat': [], 'abbrev': []}
        loc_dict[new_key]['lon'].append(mod_utils.format_lon(lonstr))
        loc_dict[new_key]['lat'].append(mod_utils.format_lat(latstr))
        try:
            loc_dict[new_key]['abbrev'].append(_lookup_tccon_abbrev(files[0]))
        except KeyError:
            loc_dict[new_key]['abbrev'].append('xx')

    return loc_dict


def _make_atm_key(file_or_header, include_floor_time=False):
    """
    Make a dictionary key for an .atm file

    :param file_or_header: the .atm file to make a key for. Either give the path to the file or the header information
    :type file_or_header: str or dict

    :return: a key consisting of the GEOS date range to pass to mod_maker (as separate entries), the longitude string
     and the latitude string
    :rtype: tuple
    """
    if isinstance(file_or_header, str):
        _, file_or_header = butils.read_atm_file(file_or_header)

    lon = mod_utils.format_lon(file_or_header['TCCON_site_longitude_E'], prec=2)
    lat = mod_utils.format_lat(file_or_header['TCCON_site_latitude_N'], prec=2)

    floor_time = get_atm_date(file_or_header)
    start_geos_time = _to_3h(floor_time)
    floor_time = _to_minute(floor_time)
    # We will produce the profiles interpolated to the floor time of the profile. I chose that because we don't have
    # times for each level in the .atm file, so we can't interpolate each level separately, and we probably want to
    # get closer in time to the floor than the ceiling, as I expect the surface will be more variable.
    #
    # Since mod_maker treats the end date as exclusive, we need to go 2 GEOS times past the floor time to produce the
    # two .mod files that bracket it.
    stop_geos_time = start_geos_time + dt.timedelta(hours=6)

    if include_floor_time:
        return start_geos_time, stop_geos_time, floor_time, lon, lat
    else:
        return start_geos_time, stop_geos_time, lon, lat


def _make_vmr_key(filename):
    """
    Make a dictionary key for a .vmr file

    :param filename: the path to the .vmr file to make a key for.
    :type filename: str

    :return: a key consisting of the GEOS date, the longitude string
     and the latitude string
    :rtype: tuple
    """
    filename = os.path.basename(filename)
    file_date = mod_utils.find_datetime_substring(filename, out_type=dt.datetime)
    file_lon = mod_utils.find_lon_substring(filename, to_float=False)
    file_lat = mod_utils.find_lat_substring(filename, to_float=False)

    # remove leading 0s to be consistent with how the aircraft keys are made, but leave one zero before the decimal if
    # there are only zeros before the decimal
    file_lon = re.sub(r'^0+(?=\d)', '', file_lon)
    file_lat = re.sub(r'^0+(?=\d)', '', file_lat)
    return file_date, file_lon, file_lat


def get_atm_date(file_or_header):
    """
    Get the representative datetime of an .atm file

    :param file_or_header: the path to the .atm file or the dictionary of header information
    :type file_or_header: str or dict

    :return: the datetime for the observation at the bottom of the profile
    :rtype: :class:`datetime.datetime`
    """
    if isinstance(file_or_header, str):
        _, file_or_header = butils.read_atm_file(file_or_header)

    floor_time_key = _find_key(file_or_header, r'floor_time_UTC$')
    return file_or_header[floor_time_key]


def _find_key(dict_in, key_regex):
    """
    Find a key in a dictionary matching a given regex

    :param dict_in: the dictionary to search
    :type dict_in: dict

    :param key_regex: the regular expression to use
    :type key_regex: str

    :return: the matching key
    :rtype: str
    :raises ValueError: if not exactly one key is found
    """
    keys = list(dict_in.keys())
    found_key = None
    for k in keys:
        if re.search(key_regex, k) is not None:
            if found_key is None:
                found_key = k
            else:
                raise ValueError('Multiple keys matching "{}" found'.format(key_regex))

    if found_key is None:
        raise ValueError('No key matching "{}" found'.format(key_regex))
    return found_key


def _to_3h(dtime):
    """
    Round a datetime to the previous multiple of 3 hours

    :param dtime: the datetime
    :type dtime: datetime-like

    :return: the rounded datetime
    :rtype: :class:`datetime.datetime`
    """
    hr = (dtime.hour // 3) * 3
    return dt.datetime(dtime.year, dtime.month, dtime.day, hr)


def _to_minute(dtime):
    return dt.datetime(dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute)


def _lookup_tccon_abbrev(file_or_header, max_dist=0.1):
    """
    Look up the abbreviation of the TCCON site colocated with a .atm file

    :param file_or_header: the path to the .atm file or the header dictionary from it
    :type file_or_header: str or dict

    :param max_dist: the maximum distance (in degrees) away from a TCCON site the profile may be. Note that this is only
     used if the TCCON name in the .atm file is not recognized.
    :type max_dist: float

    :return: the two-letter abbreviation of the closest site
    :rtype: str
    """
    if isinstance(file_or_header, str):
        _, file_or_header = butils.read_atm_file(file_or_header)

    # First try looking up by the TCCON site name.
    atm_date = get_atm_date(file_or_header)
    site_dict = tccon_sites.tccon_site_info_for_date(atm_date)
    try:
        site_name = file_or_header['TCCON_site_name'].lower()
    except KeyError:
        site_name = None
    else:
        for key, info in site_dict.items():
            if info['name'].lower() == site_name:
                return key

    # Okay, couldn't find by name - either the .atm file didn't have the site name or it wasn't in the site dictionary.
    # Fall back on lat/lon
    atm_lon = file_or_header['TCCON_site_longitude_E']
    if atm_lon > 180:
        atm_lon -= 180
    atm_lat = file_or_header['TCCON_site_latitude_N']

    best_key = None
    best_r = 1000.0
    for key, info in site_dict.items():
        r = (atm_lon - site_dict['lon_180'])**2 + (atm_lat - site_dict['lat'])
        if r < max_dist and r < best_r:
            best_key = key
            best_r = r

    if best_key is None:
        raise KeyError('Could not find a TCCON site near {} {}'.format(
            mod_utils.format_lon(atm_lon, prec=2), mod_utils.format_lat(atm_lat, prec=2)))

    return best_key


def weighted_avg_vmr_files(vmr1, vmr2, wt):
    """
    Calculate the time-weighted average of the quantities in two .vmr files

    :param vmr1: the path to the earlier .vmr file
    :type vmr1: str

    :param vmr2: the path to the later .vmr file
    :type vmr2: str

    :param wt: the weight to apply to each .vmr profile/scalar value. Applied as :math:`w * vmr1 + (1-w) * vmr2`
    :type wt: float

    :return: the dictionary, as if reading the .vmr file, with the time average of the two files.
    :rtype: dict
    """
    vmrdat1 = readers.read_vmr_file(vmr1, lowercase_names=False)
    vmrdat2 = readers.read_vmr_file(vmr2, lowercase_names=False)
    return weighted_avg_dicts(vmrdat1, vmrdat2, wt)


def weighted_avg_mod_files(mod1, mod2, wt):
    def load_mod_file(fname):
        dat = readers.read_mod_file(fname)
        return dat

    moddat1 = load_mod_file(mod1)
    moddat2 = load_mod_file(mod2)
    return weighted_avg_dicts(moddat1, moddat2, wt)


def weighted_avg_dicts(dict1, dict2, wt):
    avg = OrderedDict()
    for k in dict1:
        group = OrderedDict()
        for subk in dict1[k]:
            data1 = dict1[k][subk]
            data2 = dict2[k][subk]
            try:
                group[subk] = wt * data1 + (1 - wt) * data2
            except TypeError as err:
                print('Cannot weight {}/{}: {}'.format(k, subk, err.args[0]))
        avg[k] = group

    return avg


def _load_obs_profile(obsfile, limit_below_ceil=False):
    """
    Load data from an .atm

    :param obsfile: the path to the .atm file
    :type obsfile: str

    :param limit_below_ceil: set to ``True`` to only return altitude and profile values below the observation ceiling
    :type limit_below_ceil: bool

    :return: the altitude vector, concentration vector, floor altitude, and ceiling altitude. All altitudes are in
     kilometers, the concentrations are in mole fraction.
    """
    conc_scaling = {'ppm': 1e-6, 'ppmv': 1e-6, 'ppb': 1e-9, 'ppbv': 1e-9}
    obsdat, obsinfo = butils.read_atm_file(obsfile)

    # If "Altitude_m" is not the key for altitude, or the concentration is not the part after the last _ in the column
    # header, we'll get a key error so we should be able to catch different units.
    obsz = obsdat['Altitude_m'].to_numpy() * 1e-3

    conc_key = obsdat.keys()[-1]
    unit = conc_key.split('_')[-1]
    obsconc = obsdat[conc_key].to_numpy() * conc_scaling[unit]

    floor_key = _find_key(obsinfo, r'floor_m$')
    ceil_key = _find_key(obsinfo, r'ceiling_m$')
    floor_km = obsinfo[floor_key]*1e-3
    ceil_km = obsinfo[ceil_key] * 1e-3

    if limit_below_ceil:
        zz = obsz <= ceil_km
        obsz = obsz[zz]
        obsconc = obsconc[zz]

    # We don't want NaNs in the aircraft data. 
    nans = np.isnan(obsconc)
    obsz = obsz[~nans]
    obsconc = obsconc[~nans]

    # We want the aircraft data to be ordered surface-to-ceiling, monotonically increasing
    sort_inds = np.argsort(obsz)  #  can use this if there's a case where the profile is read in upside down
    obsz = obsz[sort_inds]
    obsconc = obsconc[sort_inds]

    return obsz, obsconc, floor_km, ceil_km


def interp_obs_to_vmr_alts(obsfile, vmralts, vmrprof, force_prior_fxn=None, filter_obs_fxn=None,
                           adjust_to_overworld=False, min_req_top_alt=_default_min_req_top_alt, vmr_trop_alt=None,
                           vmr_theta=None):
    """
    Stitch together the observed profile and prior profile with linear interpolation

    :param obsfile: the path to the .atm file to use
    :type obsfile: str

    :param vmralts: the vector of altitude levels in the .vmr priors, in kilometers
    :type vmralts: array-like

    :param vmrprof: the vector of concentrations in the .vmr priors, in mole fraction
    :type vmrprof: array-like

    :param force_prior_fxn: a function that takes four inputs (altitudes of the .vmr prior levels, the prior profile in
     DMF, the combined obs+prior profile, and the .atm file name) and returns a modified combined profile with some
     levels replaced with the prior.
    :type force_prior_fxn: callable

    :param filter_obs_fxn: a function that takes five inputs (observation altitudes, observation concentrations, the
     prior altitudes, the prior concentrations, and the .atm file name) and returns modified obs. altitudes and
     concentrations. This is to provide a way to filter/replace bad data in the raw observations. It is called
     immediately after the obs. data is loaded, before any other action is taken.
    :type filter_obs_fxn: callable

    :param adjust_to_overworld: if True, the profile will be extended to the overworld as follows: if the obs. ceiling
     is above ``min_req_prof_alt``, then the top obs. bin will be extended to the tropopause. The concentration will
     then do a linear interpolation in theta between the tropopause and 380 K. If the top obs. bin is above 380 K, then
     nothing is changed. If the top obs. bin is below ``min_req_prof_alt``, then the prior is inserted above that level,
     no special treatment is used.
    :type adjust_to_overworld: bool

    :param min_req_top_alt: the minimum altitude that the top alt. bin of the observations must be above for the
     special logic triggered by ``adjust_to_overworld`` to be used.
    :type min_req_top_alt: float

    :param vmr_trop_alt: the altitude (in the same units as ``vmralts``) of the tropopause as read from the .vmr file.
     Not needed if ``adjust_to_overworld == False``.
    :type vmr_trop_alt: float

    :param vmr_theta: the profile of potential temperature (in K) on the vertical levels defined by ``vmralts``.
     Not needed if ``adjust_to_overworld == False``.
    :type vmr_theta: array-like

    :return: the combined observation + prior profile on the .vmr levels, and the observation ceiling (in kilometers)
    """
    obsz, obsprof, obsfloor, obsceil = _load_obs_profile(obsfile, limit_below_ceil=True)
    if filter_obs_fxn is not None:
        obsz, obsprof = filter_obs_fxn(obsz, obsprof, vmralts, vmrprof, obsfile)
    
    zz_vmr = vmralts <= obsceil
    if np.any(vmralts[zz_vmr] > obsz.max()):
        print('Warning: {} does not have values all the way up to the obs. ceiling. Lowering the effective ceiling'.format(obsfile))
        obsceil = obsz.max()
        zz_vmr = vmralts <= obsceil

    # We want to do constant-value extrapolation down, so we don't specify a "left" value. But we don't always want
    # to do the same up, it depends on the options given, so we set those to NaN.
    interp_prof = np.interp(vmralts[zz_vmr], obsz, obsprof, right=np.nan)

    if np.any(np.isnan(interp_prof)):
        raise RuntimeError('Not all levels got a value')

    combined_prof = vmrprof.copy()
    combined_prof[vmralts <= obsceil] = interp_prof
    adj_flag = 0

    if adjust_to_overworld and obsceil > min_req_top_alt:
        combined_prof = _adjust_prof_to_overworld(prof_alts=vmralts, prof=combined_prof, prof_theta=vmr_theta,
                                                  tropopause_alt=vmr_trop_alt, obs_ceiling=obsceil)
        adj_flag = 1

    if force_prior_fxn is not None:
        combined_prof = force_prior_fxn(vmralts, vmrprof, combined_prof, obsfile)

    return combined_prof, obsceil, adj_flag


def bin_obs_to_vmr_alts(vmralts, obsz, obsprof, obsceil=None, full_size=True, fill_surface=False, filter_nans=True):
    """Bin observational data to prior profile altitudes using a linear weighting

    Parameters
    ----------
    vmralts : array-like
        The altitudes of the prior profile levels

    obsz : array-like
        The altitudes of the observations. Must be in the same units as vmralts

    obsprof : array-like
        The observed concentrations

    obsceil : float, optional
        If given, the top altitude of the observations. If not given, inferred from `obsz`

    Returns
    -------
    array-like
        The binned observed concentrations

    """
    if filter_nans:
        notnans = ~np.isnan(obsz) & ~np.isnan(obsprof)
        obsz = obsz[notnans]
        obsprof = obsprof[notnans]

    if obsceil is None:
        obsceil = np.max(obsz)
    zz_vmr = vmralts <= obsceil
    if full_size:
        binned_prof = np.full_like(vmralts, np.nan)
    else:
        binned_prof = np.full([zz_vmr.sum()], np.nan)

    for i in np.flatnonzero(zz_vmr):
        # What we want are weights that are 1 at the VMR level i and decrease linearly to 0 at levels i-1 and i+1. This
        # way the weighted sum of observed concentrations for level i is weighted most toward the nearest altitude
        # observations but account for the concentration between levels. This is similar to how the effective vertical
        # path is handled in GGG.
        weights = np.zeros_like(obsz)
        if i > 0:
            # The bottom level will not get these weights because it should be at zero altitude and has no level below
            # it.
            zdiff = vmralts[i] - vmralts[i - 1]
            in_layer = (obsz >= vmralts[i - 1]) & (obsz < vmralts[i])
        else:
            # There's a couple different ways we could handle the case where there is aircraft data below the bottom
            # layer. We could just ignore it, which might make sense if the profiles we defined on layer edges. For
            # TCCON, when we do that, typically the bottom edge is at 0, so there will be no values there. We could just
            # give all points below the bottom layer a relative weight of 1, but that's inconsistent with the idea that
            # points near the prior level should contribute the most. What I've chosen to do is to assume the altitude
            # is in the center of the layer, so it should extend as far below as it does above, as use that to weight
            # the aircraft data.
            zdiff = vmralts[i + 1] - vmralts[i]
            bottom_alt = vmralts[i] - zdiff
            in_layer = (obsz >= bottom_alt) & (obsz < vmralts[i])
        weights[in_layer] = (obsz[in_layer] - vmralts[i - 1]) / zdiff

        # The top bin can be handled by this code whether or not adjust_to_overworld is set, but the result will be
        # different:
        #   * if adjust_to_overworld is False, then there will be "observations" extending at least to the next bin
        #     above the obs. ceiling. Some of these will be from the prior, but the will be spaced out so that they are
        #     on the same z-grid as the obs. The weighted average will then include some amount of the prior, depending
        #     on how much of the inter-level space needed prior data. This ensures a smooth transition to the prior.
        #   * if adjust_to_overworld is True, then the will be no observations above the ceiling. The last bin below the
        #     ceiling will potentially have fewer points than the others, but it will represent only observations and
        #     so can be extended to the tropopause.
        zdiff = vmralts[i + 1] - vmralts[i]
        in_layer = (obsz >= vmralts[i]) & (obsz < vmralts[i + 1])
        weights[in_layer] = (vmralts[i + 1] - obsz[in_layer]) / zdiff

        # normalize the weights. if no values in this bin, this will result in weights all being NaN which is what
        # we want. The following sum should NOT be a nansum because we want a NaN in that bin so it can get filled
        # in later. If NaNs in the obs/prior profiles are problems, that needs to be dealt with before binning.
        weights /= weights.sum()
        binned_prof[i] = np.sum(weights * obsprof)

    if fill_surface:
        # If there are any NaNs left at the beginning (bottom) of the profile, replace them with the first non-NaN
        # value. This assumes that the .atm file already includes a surface measurement at the bottom and we just need
        # to extrapolate to below surface layers.
        nn = np.flatnonzero(~np.isnan(binned_prof))[0]
        binned_prof[:nn] = binned_prof[nn]

    if not full_size and np.any(np.isnan(binned_prof)):
        raise RuntimeError('Not all levels got a value')

    return binned_prof


def weighted_bin_obs_to_vmr_alts(obsfile, vmralts, vmrprof, force_prior_fxn=None, filter_obs_fxn=None,
                                 adjust_to_overworld=False, min_req_top_alt=_default_min_req_top_alt,
                                 vmr_trop_alt=None, vmr_theta=None):
    """
    Stitch together the observed and prior profiles with altitude-weighted binning

    For each altitude of the .vmr priors below the observation ceiling, weights are computed as:

    ..math::

        w(z) = \frac{z -z_{i-1}}{z_i - z_{i-1}} \text{ for } z \in [z_{i-1}, z_i)

        w(z) = \frac{z_{i+1} - z}{z_{i+1} - z_i} \text{ for } z \in [z_i, z_{i+1})

        w(z) = 0 \text{ otherwise }

    and normalized to 1. The observed concentration at :math:`z_i` is then :math:`w^T \cdot c` where :math:`c` is the
    observed concentration vector.

    :param obsfile: the path to the .atm file to use
    :type obsfile: str

    :param vmralts: the vector of altitude levels in the .vmr priors, in kilometers
    :type vmralts: array-like

    :param vmrprof: the vector of concentrations in the .vmr priors, in mole fraction
    :type vmrprof: array-like

    :param force_prior_fxn: a function that takes four inputs (altitudes of the .vmr prior levels, the prior profile in
     DMF, the combined obs+prior profile, and the .atm file name) and returns a modified combined profile with some
     levels replaced with the prior.
    :type force_prior_fxn: callable

    :param filter_obs_fxn: a function that takes five inputs (observation altitudes, observation concentrations, the
     prior altitudes, the prior concentrations, and the .atm file name) and returns modified obs. altitudes and
     concentrations. This is to provide a way to filter/replace bad data in the raw observations. It is called
     immediately after the obs. data is loaded, before any other action is taken.
    :type filter_obs_fxn: callable

    :param adjust_to_overworld: if True, the profile will be extended to the overworld as follows: if the obs. ceiling
     is above ``min_req_prof_alt``, then the top obs. bin will be extended to the tropopause. The concentration will
     then do a linear interpolation in theta between the tropopause and 380 K. If the top obs. bin is above 380 K, then
     nothing is changed. If the top obs. bin is below ``min_req_prof_alt``, then the prior is inserted above that level,
     no special treatment is used.
    :type adjust_to_overworld: bool

    :param min_req_top_alt: the minimum altitude that the top alt. bin of the observations must be above for the
     special logic triggered by ``adjust_to_overworld`` to be used.
    :type min_req_top_alt: float

    :param vmr_trop_alt: the altitude (in the same units as ``vmralts``) of the tropopause as read from the .vmr file.
     Not needed if ``adjust_to_overworld == False``.
    :type vmr_trop_alt: float

    :param vmr_theta: the profile of potential temperature (in K) on the vertical levels defined by ``vmralts``.
     Not needed if ``adjust_to_overworld == False``.
    :type vmr_theta: array-like

    :return: the combined observation + prior profile on the .vmr levels, and the observation ceiling (in kilometers)
    """
    # if using the adjust_to_overworld logic, then we only want the real obs. data, not any of the old prior that was
    # appended to the top of the profile before
    obsz, obsprof, obsfloor, obsceil = _load_obs_profile(obsfile, limit_below_ceil=adjust_to_overworld)
    if filter_obs_fxn is not None:
        obsz, obsprof = filter_obs_fxn(obsz, obsprof, vmralts, vmrprof, obsfile)

    zz_vmr = vmralts <= obsceil
    if np.any(vmralts[zz_vmr] > obsz.max()):
        print('Warning: {} does not have values all the way up to the obs. ceiling. Lowering the effective ceiling'.format(obsfile))
        obsceil = obsz.max()
        zz_vmr = vmralts <= obsceil

    if not adjust_to_overworld:
        # adjust_to_overworld means that we extend the top obs. bin to the met. tropopause then linearly interpolating
        # in theta-space between the trop. and 380 K. This gets messed up if we allow the top obs. bin to include some
        # of the prior, because that will change the value that gets extrapolated. On the other hand, if we're appending
        # the new priors right to the top of the obs. profile, then it does make sense to fill out the top bin with
        # prior data to make a smooth transition.
        obsz, obsprof = _blend_top_weighted_bin(obsz=obsz, obsprof=obsprof, obsceil=obsceil,
                                                vmralts=vmralts, vmrprof=vmrprof, zz_vmr=zz_vmr)

    binned_prof = bin_obs_to_vmr_alts(vmralts, obsz, obsprof, obsceil=obsceil, full_size=False, fill_surface=True)

    combined_prof = vmrprof.copy()
    combined_prof[zz_vmr] = binned_prof

    adj_flag = 0
    if adjust_to_overworld and obsceil > min_req_top_alt:
        combined_prof = _adjust_prof_to_overworld(prof_alts=vmralts, prof=combined_prof, prof_theta=vmr_theta,
                                                  tropopause_alt=vmr_trop_alt, obs_ceiling=obsceil)
        adj_flag = 1

    if force_prior_fxn is not None:
        combined_prof = force_prior_fxn(vmralts, vmrprof, combined_prof, obsfile)

    return combined_prof, obsceil, adj_flag


def _blend_top_weighted_bin(obsz, obsceil, obsprof, vmralts, vmrprof, zz_vmr):
    zz_obs = obsz <= obsceil

    # check that the obs. profiles go past the next .vmr level above the ceiling - this is to allow us to properly
    # handle the last .vmr level below the ceiling. If the obs. don't go that high, it's probably because we got
    # obs files that didn't have an old stratosphere appended to the top. In that case, what we'll do is add a bunch of
    # altitude levels that we can then put the prior values into. We'll use the average level spacing of the last 10
    # levels to decide how fine a grid to make
    i_ceil = np.flatnonzero(zz_vmr)[-1]
    if np.all(obsz < vmralts[i_ceil + 1]):
        last_obsz = np.nanmax(obsz)
        target_alt = vmralts[i_ceil + 2]
        obs_spacing = np.diff(obsz)
        if np.any(obs_spacing < 0):
            raise NotImplementedError('Obs. data is not monotonically ascending')
        # If the grid is not monotonically ascending, this will not work, however the _load_obs_profile function ensures
        # that it is. Make the minimum spacing 1 meter - this will avoid issues of overly small or 0 grid spacing caused
        # by truncation of the altitudes in the .atm files (i.e. sometimes 4000.1 and 4000.4 get truncated to 4000)
        obs_spacing = np.nanmax([0.001, np.abs(np.nanmean(obs_spacing[-10:]))])
        extra_obsz = np.arange(last_obsz, target_alt, obs_spacing)
        # sort_inds = np.argsort(obsz)  #  can use this if there's a case where the profile is read in upside down
        obsz = np.concatenate([obsz, extra_obsz])
        obsprof = np.concatenate([obsprof, np.full_like(extra_obsz, np.nan)])

        # Expand the vector indicating which levels are observation data to be the same size as the expanded profile.
        zz_obs_expanded = np.zeros(obsz.shape, dtype=np.bool_)
        zz_obs_expanded[np.nonzero(zz_obs)] = True  # zz_obs still wrong size, convert to indices for this
        zz_obs = zz_obs_expanded

    # Replace observations above the ceiling with the .vmr profiles, we'll use this to handle the last level below
    # the ceiling.
    obsprof[~zz_obs] = np.interp(obsz[~zz_obs], vmralts, vmrprof)
    return obsz, obsprof


def _adjust_prof_to_overworld(prof_alts, prof, prof_theta, tropopause_alt, obs_ceiling):
    """
    Modify the binned profile to have a constant value to the tropopause, then linearly interpolate in theta to 380 K.

    Note that if the observation ceiling is above 380 K the profile is not modified.

    :param prof_alts: the vector of altitudes that the profile is defined on.
    :type prof_alts: array-like

    :param prof: the vector of concentrations that make up the profile
    :type prof: array-like

    :param prof_theta: the vector of potential temperatures (in K) on the same levels as the profile
    :type prof_theta: array-like

    :param tropopause_alt: the altitude of the tropopause
    :type tropopause_alt: float

    :param obs_ceiling: the ceiling of the observations as reported in the .atm file
    :type obs_ceiling: float

    :return: the modified profile; it is also modified in-place
    :rtype: array-like
    """
    i_obs_top = np.flatnonzero(prof_alts <= obs_ceiling)[-1]
    i_trop = np.flatnonzero(prof_alts > tropopause_alt)[0]
    i_overworld = np.flatnonzero(prof_theta >= 380)[0]

    if i_obs_top >= i_overworld:
        return prof

    obs_top_conc = prof[i_obs_top]

    trop_theta = np.interp(tropopause_alt, prof_alts, prof_theta)
    overworld_theta = prof_theta[i_overworld]
    overworld_conc = prof[i_overworld]

    prof[i_obs_top:i_trop] = obs_top_conc
    # only want to interpolate in theta above the top of the observations. If the observations end in the middleworld,
    # the prof_alts > tropopause_alt is not enough. Also need to limit to above the observations. This occurred for
    # the Lamont aircore on 2012-01-14 at 20:54Z
    xx_middleworld = (prof_alts > tropopause_alt) & (prof_alts > prof_alts[i_obs_top]) & (prof_theta < 380)
    prof[xx_middleworld] = np.interp(prof_theta[xx_middleworld], [trop_theta, overworld_theta], [obs_top_conc, overworld_conc])

    return prof


def _get_atm_gas(atm_file, force_lower=False):
    # Assumes that the gas name will be at the end of the file name, like ..._CH4.atm or ..._CO.atm.
    gas_and_ext = atm_file.split('_')[-1]
    gas = gas_and_ext.replace('.atm','')
    if force_lower:
        gas = gas.lower()
    return gas


def _organize_atm_files_by_species(atm_files, force_lower=False):
    return {_get_atm_gas(f, force_lower): f for f in atm_files}


########################################
# FORCE PRIOR AND OBS FILTER FUNCTIONS #
########################################

def filter_obs_ggg2019(obsz, obsprof, prior_alts, prior, obsfile):
    _, obs_header = butils.read_atm_file(obsfile)

    info_key = _find_key(obs_header, 'info$')
    aircraft = obs_header[info_key].lower()
    gas = _get_atm_gas(obsfile).lower()

    if 'radiosonde' in aircraft and gas == 'h2o':
        # Paul says that the radiosonde water is not believable above 10-12 km and should be replaced with the prior if
        # less than 0.0003%.
        xx = obsprof < 3e-6
        obsprof[xx] = np.interp(obsz[xx], prior_alts, prior, left=np.nan, right=np.nan)
    elif 'aircore' in aircraft and gas == 'co':
        # I notice there's some aircores where the CO goes negative (e.g. one at 2014-09-17 20:10:57 at 97.55 W, 36.72 N)
        # If CO gets below 0.1 ppb, there's got to be something wrong with the aircore measurement.
        xx = obsprof >= 1e-10
        obsz = obsz[xx]
        obsprof = obsprof[xx]

    if np.any(np.isnan(obsprof)):
        raise NotImplementedError('Needed to replace values outside the altitude range of the prior')
    return obsz, obsprof


def force_prior_ggg2019(alts, prior, obs_prof_vmr_levels, obsfile):
    _, obs_header = butils.read_atm_file(obsfile)

    info_key = _find_key(obs_header, 'info$')
    aircraft = obs_header[info_key].lower()
    gas = _get_atm_gas(obsfile).lower()

    if 'aircore' in aircraft:
        # Paul trusts the priors more than aircore above 20-22 km.
        xx = alts >= 21
    elif 'radiosonde' in aircraft and gas == 'h2o':
        # Paul says that the radiosonde water is not believable above 10-12 km and should be replaced with the prior if
        # less than 0.0003%.
        xx = alts >= 11
    else:
        # if not one of the cases, above, do not replace any levels.
        xx = np.zeros(alts.shape, dtype=np.bool_)

    obs_prof_vmr_levels[xx] = prior[xx]
    return obs_prof_vmr_levels
