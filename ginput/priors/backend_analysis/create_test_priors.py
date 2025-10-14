from __future__ import print_function
import argparse
from datetime import datetime as dtime, timedelta as tdel
from glob import glob
import h5py
from multiprocessing import Pool
from pandas import date_range

import numpy as np
import os
import re
import shutil
import sys

from . import backend_utils as bu
from .. import tccon_priors
from ...mod_maker import mod_maker
from ...common_utils import mod_utils, readers, ioutils
from ...download import get_GEOS5

float_h5_fill = np.nan
int_h5_fill = -999999
string_h5_fill = b'N/A'

def _to_bool(val):
    if isinstance(val, str):
        if val.lower() == 'true':
            return True
        elif val.lower() == 'false':
            return False
        else:
            raise ValueError('Bad string for boolean: {}'.format(val))
    else:
        return bool(val)

# These should match the arg names in driver()
_req_info_keys = ('gas_name', 'site_file', 'geos_top_dir', 'geos_chm_top_dir', 'mod_top_dir', 'prior_save_file')
_req_info_ispath = ('site_file', 'geos_top_dir', 'mod_top_dir', 'prior_save_file')
_req_info_help = {'gas_name': 'The name of the gas to generate priors for.',
                  'site_file': 'A CSV file containing the header DATES,LATS,LONS,ATMFILES and the date, latitude, '
                               'longitude, and corresponding .atm file of each desired prior (one per line)',
                  'geos_top_dir': 'The directory containing the GEOS FP-IT data in subdirectories Np, Nx, and Nv. '
                                  'This is where it will be downloaded to, if --download is one of the actions.',
                  'geos_chm_top_dir': 'The directory containing the GEOS FP-IT chemistry data in an Nv subdirectory. ',
                  'mod_top_dir': 'The top directory to save .mod files to or read them from. Must contain '
                                 'subdirectories "fpit/xx/vertical" to read from, these will be automatically created '
                                 'if writing .mod files.',
                  'prior_save_file': 'The filename to give the HDF5 file the priors will be saved in.'}

_opt_info_keys = ('integral_file', 'base_vmr_file', 'trop_eqlat')
_opt_info_ispath = ('integral_file', 'base_vmr_file')
_opt_info_help = {'integral_file': 'A path to an integral.gnd file that specifies the altitude grid for the .vmr '
                                   'files. If not present, the .vmr files will be on the native GEOS grid.',
                  'base_vmr_file': 'A path to a summer 35N .vmr file that can be used for the secondary gases. '
                                   'If not given, the .vmr files will only include the primary gases.',
                  'trop_eqlat': 'Whether to use tropospheric effective latitude or not (a boolean).'}

_default_values = {'trop_eqlat': True}
_key_converters = {'trop_eqlat': _to_bool}

_default_file_types = ('2dmet', '3dmet')


def unfmt_lon(lonstr):
    mod_utils.format_lon(lonstr)


def _date_range_str_to_dates(drange_str):
    """
    Convert a date range string to two datetime objects
    :param drange_str: the string to convert, in the format YYYYMMDD-YYYYMMDD
    :type drange_str: str
    
    :return: start and end dates
    :rtype: :class:`datetime.datetime`, :class:`datetime.datetime`  
    """
    start_dstr, end_dstr = drange_str.split('-')
    start_date = dtime.strptime(start_dstr, '%Y%m%d')
    end_date = dtime.strptime(end_dstr, '%Y%m%d')
    return start_date, end_date


def make_lat_lon_list_for_atms(atm_files, list_file):
    """
    Create the list of locations to make priors for with the driver function

    :param atm_files: sequence of paths to .atm files to read lats/lons/dates from
    :param list_file: path to the file to write the lats/lons/dates to
    :return: none, writes file
    """
    with open(list_file, 'w') as wobj:
        wobj.write('DATE,LAT,LON,ATMFILE,TCCON\n')
        for f in atm_files:
            data, header = bu.read_atm_file(f)
            datestr = header['aircraft_floor_time_UTC'].strftime('%Y-%m-%d')
            lon = header['TCCON_site_longitude_E']
            lat = header['TCCON_site_latitude_N']
            site = header['TCCON_site_name']
            wobj.write('{date},{lat},{lon},{file},{site}\n'.format(date=datestr, lon=lon, lat=lat,
                                                                   file=os.path.basename(f), site=site))


def read_info_file(info_filename):
    """
    Read the info/config file for this set of profiles to generate.

    The file will have the format ``key = value``, one per line. Comments must be on lines by themselves, cannot start
    mid-line.

    :param info_filename: path to the info/config file
    :type info_filename: str

    :return: a dictionary with keys _reg_info_keys and _opt_info_keys containing information from the info file. Paths
     are all converted to absolute paths.
    :rtype: dict
    """
    # Setup the dictionary that will receive the data. Default everything to None; will check that required keys were
    # overwritten at the end.
    info_dict = {k: None for k in _req_info_keys + _opt_info_keys}
    info_dict.update(_default_values)
    info_file_dir = os.path.abspath(os.path.dirname(info_filename))

    with open(info_filename, 'r') as fobj:
        for line_num, line in enumerate(fobj):
            # Skip comment lines
            if re.match(r'\s*#', line):
                continue

            # Each line must have the format key = value
            key, value = [el.strip() for el in line.split('=')]
            if key not in _req_info_keys + _opt_info_keys:
                # Skip unexpected keys
                print('Ignoring line {} of {}: key "{}" not one of the required keys'.format(line_num, info_filename, key))
                continue

            elif key in _req_info_ispath + _opt_info_ispath:
                # Make any relative paths relative to the location of the info file.
                value = value if os.path.isabs(value) else os.path.join(info_file_dir, value)

            if key in _key_converters:
                value = _key_converters[key](value)
                
            info_dict[key] = value

    # Check that all required keys were read in. Any optional keys not read in will be left as None.
    for key in _req_info_keys:
        if info_dict[key] is None:
            raise RuntimeError('Key "{}" was missing from the input file {}'.format(key, info_filename))

    return info_dict


def read_date_lat_lon_file(acinfo_filename, date_fmt='str'):
    """
    Read the .csv file containing the date, latitudes, and longitudes to generate priors for
    
    :param acinfo_filename: the path to the file to read
    :type acinfo_filename: str
     
    :param date_fmt: how to format the dates read. "str" will return string date ranges (the date in the file to one
     day later), while "datetime" will just return the date in the file as a datetime object.
    :type date_fmt: str
     
    :return: the longitudes, latitudes, dates, and .atm files as lists. 
    """
    with open(acinfo_filename, 'r') as acfile:
        # Check that the header matches what is expected
        header_parts = acfile.readline().strip().split(',')
        expected_header_parts = ('DATE', 'LAT', 'LON', 'ATMFILE')
        if any(a != b for a, b in zip(header_parts, expected_header_parts)):
            raise IOError('The first {ncol} columns in the info file ({infofile}) do not match what is expected: '
                          '{expected}'.format(ncol=len(expected_header_parts), infofile=acinfo_filename,
                                              expected=', '.join(expected_header_parts)))
        acdates = []
        aclats = []
        aclons = []
        atmfiles = []
        for line in acfile:
            if line.startswith('#'):
                continue
            elif '#' in line:
                line = line.split('#')[0].strip()
            line_parts = line.split(',')
            date_str = line_parts[0]
            date1 = dtime.strptime(date_str, '%Y-%m-%d')
            if date_fmt == 'str':
                date2 = date1 + tdel(days=1)
                acdates.append(date1.strftime('%Y%m%d') + '-' + date2.strftime('%Y%m%d'))
            elif date_fmt == 'datetime':
                acdates.append(date1)
            else:
                raise ValueError('date_fmt must be either "str" or "datetime"')

            aclats.append(float(line_parts[1]))
            aclons.append(float(line_parts[2]))
            atmfiles.append(line_parts[3])

    return aclons, aclats, acdates, atmfiles


def make_full_mod_dir(top_dir, product):
    """
    Provide the full path to the directory containing the .mod file
    
    :param top_dir: the directory that mod_maker was told to save the .mod files in
    :type top_dir: str
     
    :param product: which GEOS product ("fp" or "fpit") was created.
    :type product: str
     
    :return: the path to the directory containing the .mod files
    :rtype: str 
    """
    return os.path.join(top_dir, product.lower(), 'xx', 'vertical')


def check_geos_files(acdates, download_to_dir, chem_download_dir=None, file_type=get_GEOS5._default_file_type,
                     levels=get_GEOS5._default_level_type):
    """
    Check that all the requires GEOS files are present.
    
    :param acdates: the list of dates required, as string date ranges
    :type acdates: list(str)
     
    :param download_to_dir: directory that the GEOS data was downloaded to. This should have "Nv", "Nx", and/or "Np" 
     sub-directories.
    :type download_to_dir: str
     
    :param chem_download_dir: where the chemistry files were downloaded to. Not needed if they were downloaded to the 
     same location as the met files.
    :type chem_download_dir: str or None
     
    :param file_type: which file type ("met" or "chm") to check for. May also be a sequence of types. If "chm", then 
     ``chem_download_dir`` must be given *if* the chemistry files are not in ``download_to_dir``.
    :type file_type: str or list(str)
    
    :param levels: which file levels ("surf", "p" or "eta") to check for, or a sequence of levels. If a sequence, then
     both this and ``file_type`` must have the same number of elements.
    :type levels: str or list(str)
     
    :return: none, prints to screen the missing files 
    """
    acdates = [dtime.strptime(d.split('-')[0], '%Y%m%d') for d in acdates]
    file_type, levels = get_GEOS5.check_types_levels(file_type, levels)

    missing_files = dict()
    for ftype, ltype in zip(file_type, levels):
        target_dir = chem_download_dir if ftype == 'chm' and chem_download_dir is not None else download_to_dir
        file_names, file_dates = mod_utils.geosfp_file_names_by_day('fpit', ftype, ltype, utc_dates=acdates,
                                                                    add_subdir=True)
        for f, d in zip(file_names, file_dates):
            d = d.date()
            ffull = os.path.join(target_dir, f)
            if not os.path.isfile(ffull):
                if d in missing_files:
                    missing_files[d].append(f)
                else:
                    missing_files[d] = [f]

    for d in sorted(missing_files.keys()):
        nmissing = len(missing_files[d])
        missingf = set(missing_files[d])
        print('{date}: {n} ({files})'.format(date=d.strftime('%Y-%m-%d'), n=min(8, nmissing), files=', '.join(missingf)))

    print('{} of {} dates missing at least one file'.format(len(missing_files), len(acdates)))


def download_geos(acdates, download_to_dir, chem_download_dir=None,
                  file_type=get_GEOS5._default_file_type, levels=get_GEOS5._default_level_type):
    """
    Download GEOS files needed for a set of priors
    
    :param acdates: the list of dates required, as string date ranges
    :type acdates: list(str)
     
    :param download_to_dir: directory to download the GEOS data to. Subdirectories Nv, Np, or Nx will be created as
     necessary.
    :type download_to_dir: str
     
    :param chem_download_dir: where the chemistry files should be downloaded to. Not needed if they are to be downloaded
     to the same location as the met files.
    :type chem_download_dir: str or None
     
    :param file_type: which file type ("met" or "chm") to download. May also be a sequence of types. If "chm", then 
     ``chem_download_dir`` must be given *if* the chemistry files are not in ``download_to_dir``.
    :type file_type: str or list(str)
    
    :param levels: which file levels ("surf", "p" or "eta") to download, or a sequence of levels. If a sequence, then
     both this and ``file_type`` must have the same number of elements.
    :type levels: str or list(str):param acdates: 

    :return: none, downloads files.
    """
    file_type, levels = get_GEOS5.check_types_levels(file_type, levels)
    for ftype, ltype in zip(file_type, levels):
        dl_path = chem_download_dir if ftype == 'chm' and chem_download_dir is not None else download_to_dir
        for dates in set(acdates):
            date_range = _date_range_str_to_dates(dates)
            get_GEOS5.driver(date_range, mode='FPIT', path=dl_path, filetypes=ftype, levels=ltype)


def _make_mod_atm_map(acdates, aclons, aclats, acfiles):
    """
    Make a dictionary mapping .mod files to the .atm files they were created for.
    
    :param acdates: list of string date ranges the .mod files were created for.
    :type acdates: list(str)
     
    :param aclons: list of longitudes the .mod files were created for 
    :type aclons: list(float)
     
    :param aclats: list of latitudes the .mod files were created for
    :type aclats: list(float)
     
    :param acfiles: list of .atm files corresponding to the dates, lats, and lons in the first three arguments.
    :type acfiles: list(str)
     
    :return: a dictionary with .mod file names as keys and .atm files as values. Both will just be the basenames, not
     full paths.
    :rtype: dict
    """
    atm_mod_map = dict()

    for dates, lon, lat, atmfile in zip(acdates, aclons, aclats, acfiles):
        start_date, end_date = _date_range_str_to_dates(dates)
        mod_files = _list_mod_files_required(start_date, end_date, lon, lat)

        for modf in mod_files:
            if modf in atm_mod_map:
                atm_mod_map[modf].append(atmfile)
            else:
                atm_mod_map[modf] = [atmfile]

    return atm_mod_map


def _list_mod_files_required(start_date, end_date, lon, lat):
    """
    List the .mod files required for a date/lat/lon
    
    :param start_date: the start of the time period requested
    :type start_date: :class:`datetime.datetime`
     
    :param end_date: the end of the time period requested (exclusive)
    :type end_date: :class:`datetime.datetime`
     
    :param lon: the longitude requested
    :type lon: float
     
    :param lat: the latitude requested
    :type lat: float
     
    :return: the list of .mod files required (basenames only)
    :rtype: list(str) 
    """
    mod_files = []

    for date in date_range(start_date, end_date, freq='3h', inclusive='left'):
        modf = mod_utils.mod_file_name_for_priors(datetime=date, site_lat=lat, site_lon_180=lon, round_latlon=False, in_utc=True)
        mod_files.append(modf)

    return mod_files


def make_mod_files(acdates, aclons, aclats, geos_dir, out_dir, chem_dir=None, include_chm=True, nprocs=0,
                   geos_mode='fpit-eta'):
    """
    Make the .mod files required for a set of test priors
    
    :param acdates: list of string date ranges the .mod files were created for.
    :type acdates: list(str)
     
    :param aclons: list of longitudes the .mod files were created for 
    :type aclons: list(float)
     
    :param aclats: list of latitudes the .mod files were created for
    :type aclats: list(float)
    
    :param geos_dir: directory that the GEOS data was downloaded to. This should have "Nv", "Nx", and/or "Np" 
     sub-directories.
    :type geos_dir: str
     
    :param out_dir: directory to save the .mod files to. The :file:`<product>/xx/vertical` subdirectory tree will be 
     automatically created.
    :type out_dir: str
      
    :param chem_dir: where the chemistry files were downloaded to. Not needed if they were downloaded to the 
     same location as the met files.
    :type chem_dir: str
     
    :param include_chm: whether to include the chemistry variables (CO currently) in the .mod files. If ``True``, then
     the chemistry files must be supplied.
    :type include_chm: bool
      
    :param nprocs: number of processors to use to generate the .mod files
    :type nprocs: int
     
    :param geos_mode: mode argument to :func:`ginput.mod_maker.mod_maker.driver, usually "fpit" or "fpit-eta".
    :type geos_mode: str
     
    :return: none, writes .mod files 
    """
    if chem_dir is None:
        chem_dir = geos_dir
    print('Will save to', out_dir)
    mod_dir = make_full_mod_dir(out_dir, 'fpit')
    print('  (Listing GEOS files...)')
    geos_files = sorted(glob(os.path.join(geos_dir, 'Nv', 'GEOS*.nc4')))
    geos_dates = set([dtime.strptime(re.search(r'\d{8}', f).group(), '%Y%m%d') for f in geos_files])
    geos_chm_files = sorted(glob(os.path.join(chem_dir, 'Nv', 'GEOS*.nc4')))
    geos_chm_dates = set([dtime.strptime(re.search(r'\d{8}', f).group(), '%Y%m%d') for f in geos_chm_files])

    mm_args = dict()

    print('  (Making list of .mod files to generate...)')
    for (dates, lon, lat) in zip(acdates, aclons, aclats):
        # First, check whether this .mod file already exists. If so, we can skip it.
        start_date, end_date = [dtime.strptime(d, '%Y%m%d') for d in dates.split('-')]
        if start_date not in geos_dates or start_date not in geos_chm_dates:
            print('Cannot run {}, missing either met or chem GEOS data'.format(start_date))
            continue
        req_mod_files = _list_mod_files_required(start_date, end_date, lon, lat)
        files_complete = [os.path.exists(os.path.join(mod_dir, f)) for f in req_mod_files]

        if all(files_complete) and len(files_complete) == 8:
            print('All files for {} at {}/{} complete, skipping'.format(dates, lon, lat))
            continue
        else:
            print('One or more files for {} at {}/{} needs generated'.format(dates, lon, lat))

        # If we're here, this combination of date/lat/lon needs generated. But we can be more efficient if we do all
        # locations for one date in one go because then we only have to make the eq. lat. interpolators once, so we
        # create one set of args per day and take advantage of the mod maker driver's ability to loop over lat/lons.
        key = (start_date, end_date)
        if key in mm_args:
            mm_args[key]['mm_lons'].append(lon)
            mm_args[key]['mm_lats'].append(lat)
        else:
            # The keys here must match the argument names of mm_helper_internal as the dict will be ** expanded.
            mm_args[key] = {'mm_lons': [lon], 'mm_lats': [lat], 'geos_dir': geos_dir, 'chem_dir': chem_dir,
                            'with_chm': include_chm, 'out_dir': out_dir, 'nprocs': nprocs, 'date_range': key,
                            'mode': geos_mode}

    if nprocs == 0:
        print('Making .mod files in serial mode')
        for kwargs in mm_args.values():
            mm_helper(kwargs)
    else:
        # Convert this so that each value is a list with one element: the args dict. This way, starmap will expand
        # the list into a single argument for mm_helper, which then expands the dict into a set of keyword arguments
        # for mm_helper_internal
        mm_args = {k: [v] for k, v in mm_args.items()}
        print('Making .mod file in parallel mode with {} processors'.format(nprocs))
        with Pool(processes=nprocs) as pool:
            pool.starmap(mm_helper, mm_args.values())


def mm_helper(kwargs):
    """
    Helper function used to run mod_maker consistently whether running in serial or parallel mode.
    
    :param kwargs: keyword arguments: ``date_range`` - the start and end dates, as a list, for mod_maker. ``mm_lons`` 
     and ``mm_lats`` - the lists of longitudes and latitudes to make. ``geos_dir`` and ``chem_dir`` - the directories
     for the met and chem data. ``nprocs`` - number of processors used (determines whether the prints to the screen are
     muted or not). ``mode`` - the mod_maker mode (usually "fpit-eta"). ``with_chm`` - whether to include chemistry
     variables.
     
    :return: none. 
    """
    def mm_helper_internal(date_range, mm_lons, mm_lats, geos_dir, chem_dir, out_dir, nprocs, mode, with_chm):
        date_fmt = '%Y-%m-%d'
        # Duplicate
        print('Generating .mod files {} to {}'.format(date_range[0].strftime(date_fmt), date_range[1].strftime(date_fmt)))
        mod_maker.driver(date_range=date_range, met_path=geos_dir, chem_path=chem_dir, save_path=out_dir,
                         include_chm=with_chm, mode=mode, keep_latlon_prec=True, save_in_utc=True,
                         lon=mm_lons, lat=mm_lats, alt=0.0, muted=nprocs > 0)

    mm_helper_internal(**kwargs)


def make_priors(prior_save_file, mod_dir, gas_name, acdates, aclons, aclats, acfiles, trop_eqlat=True, zgrid_file=None, nprocs=0):
    """
    Make the priors, saving them to an .h5 file
    
    :param prior_save_file: the path to the .h5 file
    :type prior_save_file: str
     
    :param mod_dir: the directory containing the .mod files. (That is the directory that actually has the .mod files in
     it, not the "top" directory with fpit/xx/vertical subdirectories.)
    :type mod_dir: str
     
    :param gas_name: which gas to create priors for. Must be a key of ``tccon_priors.gas_records``.
    :type gas_name: str
    
    :param acdates: list of string date ranges the .mod files were created for.
    :type acdates: list(str)
     
    :param aclons: list of longitudes the .mod files were created for 
    :type aclons: list(float)
     
    :param aclats: list of latitudes the .mod files were created for
    :type aclats: list(float)

    :param acfiles: list of .atm files that correspond to the above dates, lats, and lons.
    :type acfiles: list(str)

    :param zgrid_file: levels file from GGG that specifies the levels to write the priors on (optional). If omitted, the
     priors will be written on the standard GEOS grid.
    :type zgrid_file: str or None

    :param nprocs: number of processors to use to run the priors.
    :type nprocs: int

    :return: none, writes HDF5 file.
    """
    print('Will save to', prior_save_file)
    # Find all the .mod files, get unique date/lat/lon (should be 8 files per)
    # and make an output directory for that
    mod_files = glob(os.path.join(mod_dir, '*.mod'))
    grouped_mod_files = dict()
    acdate_strings = acdates
    acdates = [dtime.strptime(d.split('-')[0], '%Y%m%d').date() for d in acdates]
    aclons = np.array(aclons)
    aclats = np.array(aclats)

    for f in mod_files:
        fbase = os.path.basename(f)
        lonstr = mod_utils.find_lon_substring(fbase)
        latstr = mod_utils.find_lat_substring(fbase)
        datestr = mod_utils.find_datetime_substring(fbase)

        utc_datetime = dtime.strptime(datestr, '%Y%m%d%H')
        utc_date = utc_datetime.date()
        utc_datestr = utc_datetime.date().strftime('%Y%m%d')
        lon = mod_utils.format_lon(lonstr)
        lat = mod_utils.format_lat(latstr)

        # If its one of the profiles in the info file, make it
        if utc_date in acdates and np.any(np.abs(aclons - lon) < 0.02) and np.any(np.abs(aclats - lat) < 0.02):
            print(f, 'matches one of the listed profiles!')
            keystr = '{}_{}_{}'.format(utc_datestr, lonstr, latstr)
            if keystr in grouped_mod_files:
                grouped_mod_files[keystr].append(f)
            else:
                grouped_mod_files[keystr] = [f]
        else:
            print(f, 'is not for one of the profiles listed in the lat/lon file; skipping')

    print('Instantiating {} record'.format(gas_name))
    try:
        gas_rec = tccon_priors.gas_records[gas_name.lower()]()
    except KeyError:
        raise RuntimeError('No record defined for gas_name = "{}"'.format(gas_name))

    prior_args = []

    for k, files in grouped_mod_files.items():
        for f in files:
            these_args = (f, gas_rec, trop_eqlat, zgrid_file)
            prior_args.append(these_args)

    atm_files_by_mod = _make_mod_atm_map(acdates=acdate_strings, aclons=aclons, aclats=aclats, acfiles=acfiles)
    mod_files_in_order = [args[0] for args in prior_args]
    atm_files = [atm_files_by_mod[os.path.basename(f)] for f in mod_files_in_order]
    atm_files_flat = []
    for f in atm_files:
        atm_files_flat += f

    nmissing = 0
    for f in acfiles:
        if f not in atm_files_flat:
            nmissing += 1
            print('.mod files necessary for {} were missing!'.format(f))

    if nmissing > 0:
        raise RuntimeError('{} .mod files were missing!'.format(nmissing))
    
    if nprocs == 0:
        results = []
        for args in prior_args:
            results.append(_prior_helper(*args))
    else:
        with Pool(processes=nprocs) as pool:
            results = pool.starmap(_prior_helper, prior_args)

    _write_priors_h5(prior_save_file, results, atm_files, mod_files_in_order,
                     root_attrs={'trop_eqlat': trop_eqlat, 'zgrid_file': zgrid_file if zgrid_file is not None else ''})


def _prior_helper(mod_file, gas_rec, trop_eqlat=True, zgrid=None):
    """
    Helper function to run the priors consistently in either serial or parallel mode

    :param mod_file: the path to the mod file to generate a prior for
    :type mod_file: str

    :param gas_rec: the :class:`~ginput.priors.tccon_priors.TraceGasRecord` subclass that specifies which gas to
     generate the priors for.
    :type gas_rec: :class:`~ginput.priors.tccon_priors.TraceGasRecord`

    :param zgrid: the levels file specifying what altitude levels to interpolate the priors to, or None to leave them
     on the GEOS grid.
    :type zgrid: str

    :return: the output of :func:`~ginput.priors.tccon_priors.generate_single_tccon_prior`
    """
    _fbase = os.path.basename(mod_file)
    print('Processing {}'.format(_fbase))
    return tccon_priors.generate_single_tccon_prior(mod_file, tdel(hours=0), gas_rec, use_eqlat_strat=True, use_eqlat_trop=trop_eqlat, zgrid=zgrid)


def _write_priors_h5(save_file, prior_results, atm_files, mod_files=None, root_attrs=dict()):
    """
    Write the output HDF5 file for the priors

    :param save_file: the file name to create
    :type save_file: str

    :param prior_results: the list of results from :func:`~ginput.priors.tccon_priors.generate_single_tccon_prior`, i.e.
     each element should be one tuple of dicts returned by that function.
    :type prior_results: list

    :param atm_files: the list of .atm files, one per element of ``prior_results``. Each element should correspond to
     the results in the same index of ``prior_results``.
    :type atm_files: list(str)

    :param mod_files: the list of .mod files corresponding to the prior results. If not given, then the .mod data will
     not be written into the .h5 file. These must be full paths to the .mod files.
    :type mod_files: list(str)

    :return: none, writes HDF5 file
    """
    def make_h5_array(data_list, data_key):
        axis = np.ndim(data_list[0][data_key])
        data_list = [el[data_key] for el in data_list]
        return np.stack(data_list, axis=axis).T

    def convert_h5_array_type(var_array):
        if np.issubdtype(var_array.dtype, np.floating):
            attrs = dict()
            fill_val = float_h5_fill
        elif np.issubdtype(var_array.dtype, np.integer):
            attrs = dict()
            fill_val = int_h5_fill
        elif np.issubdtype(var_array.dtype, np.string_) or np.issubdtype(var_array.dtype, np.unicode_):
            shape = var_array.shape
            var_array = np.array([s.encode('utf8') for s in var_array.flat])
            var_array = var_array.reshape(shape)
            attrs = dict()
            fill_val = string_h5_fill
        elif np.issubdtype(var_array.dtype, np.bool_):
            attrs = dict()
            fill_val = None
        elif var_array.dtype == np.object_:
            if hasattr(var_array.flatten()[0], 'strftime'):
                # probably some kind of date
                var_array = var_array.astype('datetime64[s]').astype('int')
                var_array[var_array < 0] = int_h5_fill
                attrs = {'units': 'seconds since 1970-01-01'}
                fill_val = int_h5_fill
            elif var_array.flatten()[0] is None:
                if all(v is None for v in var_array.flat):
                    var_array = np.full_like(var_array, float_h5_fill, dtype=float)
                    attrs = dict()
                    fill_val = float_h5_fill
                else:
                    raise NotImplementedError('Some, but not all, values are None')
            else:
                obj_type = type(var_array.flatten()[0]).__name__
                raise NotImplementedError('Converting objects of type "{}" not implemented'.format(obj_type))
        else:
            raise NotImplementedError('Arrays with datatype "{}" not implemented'.format(var_array.dtype))
        return var_array, attrs, fill_val

    def expand_atm_lists(atm_files_local):
        maxf = max(len(files) for files in atm_files_local)
        atm_files_out = []
        for files in atm_files_local:
            n = len(files)
            files += [string_h5_fill] * (maxf - n)
            atm_files_out.append(files)
        return np.array(atm_files_out)

    with h5py.File(save_file, 'w') as wobj:
        profiles, units, scalars = zip(*prior_results)

        # First record the .atm files that correspond to each profile. Allow for the possibility that we might have
        # multiple atm files corresponding to a profile by making the array be nprofiles-by-nfiles. Fill values will
        # fill out rows that don't have the max number of files.
        atm_files = expand_atm_lists(atm_files)
        atm_file_array, atm_attrs, atm_fill_val = convert_h5_array_type(atm_files)
        dset = wobj.create_dataset('atm_files', data=atm_file_array, fillvalue=atm_fill_val)
        dset.attrs.update(atm_attrs)

        # Then go ahead and add the root attributes
        all_root_attrs = {'history': ioutils.make_creation_info(save_file, 'ginput.priors.backend_analysis.create_test_priors')}
        all_root_attrs.update(root_attrs)
        wobj.attrs.update(all_root_attrs)

        # Write the profile and scalar variables
        prior_grp = wobj.create_group('Priors')
        prof_grp = prior_grp.create_group('Profiles')
        for key in profiles[0].keys():
            print('Writing', key)
            prof_array, prof_attrs, this_fill_val = convert_h5_array_type(make_h5_array(profiles, key))
            prof_attrs['units'] = units[0][key]  # assume the units are the same for all profiles
            dset = prof_grp.create_dataset(key, data=prof_array, fillvalue=this_fill_val)
            dset.attrs.update(prof_attrs)

        scalar_grp = prior_grp.create_group('Scalars')
        for key in scalars[0].keys():
            scalar_array, scalar_attrs, this_fill_val = convert_h5_array_type(make_h5_array(scalars, key))
            dset = scalar_grp.create_dataset(key, data=scalar_array, fillvalue=this_fill_val)
            dset.attrs.update(scalar_attrs)

        # Lastly, if mod files were given, read them in and pack their data into the .h5 file as well so that if we
        # need to look at a met variable we can
        if mod_files is not None:
            raw_mod_dat = [readers.read_mod_file(f) for f in mod_files]
            mod_profiles = [dat['profile'] for dat in raw_mod_dat]
            mod_scalars = [dat['scalar'] for dat in raw_mod_dat]

            mod_grp = wobj.create_group('Models')
            mod_file_array, mod_file_attrs, mod_file_fill = convert_h5_array_type(np.array([os.path.basename(f) for f in mod_files]))
            dset = mod_grp.create_dataset('mod_files', data=mod_file_array, fillvalue=mod_file_fill)
            dset.attrs.update(mod_file_attrs)

            mod_prof_grp = mod_grp.create_group('Profiles')
            for key in mod_profiles[0].keys():
                prof_array, prof_attrs, this_fill_val = convert_h5_array_type(make_h5_array(mod_profiles, key))
                dset = mod_prof_grp.create_dataset(key, data=prof_array, fillvalue=this_fill_val)
                dset.attrs.update(prof_attrs)

            mod_scalar_grp = mod_grp.create_group('Scalars')
            for key in mod_scalars[0].keys():
                scalar_array, scalar_attrs, this_fill_val = convert_h5_array_type(make_h5_array(mod_scalars, key))
                dset = mod_scalar_grp.create_dataset(key, data=scalar_array, fillvalue=this_fill_val)
                dset.attrs.update(scalar_attrs)


def driver(check_geos, download, makemod, makepriors, site_file, geos_top_dir, geos_chm_top_dir,
           mod_top_dir, prior_save_file, gas_name, nprocs=0, dl_file_types=None, dl_levels=None, integral_file=None,
           trop_eqlat=True, **_):
    if dl_file_types is None:
        dl_file_types = ('met', 'met', 'chm')
    if dl_levels is None:
        dl_levels = ('surf', 'eta', 'eta')

    aclons, aclats, acdates, acfiles = read_date_lat_lon_file(site_file)
    if check_geos:
        check_geos_files(acdates, geos_top_dir, chem_download_dir=geos_chm_top_dir,
                         file_type=dl_file_types, levels=dl_levels)

    if download:
        download_geos(acdates, geos_top_dir, chem_download_dir=geos_chm_top_dir,
                      file_type=dl_file_types, levels=dl_levels)
    else:
        print('Not downloading GEOS data')

    if makemod:
        make_mod_files(acdates, aclons, aclats, geos_top_dir, mod_top_dir, chem_dir=geos_chm_top_dir, nprocs=nprocs)
    else:
        print('Not making .mod files')

    if makepriors:
        make_priors(prior_save_file, make_full_mod_dir(mod_top_dir, 'fpit'), gas_name,
                    acdates=acdates, aclons=aclons, aclats=aclats, acfiles=acfiles, nprocs=nprocs, 
                    trop_eqlat=trop_eqlat, zgrid_file=integral_file)
    else:
        print('Not making priors')


def run_main(**args):
    info_file = args.pop('info_file')
    if info_file == 'format':
        print_config_help()
        sys.exit(0)
    else:
        info_dict = read_info_file(info_file)

    args.update(info_dict)
    driver(**args)


def parse_run_args(parser):
    parser.description = 'Generate priors for a given set of dates, lats, and lons'
    parser.add_argument('info_file', help='The file that defines the configuration variables. Pass "format" as this '
                                          'argument for more details on the format.')
    parser.add_argument('--check-geos', action='store_true', help='Check if the required GEOS files are already downloaded')
    parser.add_argument('--download', action='store_true', help='Download GEOS FP-IT files needed for these priors.')
    parser.add_argument('--makemod', action='store_true', help='Generate the .mod files for these priors.')
    parser.add_argument('--makepriors', action='store_true', help='Generate the priors as .map files.')
    parser.add_argument('-n', '--nprocs', default=0, type=int, help='Number of processors to use to run in parallel mode '
                                                          '(for --makemod and --makepriors only)')
    parser.add_argument('--dl-file-types', default=None, choices=get_GEOS5._file_types,
                        help='Which GEOS file types to download with --download (no effect if --download not specified).')
    parser.add_argument('--dl-levels', default=None, choices=get_GEOS5._level_types,
                        help='Which GEOS levels to download with --download (no effect if --download not specified).')
    parser.set_defaults(driver_fxn=run_main)


def parse_make_info_args(parser: argparse.ArgumentParser):
    parser.description = 'Make the list of dates, lats, and lons required to generate priors'
    parser.add_argument('list_file', help='Name to give the information file created')
    parser.add_argument('atm_files', nargs='+', help='.atm files to generate priors for')
    parser.set_defaults(driver_fxn=make_lat_lon_list_for_atms)


def parse_args():
    parser = argparse.ArgumentParser('Tools for creating priors to test against observed profiles from .atm files')
    subp = parser.add_subparsers()

    runp = subp.add_parser('run', help='Download GEOS data, generate .mod files, and/or generate priors')
    parse_run_args(runp)

    listp = subp.add_parser('make-list', help='Make a list of dates, lats, and lons to generate priors for')
    parse_make_info_args(listp)
    return vars(parser.parse_args())


def print_config_help():
    prologue = """The info file is a simple text file where the lines follow the format

key = value

where key is one of {keys}. 
These keys are required; order does not matter. 
The value expected for each key is:""".format(keys=', '.join(_req_info_keys))

    epilogue = """The keys {paths} are file paths. 
These may be given as absolute paths, or as relative paths. 
If relative, they will be taken as relative to the 
location of the info file.""".format(paths=', '.join(_req_info_ispath))

    print(prologue + '\n')
    for key, value in _req_info_help.items():
        print('* {}: {}'.format(key, value))

    print('\nThere are additional optional keys:\n')
    for key, value in _opt_info_help.items():
        print('* {}: {}'.format(key, value))

    print('\n' + epilogue)


def main():
    args = parse_args()
    main_fxn = args.pop('driver_fxn')
    main_fxn(**args)


if __name__ == '__main__':
    main()
