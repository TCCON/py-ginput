from collections import OrderedDict
import datetime as dt
import os
from pathlib import Path
import re

import netCDF4 as ncdf
import numpy as np
import pandas as pd

from . import mod_utils
from .mod_utils import ModelError
from .versioning import GeosVersion, GeosSource
from .ggg_logging import logger

from typing import Dict, Union

def read_out_file(out_file, as_dataframes=False, replace_fills=False):
    """Read a GGG output file

    This function can read most GGG output files, including .col files and post processing
    text files (but not the .private.nc file). It expects a file where:

    * the first line is a sequence of numbers, with the first one being the number of header
      lines
    * the last header line is a space-separated list of column names
    * the rest of the file is a space-separated data table.

    Parameters
    ----------
    out_file : str
        Path to the file to read

    as_dataframes : bool
        If `True`, then the output file is read in as a dataframe. If `False`, it is converted
        to a dictionary when returning.

    replace_fills : bool or float
        If `False`, fill values are kept as is. If `True`, this will look for a line in the header
        with the format `"missing:  <number>`. Any number within floating point rounding of that 
        number are replaced with NaNs. If this is a number, then any values greater than or equal
        to that number are treated as fills and replaced with NaNs. This last is the safest; GGG
        post processing typically uses numbers >1e35 as "missing" values, so usually any value greater
        than that can be replaced. 

        Note that `TypeError` occurences during comparison are ignored. This allows e.g. the "spectrum"
        column to be skipped, but means that any formatting issues that cause a column to be read in
        as an series of objects, rather than a numeric type, can cause that column to be skipped.

    Returns
    -------
    pandas.DataFrame or dict
        The contents of the file as a dataframe or dict, depending on `as_dataframes`.
    """
    n_header_lines = mod_utils.get_num_header_lines(out_file)
    df = pd.read_csv(out_file, header=n_header_lines-1, sep='\s+')

    if replace_fills is True:
        # Get the fill value from the header
        with open(out_file) as f:
            fill_value = None
            for _ in range(n_header_lines):
                line = f.readline()
                if line.startswith('missing:'):
                    fill_value = float(line.split(':', 1)[1])
                    break

            if fill_value is None:
                raise IOError('Could not find fill value in the header of {}'.format(out_file))

            is_fill = lambda val: np.isclose(val, fill_value)

    elif replace_fills is not False:
        fill_value = replace_fills
        replace_fills = True
        is_fill = lambda val: val >= fill_value

    if replace_fills:
        for colname, coldata in df.iteritems():
            try:
                xx_fills = is_fill(coldata)
            except TypeError:
                continue
            else:
                df.loc[xx_fills, colname] = np.nan

    if not as_dataframes:
        return df.to_dict()
    else:
        return df


def read_mod_file(mod_file, as_dataframes=False):
    """
    Read a TCCON .mod file.

    :param mod_file: the path to the mod file.
    :type mod_file: str

    :param as_dataframes: if ``True``, then the collection of variables will be kept as dataframes. If ``False``
     (default), they are converted to dictionaries of floats or numpy arrays.
    :type as_dataframes: bool

    :return: a dictionary with keys 'file' (values derived from file name), 'constants' (constant values stored in the
     .mod file header), 'scalar' (values like surface height and tropopause pressure that are only defined once per
     profile) and 'profile' (profile variables) containing the respective variables. These values will be dictionaries
     or data frames, depending on ``as_dataframes``.
    :rtype: dict
    """
    n_header_lines = mod_utils.get_num_header_lines(mod_file)
    # Read the constants from the second line of the file. There's no header for these, we just have to rely on the
    # same constants being in the same position.
    constant_vars = pd.read_csv(mod_file, sep=r'\s+', header=None, nrows=1, skiprows=1,
                                names=('earth_radius', 'ecc2', 'obs_lat', 'surface_gravity',
                                       'profile_base_geometric_alt', 'base_pressure', 'tropopause_pressure'))
    # Read the scalar variables (e.g. surface pressure, SZA, tropopause) first. We just have to assume their headers are
    # on line 3 and values on line 4 of the file, the first number in the first line gives us the line the profile
    # variables start on.
    scalar_vars = pd.read_csv(mod_file, sep=r'\s+', header=2, nrows=1)

    # Get the GEOS versions from the header. Also extract the CO source, since the priors
    # code relies on that and we should support pre-version 1.2.1 files.
    geos_versions = _read_mod_file_geos_sources(mod_file)
    constant_vars['co_source'] = _read_mod_file_co_source(mod_file, geos_versions)

    # Now read the profile vars.
    profile_vars = pd.read_csv(mod_file, sep=r'\s+', header=n_header_lines-1)

    # Also get the information that's only in the file name (namely date and longitude, we'll also read the latitude
    # because it's there).
    file_vars = dict()
    base_name = os.path.basename(mod_file)
    file_vars['datetime'] = mod_utils.find_datetime_substring(base_name, out_type=dt.datetime)
    file_vars['lon'] = mod_utils.find_lon_substring(base_name, to_float=True)
    file_vars['lat'] = mod_utils.find_lat_substring(base_name, to_float=True)

    # Check that the header latitude and the file name latitude don't differ by more than 0.5 degree. Even if rounded
    # to an integer for the file name, the difference should not exceed 0.5 degree.
    lat_diff_threshold = 0.5
    if np.abs(file_vars['lat'] - constant_vars['obs_lat'].item()) > lat_diff_threshold:
        raise ModelError('The latitude in the file name and .mod file header differ by more than {lim} deg ({name} vs. '
                         '{head}). This indicates a possibly malformed .mod file.'
                         .format(lim=lat_diff_threshold, name=file_vars['lat'], head=constant_vars['obs_lat'].item())
                         )

    out_dict = dict()
    if as_dataframes:
        out_dict['file'] = pd.DataFrame(file_vars, index=[0])
        out_dict['constants'] = constant_vars
        out_dict['scalar'] = scalar_vars
        out_dict['profile'] = profile_vars
    else:
        out_dict['file'] = file_vars
        out_dict['constants'] = {k: v.item() for k, v in constant_vars.items()}
        out_dict['scalar'] = {k: v.item() for k, v in scalar_vars.items()}
        out_dict['profile'] = {k: v.values for k, v in profile_vars.items()}
    out_dict['geos_versions'] = geos_versions
    return out_dict


def _read_mod_file_geos_sources(mod_file: str) -> Dict[str, GeosVersion]:
    nhead = mod_utils.get_num_header_lines(mod_file)
    sources = dict()
    with open(mod_file) as f:
        for idx, line in enumerate(f):
            if idx == nhead:
                return sources
            elif line.startswith('GEOS source'):
                # Assume a line like "GEOS source : Met3d : {version info} : {filename} : {checksum}"
                parts = line.split(':', maxsplit=2)
                key = parts[1].strip()
                info = parts[2].strip()
                sources[key] = GeosVersion.from_str(info)
                


def _read_mod_file_co_source(mod_file: str, geos_versions: Dict[str, GeosVersion]) -> GeosSource:
    # >= v1.2.1 .mod files (or those patched to look like them) have all GEOS sources in the header,
    # so we don't need to re-read the file.
    if geos_versions:
        chm_version = geos_versions.get('Chm3d')
        return GeosSource.UNKNOWN if chm_version is None else chm_version.source

    # v1.2.0 files have the line "CO source" in the header and pre-v1.2.0 files have no indication
    # of version.
    nhead = mod_utils.get_num_header_lines(mod_file)
    with open(mod_file) as f:
        for idx, line in enumerate(f):
            if idx == nhead:
                # This is the default; .mod files before v1.2.0 did not include a CO
                # source in the header because it was always from GEOS FP-IT. Also 
                # prior to v1.2.0, the header only had 7 lines (by default), so if
                # there's >7 lines, that probably means we messed up.
                if nhead > 7:
                    logger.warning((f'In .mod file {mod_file}, did not find a "CO source" line in the header, but the header has more than 7 lines. '
                                    'Unless this is a custom .mod file, this means I may have missed the CO source line.'))
                return GeosSource.FPIT
            elif line.startswith('CO source'):
                source = line.split(':', maxsplit=2)[1].strip()
                return GeosSource(source)


def read_mod_file_units(mod_file):
    """
    Get the units for the profile variables in a .mod file

    :param mod_file: the .mod file to read
    :type mod_file: str

    :return: a dictionary with the variable names as keys and the units as values.
    """
    n_header_lines = mod_utils.get_num_header_lines(mod_file)
    # Assume that the profile units are the second to last line of the header
    # and the profile variable names are the last line
    with open(mod_file, 'r') as robj:
        for i in range(n_header_lines):
            line = robj.readline()
            if i == (n_header_lines-2):
                units = line.split()
            elif i == (n_header_lines-1):
                names = line.split()

    return {n: u for n, u in zip(names, units)}
 

def read_map_file(map_file, as_dataframes=False, skip_header=False, fmt=None):
    """
    Read a .map file

    :param map_file: the path to the .map file
    :type map_file: str

    :param as_dataframes: set to ``True`` to return the constants and profiles data as Pandas dataframes. By default,
     (``False``) they are returned as dictionaries of numpy arrays.
    :type as_dataframes: bool

    :param skip_header: set to ``True` to avoid reading the header. This is helpful for reading .map files that
     have a slightly different header format.
    :type skip_header: bool

    :param fmt: can use to tell this function which file type the .map file is, either "txt" for text or "nc" for
     netCDF. If not specified, it will attempt to infer from the extension.
    :type fmt: str or None

    :return: a dictionary with keys 'constants' and 'profile' that hold the header values and main profile data,
     respectively. The form of these values depends on ``as_dataframes``.
    :rtype: dict
    """
    if map_file.endswith('.nc') or fmt == 'nc':
        return _read_map_nc_file(map_file, as_dataframes=as_dataframes, skip_header=skip_header)
    elif map_file.endswith('.map') or fmt == 'txt':
        return _read_map_txt_file(map_file, as_dataframes=as_dataframes, skip_header=skip_header)
    else:
        ext = os.path.splitext(map_file)
        raise ValueError('Unknown extension "{}" for map file. Use the `fmt` keyword to indicate the file type.'.format(ext))


def _read_map_nc_file(map_file, as_dataframes=False, skip_header=False):
    profile_dict = dict()
    constant_dict = dict()
    with ncdf.Dataset(map_file) as ds:
        for varname, vardat in ds.variables.items():
            if varname == 'time':
                vardat = ncdf.num2date(vardat[:], vardat.units, only_use_cftime_datetimes=False)
                profile_dict[varname] = pd.DatetimeIndex(vardat)
            else:
                profile_dict[varname] = vardat[:].filled(np.nan)

        if not skip_header:
            for attr in ds.ncattrs():
                if attr.startswith('constant') and not attr.endswith('units'):
                    k = attr.replace('constant_', '')
                    constant_dict[k] = ds.getncattr(attr)

    if as_dataframes:
        # these are both scalar values that can't go into the dataframe
        profile_dict.pop('time')
        profile_dict.pop('lat')
        profile_dict = pd.DataFrame(profile_dict)
        profile_dict.set_index('altitude', inplace=True)

        constant_dict = pd.DataFrame(constant_dict, index=[0])

    return {'profile': profile_dict, 'constants': constant_dict}


def _read_map_txt_file(map_file, as_dataframes=False, skip_header=False):
    n_header_lines = mod_utils.get_num_header_lines(map_file)
    constants = dict()
    if not skip_header:
        with open(map_file, 'r') as mapf:
            n_skip = 4
            # Skip the first four lines to get to the constants - these should be (1) the number of header lines &
            # columns, (2) filename, (3) version info, and (4) wiki reference.
            for i in range(n_skip):
                mapf.readline()

            # The last two lines of the header are the column names and units; everything between line 5 and that should
            # be physical constants. Start at n_skip+1 to account for 0 indexing vs. number of lines.

            for i in range(n_skip+1, n_header_lines-1):
                line = mapf.readline()
                # Lines have the form Name (units): value - ignore anything in parentheses
                name, value = line.split(':')
                name = re.sub(r'\(.+\)', '', name).strip()
                constants[name] = float(value)

    df = pd.read_csv(map_file, header=n_header_lines-2, skiprows=[n_header_lines-1], na_values='NAN')
    # Sometimes extra space gets kept in the headers - remove that
    df.rename(columns=lambda h: h.strip(), inplace=True)
    if not as_dataframes:
        data = {k: v.values for k, v in df.items()}
    else:
        data = df
        constants = pd.DataFrame(constants, index=[0])

    out_dict = dict()
    out_dict['constants'] = constants
    out_dict['profile'] = data
    return out_dict


def read_isotopes(isotopes_file, gases_only=False):
    """
    Read the isotopes defined in an isotopologs.dat file

    :param isotopes_file: the path to the isotopologs.dat file
    :type isotopes_file: str

    :param gases_only: set to ``True`` to return a tuple of only the distinct gases, not the individual isotopes.
     Default is ``False``, which includes the different isotope numbers.
    :type gases_only: bool

    :return: tuple of isotope or gas names
    :rtype: tuple(str)
    """
    nheader = mod_utils.get_num_header_lines(isotopes_file)
    with open(isotopes_file, 'r') as fobj:
        for i in range(nheader):
            fobj.readline()

        isotopes = []
        for line in fobj:
            iso_number = line[3:5].strip()
            iso_name = line[6:14].strip()
            if not gases_only:
                iso_name = iso_number + iso_name
            if iso_name not in isotopes:
                isotopes.append(iso_name)

        return tuple(isotopes)


def read_vmr_file(vmr_file, as_dataframes=False, lowercase_names=True, style='new'):
    nheader = mod_utils.get_num_header_lines(vmr_file)

    if style == 'new':
        last_const_line = nheader - 1
        old_style = False
    elif style == 'old':
        last_const_line = 4
        old_style = True
    else:
        raise ValueError('style must be one of "new" or "old"')

    header_data = dict()
    with open(vmr_file, 'r') as fobj:
        # Skip the line with the number of header lines and columns
        fobj.readline()
        for i in range(1, last_const_line):
            line = fobj.readline()
            const_name, const_val = [v.strip() for v in line.split(':')]
            if lowercase_names:
                const_name = const_name.lower()

            try:
                const_val = float(const_val)
            except ValueError:
                pass
            header_data[const_name] = const_val

        prior_info = dict()
        if old_style:
            for i in range(last_const_line, nheader-1, 2):
                category_line = fobj.readline()
                category = re.split(r'[:\.]', category_line)[0].strip()
                data_line = fobj.readline()
                data_line = data_line.split(':')[1].strip()
                split_data_line = re.split(r'\s+', data_line)
                prior_info[category] = np.array([float(x) for x in split_data_line])

    data_table = pd.read_csv(vmr_file, sep='\s+', header=nheader-1)
    
    # Also get the information that's only in the file name (namely date and longitude, we'll also read the latitude
    # because it's there).
    file_vars = dict()
    base_name = os.path.basename(vmr_file)
    try:
        file_vars['datetime'] = mod_utils.find_datetime_substring(base_name, out_type=dt.datetime)
        file_vars['lon'] = mod_utils.find_lon_substring(base_name, to_float=True)
        file_vars['lat'] = mod_utils.find_lat_substring(base_name, to_float=True)
    except AttributeError:
        # Happens when the regex can't find a date/lon/lat in the file name
        # usually means we're reading a climatological file
        file_vars = dict(datetime=None, lon=None, lat=None)

    if lowercase_names:
        data_table.columns = [v.lower() for v in data_table]

    if as_dataframes:
        header_data = pd.DataFrame(header_data, index=[0])
        # Rearrange the prior info dict so that the data frame has the categories as the index and the species as the
        # columns.
        categories = list(prior_info.keys())
        tmp_prior_info = dict()
        for i, k in enumerate(data_table.columns.drop('altitude')):
            tmp_prior_info[k] = np.array([prior_info[cat][i] for cat in categories])
        prior_info = pd.DataFrame(tmp_prior_info, index=categories)
    else:
        # use an ordered dict to ensure we keep the order of the gases. This is important if we use this .vmr file as
        # a template to write another .vmr file that gsetup.f can read.
        data_table = OrderedDict([(k, v.to_numpy()) for k, v in data_table.items()])

    return {'scalar': header_data, 'profile': data_table, 'prior_info': prior_info, 'file': file_vars}


def read_runlog(runlog_file, as_dataframes=False, remove_commented_lines=True):
    nhead = mod_utils.get_num_header_lines(runlog_file)
    with open(runlog_file, 'r') as robj:
        for i in range(nhead-2):
            # Skip ahead to the last two lines of the header which contain the fortran format specification and the
            # column names
            robj.readline()

        fmt_line = robj.readline()
        # we need to read the column names twice. read_fwf expects the first line it reads to be the column names but
        # with the right widths. Since that's not the case for the runlog, we read in the line manually once (to parse
        # later into the proper column names) then rewind the file pointed to the beginning of that line so that
        # read_fwf reads it in as the column names. It'll get them wrong, but at least it doesn't stick a data row in
        # as the header.
        pos = robj.tell()
        column_names_line = robj.readline()
        robj.seek(pos)

        # the format line will be something like: "format=(a1,a57,1x,2i4,f8.4,f8.3,f9.3,...)\n"; we want to pass only
        # the part inside the parentheses (including the parentheses)
        colspecs, _ = mod_utils.fortran_fmt_to_fwf_tuples(fmt_line.split('=')[1].strip())
        df = pd.read_fwf(robj, colspecs=colspecs)

        # Fix the header. The first column is just where the colon goes if the line is commented out so we have to give
        # that a name (there isn't one in the header)
        column_names = column_names_line.split()
        df.columns = ['comment'] + column_names_line.split()

        # Remove comments if requested.
        if remove_commented_lines:
            df = df[~(df['comment'].apply(lambda x: x == ':'))]

        # Always remove the comment column itself
        df = df[column_names]

        if as_dataframes:
            return df
        else:
            return df.to_dict()


def read_tabular_file_with_header(file_path: Union[str, Path], comment_str: str = '#') -> pd.DataFrame:
    with open(file_path) as f:
        for line in f:
            if not line.startswith(comment_str):
                break

        columns = line.strip().split()
        df = pd.read_csv(f, header=None, delim_whitespace=True)
        df.columns = columns
        return df