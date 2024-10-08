"""
Utility functions shared among the mod maker and prior generation code.

Compatibility notes:
    * We have found that Pandas version 0.20 is incompatible with this module. Pandas >= 0.24 works.
"""

from __future__ import print_function, division

import datetime as dt
from datetime import timedelta
from glob import glob
from hashlib import md5

import xarray as xr
from dateutil.relativedelta import relativedelta
import netCDF4 as ncdf
import numpy as np
from numpy import ma
import os
import pandas as pd
import re

from numpy.core._multiarray_umath import arctan, tan, sin, cos
from scipy.interpolate import interp1d, interp2d
import subprocess
import sys
from warnings import warn

from . import mod_constants as const
from .mod_constants import days_per_year
from ..mod_maker.tccon_sites import site_dict

_std_model_pres_levels = np.array([1000.0, 975.0, 950.0, 925.0, 900.0, 875.0, 850.0, 825.0, 800.0, 775.0, 750.0, 725.0,
                                   700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0, 150.0,
                                   100.0, 70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.7, 0.5,
                                   0.4, 0.3, 0.1])
# as of 2019-03-14, both GEOS-FP and MERRA-2 use the same standard 42 pressure levels
# (c.f. https://gmao.gsfc.nasa.gov/GMAO_products/documents/GEOS_5_FP_File_Specification_ON4v1_2.pdf page 52 and
# https://gmao.gsfc.nasa.gov/reanalysis/MERRA/docs/MERRA_File_Specification.pdf page 27 for GEOS-FP and MERRA-2
# respectively)
merra_pres_levels = _std_model_pres_levels
geosfp_pres_levels = _std_model_pres_levels
earth_radius = 6371  # kilometers


class TropopauseError(Exception):
    """
    Error if could not find the tropopause
    """
    pass


class ModelError(Exception):
    """
    Error if a model file is nonsensical.
    """
    pass


class GGGPathError(Exception):
    pass


class ModPathError(IOError):
    pass


class ProgressBar(object):
    """
    Create a text-based progress bar

    An instance of this class can be used to print a text progress bar that does not need a new line for each progress
    step. It uses carriage returns to reset to the beginning of each line before printing the next. This therefore
    does not work well if other print statements occur in between calls to :meth:`print_bar`, the progress bar will
    either end up on a new line anyway or potentially overwrite previous print statements if they did not end with a
    newline.

    :param num_symbols: how many steps there should be in the progress bar. In other words, the progress bar will be
     complete when :meth:`print_bar` is called with ``num_symbols-1``.
    :type num_symbols: int

    :param prefix: a string to include before the beginning of each progress bar. The class will ensure that at least
     one space is present between the prefix and the progress bar, but will not add one if one is already present at
     the end of the prefix.
    :type prefix: str

    :param suffix: a string to include at the end of each progress bar. The class will ensure that at least one space
     is present between the progress bar and the suffix, but will not add one if one is already present at the beginning
     of the suffix.
    :type suffix: str

    :param add_one: if ``True``, the number of symbols printed in the progress bar is equal to ``i+1`` where ``i`` is
     the argument to :meth:`print_bar`. This works well with Python loops over ``i in range(n)``, since the last value
     of ``i`` will be ``n-1``, setting ``add_one`` to ``True`` ensures that a full progress bar is printed at the end.
    :type add_one: bool

    :param style: can be either '*' or 'counter'. The former prints a symbolic progress bar of the form:

        [*   ]
        [**  ]
        [*** ]
        [****]

     where the number of *'s is set by ``num_symbols``. The latter will instead print 'i/num_symbols' for each step.
    :type style: str
    """
    def __init__(self, num_symbols, prefix='', suffix='', add_one=True, style='*'):
        """
        See class help.
        """
        if len(prefix) > 0 and not prefix.endswith(' '):
            prefix += ' '
        if len(suffix) > 0 and not suffix.startswith(' '):
            suffix = ' ' + suffix

        if style == '*':
            self._fmt_str = '{pre}[{{pstr:<{n}}}]{suf}'.format(pre=prefix, n=num_symbols, suf=suffix)
        elif style == 'counter':
            self._fmt_str = '{pre}{{i:>{l}}}/{n}{suf}'.format(pre=prefix, n=num_symbols, suf=suffix, l=len(str(num_symbols)))
        else:
            raise ValueError('style "{}" not recognized'.format(style))
        self._add_one = add_one

    def print_bar(self, i):
        """
        Print the iteration of the progress bar corresponding to step ``i``.

        :param i: defines the progress step, either the number of *'s to print with ``style='*'`` or the counter number
         with ``style='counter'``.
        :type i: int
        :return: None, prints to screen.
        """
        if self._add_one:
            i += 1

        pstr = '*' * i
        pbar = self._fmt_str.format(pstr=pstr, i=i)
        sys.stdout.write('\r' + pbar)
        sys.stdout.flush()

    def finish(self):
        """
        Close the progress bar. By default, just prints a newline.
        :return: None
        """
        sys.stdout.write('\n')
        sys.stdout.flush()


def check_depedencies_newer(out_file, *dependency_files):
    """
    Check if
    :param out_file:
    :param dependency_files:
    :return:
    """
    if len(dependency_files) == 0:
        raise ValueError('Give at least one dependency file')

    out_last_modified = os.path.getmtime(out_file)
    for dep in dependency_files:
        if os.path.getmtime(dep) > out_last_modified:
            return True

    return False


def get_num_header_lines(filename):
    """
    Get the number of header lines in a standard GGG file

    This assumes that the file specified begins with a line with two numbers: the number of header rows and the number
    of data columns.

    :param filename: the file to read
    :type filename: str

    :return: the number of header lines
    :rtype: int
    """
    with open(filename, 'r') as fobj:
        header_info = fobj.readline()

    if ',' in header_info:
        header = header_info.split(',')
    else:
        header = header_info.split()
    return int(header[0])


def get_isotopes_file(isotopes_file=None, use_gggpath=False):
    """
    Get the path to the isotopologs.dat file

    :param isotopes_file: user input path. If this is not None, it is returned after checking that it exists.
    :type isotopes_file: str or None

    :param use_gggpath: set to ``True`` to find the isotopologs.dat file at the location defined by the GGGPATH
     environmental variable. If ``False`` and ``isotopes_file`` is ``None`` then the isotopologs.date file included in
     this repo is used.
    :type use_gggpath: bool

    :return: the path to the isotopologs.dat file
    :rtype: str
    """
    if isotopes_file is not None:
        if not os.path.isfile(isotopes_file):
            raise IOError('The isotopes path {} is not a file'.format(isotopes_file))
        return isotopes_file
    elif use_gggpath:
        gggpath = os.getenv('GGGPATH')
        if gggpath is None:
            raise EnvironmentError('use_gggpath=True requires the GGGPATH environmental variable to be set')
        isotopes_file = os.path.join(gggpath, 'isotopologs', 'isotopologs.dat')
        if not os.path.isfile(isotopes_file):
            raise IOError('Failed to find isotopologs.dat at {}. Either update your GGGPATH environmental variable or '
                          'set use_gggpath to False.'.format(isotopes_file))
        return isotopes_file
    else:
        return os.path.join(const.data_dir, 'isotopologs.dat')


def mod_output_subdir(top_dir, site_abbrev, product='fpit', slant=False):
    path_dir = 'slant' if slant else 'vertical'
    return os.path.join(top_dir, product, site_abbrev, path_dir)


def vmr_output_subdir(top_dir, site_abbrev, product='fpit', slant=False):
    path_dir = 'vmrs-slant' if slant else 'vmrs-vertical'
    return os.path.join(top_dir, product, site_abbrev, path_dir)


def vmr_file_name(obs_date, lon, lat, keep_latlon_prec=False, date_fmt='%Y%m%d%H', in_utc=True):
    """
    Construct the standard filename for a .vmr file produced by this code

    :param obs_date: the datetime of the profiles
    :type obs_date: datetime-like

    :param lon: the longitude of the profiles.
    :type lon: float

    :param lat: the latitude of the profiles
    :type lat: float

    :param keep_latlon_prec: by default, lat and lon are rounded to the nearest whole number. Set this to ``True`` to
     keep 2 decimal places of precision.
    :type keep_latlon_prec: bool

    :return: the .vmr file name, with format "JLv_yyyymmddhh_XX[NS]_YYY[EW].vmr" where "v" is the major version,
     "yyyymmddhh" the date/time, XX[NS] the latitude and YYY[EW] the longitude.
    :rtype: str
    """
    # The desired behavior is to zero pad if rounding to whole degrees but not otherwise. This mimics the behavior for
    # .mod files, although that function does it slightly differently, and reconciling the two should be done at some
    # point.
    prec = 2 if keep_latlon_prec else 0
    pad = False if keep_latlon_prec else True
    lat = format_lat(lat, prec=prec, zero_pad=pad)
    lon = format_lon(lon, prec=prec, zero_pad=pad)
    major_version = _get_priors_major_version()
    return 'JL{ver}_{date}{tz}_{lat}_{lon}.vmr'.format(ver=major_version, date=obs_date.strftime(date_fmt),
                                                       tz='Z' if in_utc else 'L', lat=lat, lon=lon)


def vmr_file_name_from_mod(mod_file_name: str) -> str:
    """
    Infer the base file name for a .vmr file given the corresponding .mod file.

    :param mod_file_name: The base name (NOT the full path) of the .mod file. This must have the
     where everything after the first underscore and leading up to the extension is shared with
     the .vmr file name.

    :returns: the base name for the .vmr file.
    """
    # Assume the mod file name is something like "FPIT_2021042300Z_52N_000W.mod" and we want
    # something like "JL1_2021042300Z_52N_000W.vmr". All we need do is replace the part before
    # the first underscore.
    date_loc = mod_file_name.split('_', maxsplit=1)[1].replace('.mod', '')
    return 'JL{}_{}.vmr'.format(_get_priors_major_version(), date_loc)


def _get_priors_major_version():
    return const.priors_version.split('.')[0]


def map_file_name_from_mod_vmr_files(site_abbrev, mod_file, vmr_file, map_fmt):
    if map_fmt not in {'nc', 'netcdf', 'txt', 'text'}:
        raise ValueError(f'Unknown value for map_fmt: {map_fmt}')
    is_ncdf = map_fmt in {'nc', 'netcdf'}

    vmr_date = find_datetime_substring(os.path.basename(vmr_file), out_type=dt.datetime)
    mod_date = find_datetime_substring(os.path.basename(mod_file), out_type=dt.datetime)
    if vmr_date != mod_date:
        raise RuntimeError('The .vmr and .mod files have different dates in their filenames!')
    vmr_lonstr = find_lon_substring(os.path.basename(vmr_file))
    mod_lonstr = find_lon_substring(os.path.basename(mod_file))
    if vmr_lonstr != mod_lonstr:
        raise RuntimeError('The .vmr and .mod files have different longitudes in their filenames!')
    vmr_latstr = find_lat_substring(os.path.basename(vmr_file))
    mod_latstr = find_lat_substring(os.path.basename(mod_file))
    if vmr_latstr != mod_latstr:
        raise RuntimeError('The .vmr and .mod files have different latitudes in their filenames!')
    
    return '{site}_{lat}_{lon}_{date}Z.map{suffix}'.format(
        site=site_abbrev, lat=vmr_latstr, lon=vmr_lonstr, 
        date=vmr_date.strftime('%Y%m%d%H'),
        suffix='.nc' if is_ncdf else ''
    )


def format_lon(lon, prec=2, zero_pad=False):
    """
    Convert longitude between string and numeric representations.

    If ``lon`` is a number, then it is converted to a string. The string will be the absolute value with "W" or "E" at
    the end to indicate west (<= 0) or east (> 0). If given a string in that format, it converts it to a number.

    :param lon: the longitude to convert
    :type lon: float or str

    :param prec: the precision after the decimal point to use. Only has an effect when converting float to string.
    :type prec: int

    :param zero_pad: set to ``True`` to zero pad the longitude string so that there are 3 digits before the decimal
     place. Only has an effect when converting float to string.

    :return: the formatted longitude string or the float representation of the longitude, with west being negative.
    :rtype: str or float
    """
    def to_str(lon):
        ew = 'E' if lon >= 0 else 'W'
        # In Python float format specification, "0X.Y" means to zero pad so that there's X total characters and Y after
        # the decimal point. We want lon zero padded to have three numbers before the decimal point, so the total width
        # needs to be the precision + 4 if there will be a decimal point otherwise just 3.
        width = prec + 4 if prec > 0 else prec + 3
        pad = '0{}'.format(width) if zero_pad else ''
        fmt_str = '{{:{padding}.{prec}f}}{{}}'.format(padding=pad, prec=prec)
        return fmt_str.format(abs(lon), ew)

    def to_float(lon):
        if lon[-1] == 'E':
            sign = 1
        elif lon[-1] == 'W':
            sign = -1
        else:
            raise ValueError('A longitude string must end in "E" or "W"')

        return float(lon[:-1]) * sign

    if isinstance(lon, str):
        return to_float(lon)
    else:
        if lon > 180:
            lon -= 360
        return to_str(lon)


def find_lon_substring(string, to_float=False):
    """
    Find a longitude substring in a string.

    A longitude substring will match \d+[EW] or \d+\.\d+[EW].

    :param string: the string to search for the longitude substring
    :type string: str

    :param to_float: when ``True``, converts the longitude to a float value using :func:`format_lon`, else returns the
     string itself.
    :type to_float: bool

    :return: the longitude substring or float value
    :rtype: str or float
    """
    # search for one or more numbers, which may include a decimal point followed by at least one number then E or W.
    lon_re = r'\d+(\.\d+)?[EW]'
    lon_str = re.search(lon_re, string).group()
    if to_float:
        return format_lon(lon_str)
    else:
        return lon_str


def format_lat(lat, prec=2, zero_pad=False):
    """
    Convert latitude between string and numeric representations.

    If ``lat`` is a number, then it is converted to a string. The string will be the absolute value with "N" or "S" at
    the end to indicate south (<= 0) or north (> 0). If given a string in that format, it converts it to a number.

    :param lat: the latitude to convert
    :type lat: float or str

    :param prec: the precision after the decimal point to use. Only has an effect when converting float to string.
    :type prec: int

    :param zero_pad: set to ``True`` to zero pad the latitude string so that there are 2 digits before the decimal
     place. Only has an effect when converting float to string.

    :return: the formatted latitude string or the float representation of the latitude, with south being negative.
    :rtype: str or float
    """
    def to_str(lat):
        ns = 'N' if lat >= 0 else 'S'
        # In Python float format specification, "0X.Y" means to zero pad so that there's X total characters and Y after
        # the decimal point. We want lat zero padded to have two numbers before the decimal point, so the total width
        # needs to be the precision + 3 if there will be a decimal point otherwise just 2.
        width = prec + 3 if prec > 0 else prec + 2
        pad = '0{}'.format(width) if zero_pad else ''
        fmt_str = '{{:{padding}.{prec}f}}{{}}'.format(padding=pad, prec=prec)
        return fmt_str.format(abs(lat), ns)

    def to_float(lat):
        if lat[-1] == 'N':
            sign = 1
        elif lat[-1] == 'S':
            sign = -1
        else:
            raise ValueError('A latitude string must end in "N" or "S"')

        return float(lat[:-1]) * sign

    if isinstance(lat, str):
        return to_float(lat)
    else:
        return to_str(lat)


def find_lat_substring(string, to_float=False):
    """
    Find a latitude substring in a string.

    A latitude substring will match \d+[NS] or \d+\.\d+[NS].

    :param string: the string to search for the latitude substring
    :type string: str

    :param to_float: when ``True``, converts the latitude to a float value using :func:`format_lat`, else returns the
     string itself.
    :type to_float: bool

    :return: the latitude substring or float value
    :rtype: str or float
    """
    # search for one or more numbers, which may include a decimal point followed by at least one number then N or S.
    lat_re = r'\d+(\.\d+)?[NS]'
    lat_str = re.search(lat_re, string).group()
    if to_float:
        return format_lat(lat_str)
    else:
        return lat_str


def find_datetime_substring(string, out_type=str):
    """
    Extract a date/time substring from a string.

    This assumes that the date/time is formatted as %Y%m%d (YYYYMMDD) or %Y%m%d_%H%M (YYYYMMDD_hhmm).

    :param string: the string to search for the date/time substring.
    :type string: str

    :param out_type: what type to return the date/time as. Default is to return the string. If another type is passed,
     then it must have a ``strptime`` class method that accepts the string to parse and the format string as arguments,
     i.e. it must behave like :func:`datetime.datetime.strptime`.
    :type out_type: type

    :return: the string or parsed datetime value
    """

    # Search for YYYYMMDDHH or YYYYMMDD_HHMM preferentially but allow just YYYYMMDD
    date_re = r'\d{8}(_\d{4}|\d{2})?'
    date_str = re.search(date_re, string).group()

    if out_type is str:
        return date_str
    else:
        date_fmts = {8: '%Y%m%d', 10: '%Y%m%d%H', 13: '%Y%m%d_%H%M'}
        date_fmt = date_fmts[len(date_str)]
        return out_type.strptime(date_str, date_fmt)
    

def _vcs_dir_helper(vcs_dir):
    if vcs_dir is None:
        vcs_dir = os.path.dirname(__file__)
    if len(vcs_dir) == 0:
        # If in the current directory, then dirname(__file__) gives an empty string, which isn't allowed as the argument
        # to cwd in check_output
        vcs_dir = '.'
    return os.path.abspath(vcs_dir)


def _is_git_repo(vcs_dir=None):
    vcs_dir = _vcs_dir_helper(vcs_dir)
    try:
        subprocess.check_call(['git', 'status'], cwd=vcs_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    else:
        return True

def vcs_commit_info(vcs_dir=None):
    """
    Return the last commit hash on the current branch, current branch, and last commit date for a VCS repo.

    This function will use Git if ``vcs_dir`` points to a valid Git repo and only fall back
    on Mercurial if ``vcs_dir`` is not a Git repo.

    :param vcs_dir: optional, the VCS directory to check. If not given, defaults to the one containing this repo.
    :type vcs_dir: str

    :return: the commit hash, branch name, and date. The date will be in the format e.g. "Mon Aug 01 18:20:15 2023 -0700",
     i.e. "%a %b %d %H:%M:%S %Y %z" format in terms of strftime codes.
    :rtype: str, str, str
    """
    if _is_git_repo(vcs_dir):
        return _git_commit_info(vcs_dir)
    else:
        return _hg_commit_info(vcs_dir)


def _git_commit_info(git_dir=None):
    git_dir = _vcs_dir_helper(git_dir)
    # Get the last commit in the current branch
    parent = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=git_dir).decode('utf8').strip()
    # Get the current branch
    branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=git_dir).decode('utf8').strip()
    # And the date of the last commit. Git by default doesn't seem to zero pad the date,
    # so force it to in order to match Mercurial
    date = subprocess.check_output(['git', 'log', '-1', r'--format=%cd', r'--date=format:"%a %b %d %H:%M:%S %Y %z"'], cwd=git_dir).decode('utf8').strip()

    return parent, branch, date


def _hg_commit_info(hg_dir=None):
    hg_dir = _vcs_dir_helper(hg_dir)

    # Get the last commit (-l 1) in the current branch (-f)
    summary = subprocess.check_output(['hg', 'log', '-f', '-l', '1'], cwd=hg_dir).splitlines()
    log_dict = dict()
    # Since subprocess returns a bytes object (at least on Linux) rather than an encoded string object, all the strings
    # below must be bytes, not unicode strings
    for line in summary:
        splitline = line.split(b':', 1)
        if len(splitline) < 2:
            continue
        k, v = splitline
        log_dict[k.strip()] = v.strip()

    parent = re.search(b'(?<=:)\\w+', log_dict[b'changeset']).group()
    # In Mercurial, if on the default branch, then log does not include a branch name in the output
    branch = log_dict[b'branch'] if b'branch' in log_dict else b'default'
    parent_date = log_dict[b'date']
    # Convert to unicode strings to avoid them getting formatted as "b'abc'" or "b'default'" in unicode strings
    return parent.decode('utf8'), branch.decode('utf8'), parent_date.decode('utf8')

def vcs_is_commit_clean(vcs_dir=None, ignore_untracked=True, ignore_files=tuple()):
    """
    Checks if a VCS directory is clean.

    By default, a directory is considered clean if all tracked files have no uncommitted changes. Untracked files are
    not considered. Setting ``ignore_untracked`` to ``False`` means that there must be no untracked files for the
    directory to be clean. This function will use Git if ``vcs_dir`` points to a valid Git repo and only fall back
    on Mercurial if ``vcs_dir`` is not a Git repo.

    :param vcs_dir: optional, the VCS directory to check. If not given, defaults to the one containing this repo.
    :type vcs_dir: str

    :param ignore_untracked: optional, set to ``False`` to require that there be no untracked files in the directory for
     it to be considered clean.
    :type ignore_untracked: bool

    :param ignore_files: optional, a sequence of file paths (absolute or relative to the current working directory, not the
     repo root) to ignore for the sake of determining if the repo is clean.
    :type ignore_files: Sequence[str]

    :return: ``True`` if the directory is clean, ``False`` otherwise.
    :rtype: bool
    """
    if _is_git_repo(vcs_dir):
        return _git_is_commit_clean(vcs_dir, ignore_untracked=ignore_untracked, ignore_files=ignore_files)
    else:
        return _hg_is_commit_clean(vcs_dir, ignore_untracked=ignore_untracked, ignore_files=ignore_files)

def _in_vcs_ignore(f: str, root: str, ignore_files):
    """Returns True if ``f`` was in ``ignored_files``

    ``f`` must be a string that is a file path relative to ``root``, which is
    the root of the VCS repo. ``ignore_files`` then is a sequence of strings
    giving the list of files to ignore if they have changed since the last commit.
    Note that ``ignore_files`` will *not* be relative to ``root``, they must be 
    absolute paths or relative to the current working directory.
    """
    f = os.path.join(root, f)
    for ignore in ignore_files:
        if os.path.exists(ignore) and os.path.samefile(f, ignore):
            return True
    return False

def _git_is_commit_clean(git_dir=None, ignore_untracked=True, ignore_files=tuple()):
    git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('utf8').strip()
    summary = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf8').strip().splitlines()
    for line in summary:
        stat, filename = line.split(maxsplit=2)
        if stat == '??' and ignore_untracked:
            continue
        elif stat == '!!':
            # ignored file according to the git status manpage?
            continue
        elif _in_vcs_ignore(filename, git_root, ignore_files):
            continue
        else:
            return False
        
    return True


def _hg_is_commit_clean(hg_dir=None, ignore_untracked=True, ignore_files=tuple()):
    """
    Checks if a mercurial directory is clean.

    By default, a directory is considered clean if all tracked files have no uncommitted changes. Untracked files are
    not considered. Setting ``ignore_untracked`` to ``False`` means that there must be no untracked files for the
    directory to be clean.

    :param hg_dir: optional, the mercurial directory to check. If not given, defaults to the one containing this repo.
    :type hg_dir: str

    :param ignore_untracked: optional, set to ``False`` to require that there be no untracked files in the directory for
     it to be considered clean.
    :type ignore_untracked: bool

    :return: ``True`` if the directory is clean, ``False`` otherwise.
    :rtype: bool
    """
    hg_dir = _vcs_dir_helper(hg_dir)
    hg_root = subprocess.check_output(['hg', 'root'], cwd=hg_dir).strip()
    summary = subprocess.check_output(['hg', 'status'], cwd=hg_dir).splitlines()


    # Since subprocess returns a bytes object (at least on Linux) rather than an encoded string object, all the strings
    # below must be bytes, not unicode strings
    for line in summary:
        status, hg_file = [p.strip() for p in line.split(b' ', 1)]
        if ignore_untracked and status == b'?':
            pass
        elif _in_vcs_ignore(hg_file, hg_root, ignore_files):
            pass
        else:
            return False

    return True


def _lrange(*args):
    # Ensure Python 3 compatibility for adding range() calls together
    r = range(*args)
    if not isinstance(r, list):
        r = list(r)
    return r


def round_to_zero(val):
    sign = np.sign(val)
    return np.floor(np.abs(val)) * sign


def iter_mod_dirs(mod_top_dir, path=None, check_if_exists=False):
    """
    Iterate over available .mod site directories.

    :param mod_top_dir: the root directory for the .mod files. Must include subdirectories that are the two-letter site
     abbreviations.
    :type mod_top_dir: str

    :param path: the subpath within each of the site dirs to return. If None, then the site directory itself is
     returned. Otherwise this is appended to the end and that returned.
    :type path: str

    :param check_if_exists: if ``True`` verify that the subpath exists before yielding. No effect if ``path`` is None.
    :type check_if_exists: bool

    :return: iterable of .mod file directories.
    :rtype: iterable
    """
    # only match two-character file names
    site_dirs = sorted(glob(os.path.join(mod_top_dir, '??')))
    for sdir in site_dirs:
        if path is None:
            yield sdir
        else:
            fullpath = os.path.join(sdir, path)
            if check_if_exists and not os.path.isdir(fullpath):
                raise ModPathError('Requested subpath "{}" does not exist in {}'.format(path, sdir))
            yield fullpath


def extract_mod_site_abbrevs(mod_files, default='xx'):
    """
    Extract the site abbreviations from .mod file paths

    The .mod files no longer include the site abbreviation anywhere in the file name or header, so we need to extract
    those from the directory paths. This function assumes paths of the format ".../abbr/<something>/*.mod", i.e. that
    there are are least three levels (including the .mod file) and that the third from the end is the site abbreviation.

    For each path, if (a) there are not three directory levels, (b) the third part from the end is not a two-letter
    name, or (c) if it is a dictionary, this returns the default abbreviation.

    :param mod_files: a single path to a .mod file, or a list of .mod files
    :type mod_files: str or list(str)

    :param default: the default site abbreviation to return if cannot find an abbreviation
    :type default: str

    :return: a single site abbreviation if ``mod_files`` is a string, a tuple of abbreviations otherwise.
    :rtype: str or tuple(str)
    """
    def extract_single(modf):
        if isinstance(modf, dict):
            return default
        elif not isinstance(modf, str):
            raise TypeError('Expected string or dictionary for each mod file')

        fileparts = modf.split(os.sep)
        if len(fileparts) < 3:
            return default
        elif not re.match(r'[a-zA-Z0-9]{2}', fileparts[-3]):
            return default
        else:
            return fileparts[-3]

    if isinstance(mod_files, str):
        return extract_single(mod_files)
    else:
        return tuple([extract_single(modf) for modf in mod_files])


def iter_mod_files(mod_top_dir, include_slant=False):
    """
    Iterate over available .mod files.

    :param mod_top_dir: the root directory for the .mod files. Must include subdirectories that are the two-letter site
     abbreviations.
    :type mod_top_dir: str

    :param include_slant: not yet implemented, but in the future will turn on returning pairs of files (vertical and
     slant).
    :type include_slant: bool

    :return: iterable of paths to .mod files. The paths will start with the given ``mod_top_dir``.
    """
    if include_slant:
        raise NotImplementedError('include_slant not yet implemented')

    for vdir in iter_mod_dirs(mod_top_dir, 'vertical'):
        vertical_files = sorted(glob(os.path.join(vdir, '*.mod')))
        for f in vertical_files:
            yield f


def calculate_model_potential_temperature(temp, pres_levels=_std_model_pres_levels):
    """
    Calculate potental temperature for model output on fixed pressure levels.

    :param temp: The absolute temperature (in K) on the model grid.
    :type temp: array-like

    :param pres_levels: the pressure levels that the temperature is defined on. Must be a 1D vector, i.e. all columns in
     the model must be on the same pressure levels. A standard set of pressure for GEOS FP is the default.
    :type pres_levels: vector-like

    :return: the potential temperature
    """
    if temp.ndim != 4:
        raise ValueError('temp expected to be 4D')

    ntime, nlev, nlat, nlon = temp.shape
    if nlev != pres_levels.size:
        raise ValueError('Number of levels in temp != number of pressure levels defined')
    pres = np.tile(pres_levels[np.newaxis, :, np.newaxis, np.newaxis], (ntime, 1, nlat, nlon))
    return calculate_potential_temperature(pres, temp)


def calculate_potential_temperature(pres, temp):
    """
    Calculate potential temperature.

    :param pres: Pressure in millibars/hPa.
    :param temp: Absolute temperature in Kelvin.
    :return: the potential temperature corresponding to those T/P coordinates.
    """
    return temp * (1000/pres) ** 0.286


def convert_geos_eta_coord(delp):
    """
    Calculate the pressure grid for a GEOS native file.

    :param pres: DELP (pressure thickness) array in Pa. May be any number of
     dimensions, as long as exactly one has a length of 72.
    :return: the pressure level midpoints, in hPa. Note the unit change, this
     is because the GEOS DELP variable is usually in Pa, but hPa is the standard
     unit for pressure levels in the Np files.
    """
    dp_shape = np.array(delp.shape)
    try:
        i_ax = np.flatnonzero(dp_shape == 72).item()
    except ValueError:
        raise ValueError('delp is either missing its 72 level dimension or has multiple dimensions with length 72')

    # From pg. 7 of the GEOS FP document (https://gmao.gsfc.nasa.gov/GMAO_products/documents/GEOS_5_FP_File_Specification_ON4v1_2.pdf)
    # the top pressure is always 0.01 hPa. Since the columns are space-to-surface, we add the cumulative sum to get the
    # level bottom pressure, then take the average along that axis to get the middle pressure
    level_shape = dp_shape.copy()
    level_shape[i_ax] = 1
    top_p = 0.01
    top_p_slice = np.full(level_shape, top_p)

    delp = delp * 0.01  # assume input is in Pa, want hPa
    p_edge = top_p + np.cumsum(delp, axis=i_ax)
    p_edge = np.concatenate([top_p_slice, p_edge], axis=i_ax)

    # Move the vertical axis to the front to do the averaging so we can just always average along the first dimension
    p_edge = np.rollaxis(p_edge, i_ax, 0)
    p_mid = 0.5 * (p_edge[:-1] + p_edge[1:])
    return np.rollaxis(p_mid, i_ax, 0)


def _construct_grid(*part_defs):
    """
    Helper function to construct coordinates for a 1D grid
    :param part_defs: a sequence of tuples (or lists) defining the start, stop, and step for each grid component. Each
     tuple gets expanded as the arguments to :func:`numpy.arange`.
    :return: the coordinates, sorted, and made sure to be unique
    """
    grid_components = [np.arange(*part) for part in part_defs]
    # Keep only unique values and sort them
    return np.unique(np.concatenate(grid_components))


# Compute area of each grid cell and the total area
def calculate_area(lat, lon, lat_res=None, lon_res=None, muted=False):
    """
    Calculate grid cell area for an equirectangular grid.

    :param lat: the vector of grid cell center latitudes, in degrees
    :param lon: the vector of grid cell center longitudes, in degrees
    :param lat_res: the width of a single grid cell in the latitudinal direction, in degrees. If omitted, will be
     calculated from the lat vector.
    :param lon_res: the width of a single grid cell in the longitudinal direction, in degrees. If omitted, will be
     calculated from the lat vector.
    :return: 2D array of areas, in units of fraction of Earth's surface area.
    """
    def calculate_resolution(coord_vec, coord_name):
        res = np.diff(coord_vec)
        if not np.all(res - res[0] < 0.001):
            raise RuntimeError('Could not determine a unique {} resolution'.format(coord_name))
        return res[0]

    nlat = lat.size
    nlon = lon.size

    if lat_res is None:
        lat_res = calculate_resolution(lat, 'lat')
    if lon_res is None:
        lon_res = calculate_resolution(lon, 'lon')

    lat_half_res = 0.5 * lat_res

    area = np.zeros([nlat, nlon])

    for j in range(nlat):
        Slat = lat[j]-lat_half_res
        Nlat = lat[j]+lat_half_res

        Slat = np.deg2rad(Slat)
        Nlat = np.deg2rad(Nlat)
        for i in range(nlon):
            area[j, i] = np.deg2rad(lon_res)*np.abs(np.sin(Slat)-np.sin(Nlat))

    if abs(np.sum(area) - 4*np.pi) > 0.0001: # ensure proper normalization so the total area of Earth is 4*pi
        #if not muted:
        #    print('Total earth area is {:g} not 4pi (difference of {:g}), normalizing to 4pi.'
        #          .format(np.sum(area), np.sum(area) - 4*np.pi))
        area *= 4*np.pi/np.sum(area)

    return area


def calculate_eq_lat_on_grid(EPV, PT, area):
    """
    Calculate equivalent latitude on a 4D grid.

    :param EPV: the potential vorticity on the 4D grid.
    :type EPV: :class:`numpy.ndarray`

    :param PT: the potential temperature on the 4D grid.
    :type PT: :class:`numpy.ndarray`

    :param area: the 2D grid of surface area (in steradians) that corresponds to the 2D slices of the 4D grid.
    :type area: :class:`numpy.ndarray`

    :return: equivalent latitude
    :rtype: :class:`numpy.ndarray`
    """
    EL = np.full_like(PT, np.nan)

    for itime in range(PT.shape[0]):
        interpolator = calculate_eq_lat(EPV[itime], PT[itime], area)
        # This is probably going to be horrifically slow - but interp2d sometimes gives weird results when called with
        # vectors, so unfortunately we have to call this one element at a time
        pbar = ProgressBar(PT[itime].size, prefix='Calculating eq. lat for time {}/{}:'.format(itime, PT.shape[0]),
                           style='counter')
        for i in range(PT[itime].size):
            pbar.print_bar(i)
            ilev, ilat, ilon = np.unravel_index(i, PT[itime].shape)
            this_pt = PT[itime, ilev, ilat, ilon]
            this_epv = EPV[itime, ilev, ilat, ilon]
            EL[itime, ilev, ilat, ilon] = interpolator(this_epv, this_pt)[0]

    return EL


def calculate_eq_lat(EPV, PT, area):
    """
    Construct an interpolator for equivalent latitude.

    :param EPV: a 3D grid of potential vorticity
    :type EPV: :class:`numpy.ndarray`

    :param PT: a 3D grid of potential temperature
    :type PT:  :class:`numpy.ndarray`

    :param area: the 2D grid of surface area (in steradians) that corresponds to the 2D slices of the 4D grid.
    :type area: :class:`numpy.ndarray`

    :return: a 2D interpolator for equivalent latitude, requires potential vorticity and potential temperature as inputs
    :rtype: :class:`scipy.interpolate.interp2d`

    Note: when querying the interpolator for equivalent latitude, it is often best to call it with scalar values, even
    though that is slower than calling it with the full vector of PV and PT that you wish to get EL for. The problem is
    that scipy 2D interpolators, when given vectors as input, return a grid. This would be fine, except that the values
    corresponding to the vector of PV and PT are not always along the diagonal and so cannot be extracted with
    :func:`numpy.diag`. (I suspect what is happening is that the interpolator sorts the input values when constructing
    the grid, but I have not tested this. -JLL)
    """
    nlev, nlat, nlon = PT.shape
    # Get rid of fill values, this fills the bottom of profiles with the first valid value
    PT[PT > 1e4] = np.nan
    EPV[EPV > 1e8] = np.nan
    for i in range(nlat):
        pd.DataFrame(PT[:, i, :]).fillna(method='bfill', axis=0, inplace=True)
        pd.DataFrame(EPV[:, i, :]).fillna(method='bfill', axis=0, inplace=True)

    # Define a fixed potential temperature grid, with increasing spacing
    # this is done arbitrarily to get sufficient levels for the interpolation to work well, and not too much for the
    # computations to take less time
    if np.min(PT) > 300 or np.max(PT) < 1000:
        raise ValueError('Potential temperature range is smaller than the [300, 1000] K assumed to create the '
                         'interpolation grid')

    theta_grid = _construct_grid((round_to_zero(np.nanmin(PT)), 300.0, 2), (300.0, 350.0, 5.0), (350.0, 500.0, 10.0),
                                 (500.0, 750.0, 20.0), (750.0, 1000.0, 30.0), (1000.0, round_to_zero(np.nanmax(PT)), 100.0))
    new_nlev = np.size(theta_grid)

    # Get PV on the fixed PT levels ~ 2 seconds per date
    new_EPV = np.zeros([new_nlev, nlat, nlon])
    for i in range(nlat):
        for j in range(nlon):
            new_EPV[:, i, j] = np.interp(theta_grid, PT[:, i, j], EPV[:, i, j])

    # Compute equivalent latitudes
    EL = np.zeros([new_nlev, 100])
    EPV_thresh = np.zeros([new_nlev, 100])
    for k in range(new_nlev): # loop over potential temperature levels
        maxPV = np.max(new_EPV[k]) # global max PV
        minPV = np.min(new_EPV[k]) # global min PV

        # define 100 PV values between the min and max PV
        EPV_thresh[k] = np.linspace(minPV,maxPV,100)

        for l,thresh in enumerate(EPV_thresh[k]):
            area_total = np.sum(area[new_EPV[k]>=thresh])
            x = 1-area_total/(2*np.pi)

            # With Python 3.10 and those dependencies, I started getting cases where x was *just* outside the -1 to 1 allowed domain,
            # which led to NaNs in the EqL and so bad things downstream. Since values were only slightly outside the domain, it's fine
            # to clip them, but if they go too far outside the expected values, then we may have a bigger problem.
            if x < -1.01 or x > 1.01:
                warn(f'Total area divided by 2*pi (x={x}) is far outside the domain of arcsin in EqL calculation. Clipping to -1 to 1.')
                
            if x < -1:
                x = -1
            elif x > 1:
                x = 1

            
            EL[k,l] = np.arcsin(x)*90.0*2/np.pi

    # Define a fixed potential vorticity grid, with increasing spacing away from 0
    # The last term should ensure that 0 is in the grid
    pv_grid = _construct_grid((round_to_zero(np.nanmin(EPV_thresh-50.0)), -1000.0, 50.0), (-1000.0, -500.0, 20.0), (-500.0, -100.0, 10.0),
                              (-100.0, -10.0, 1.0), (-10.0, -1.0, 0.1), (-1.0, 1.0, 0.01), (1.0, 10.0, 0.1),
                              (10.0, 100.0, 1.0), (100.0, 500.0, 10.0), (500.0, 1000.0, 20.0),
                              (1000.0, round_to_zero(np.nanmax(EPV_thresh))+50.0, 50.0), (0.0, 0.1))

    # Generate interpolating function to get EL for a given PV and PT
    interp_EL = np.zeros([new_nlev,len(pv_grid)])
    for k in range(new_nlev):
        interp_EL[k] = np.interp(pv_grid,EPV_thresh[k],EL[k])

    return interp2d(pv_grid, theta_grid, interp_EL)


def get_eqlat_profile(interpolator, epv, theta):
    el = np.full_like(epv, np.nan)
    for i, (pv, pt) in enumerate(zip(epv, theta)):
        el[i] = interpolator(pv, pt)

    return el


def calculate_eq_lat_field(EPV, PT, area):
    """
    Calculate 3D equivalent latitude from same shape potential vorticity and potiential temperature fields

    :param EPV: a 3D grid of Ertel's potential vorticity
    :type EPV: :class:`numpy.ndarray`

    :param PT: a 3D grid of potential temperature
    :type PT:  :class:`numpy.ndarray`

    :param area: the 2D grid of surface area (in steradians) that corresponds to the 2D slices of the 4D grid.
    :type area: :class:`numpy.ndarray`

    :param method: the interpolation method used by griddata, one of ['linear','nearest','cubic']
    :type method: str

    :return: the 3D equivalent latitude
    :rtype: :class:`numpy.ndarray`
    """
    # scipy/dask import here to avoid making it a requirement in setup.py as this is not the main/standard function
    import dask
    import dask.array as da
    from dask.diagnostics import ProgressBar as dask_progress
    from scipy.interpolate import griddata

    nlev, nlat, nlon = PT.shape
    # Get rid of fill values, this fills the bottom of profiles with the first valid value
    PT[PT > 1e4] = np.nan
    EPV[EPV > 1e8] = np.nan
    for i in range(nlat):
        pd.DataFrame(PT[:, i, :]).fillna(method='bfill', axis=0, inplace=True)
        pd.DataFrame(EPV[:, i, :]).fillna(method='bfill', axis=0, inplace=True)

    # Define a fixed potential temperature grid, with increasing spacing
    # this is done arbitrarily to get sufficient levels for the interpolation to work well, and not too much for the
    # computations to take less time
    if np.min(PT) > 300 or np.max(PT) < 1000:
        raise ValueError('Potential temperature range is smaller than the [300, 1000] K assumed to create the '
                         'interpolation grid')

    theta_array = _construct_grid((round_to_zero(np.nanmin(PT)), 300.0, 2), (300.0, 350.0, 5.0), (350.0, 500.0, 10.0),
                                 (500.0, 750.0, 20.0), (750.0, 1000.0, 30.0), (1000.0, round_to_zero(np.nanmax(PT)), 100.0))
    #theta_array = np.geomspace(round_to_zero(np.nanmin(PT)),round_to_zero(np.nanmax(PT)),130)
    new_nlev = np.size(theta_array)

    # Get PV on the fixed PT levels ~ 2 seconds per date
    new_EPV = np.zeros([new_nlev, nlat, nlon])
    for i in range(nlat):
        for j in range(nlon):
            new_EPV[:, i, j] = np.interp(theta_array, PT[:, i, j], EPV[:, i, j])

    # Compute equivalent latitudes
    EL = np.zeros([new_nlev, 100])
    EPV_thresh = np.zeros([new_nlev, 100])
    for k in range(new_nlev): # loop over potential temperature levels
        maxPV = np.max(new_EPV[k]) # global max PV
        minPV = np.min(new_EPV[k]) # global min PV

        # define 100 PV values between the min and max PV
        #EPV_thresh[k] = np.unique(np.concatenate([-1.0*np.geomspace(1,np.abs(minPV),50),[0],np.geomspace(1,maxPV,50)]))
        EPV_thresh[k] = np.linspace(minPV,maxPV,100)

        for l,thresh in enumerate(EPV_thresh[k]):
            area_total = np.sum(area[new_EPV[k]>=thresh])
            x = 1-area_total/(2*np.pi)

            # see comment in calculate_eq_lat
            if x < -1.01 or x > 1.01:
                warn(f'Total area divided by 2*pi (x={x}) is far outside the domain of arcsin in EqL calculation. Clipping to -1 to 1.')
                
            if x < -1:
                x = -1
            elif x > 1:
                x = 1
                      
            EL[k,l] = np.arcsin(x)*90.0*2/np.pi

    # pv_array has much less elements than in calculate_eq_lat in order to be used with griddata
    pv_array = np.unique(np.concatenate([
        -1*np.geomspace(1e3,-1*round_to_zero(np.nanmin(EPV_thresh)),100),
        _construct_grid((-1000.0, -500.0, 20.0), (-500.0, -100.0, 10.0),(-100.0, -10.0, 1.0), (-10.0, -1.0, 0.1), (-1.0, 1.0, 0.01), (1.0, 10.0, 0.1),
                        (10.0, 100.0, 1.0), (100.0, 500.0, 10.0), (500.0, 1000.0, 20.0), (0.0,0.1)),
        np.geomspace(1000,round_to_zero(np.nanmax(EPV_thresh)),100),
    ]))

    # Interpolate EL (equivalent latitude) on the fixed pv_array and theta_array
    interp_EL = np.zeros([new_nlev,len(pv_array)])
    for k in range(new_nlev):
        interp_EL[k] = np.interp(pv_array,EPV_thresh[k],EL[k])

    pv_grid, theta_grid = np.meshgrid(pv_array,theta_array)

    nonan = ~np.isnan(interp_EL) # there are no nans there, it's just to flatten the arrays
    flat_pv = pv_grid[nonan]
    flat_theta = theta_grid[nonan]
    flat_EL = interp_EL[nonan]

    # Use griddata to compute the equivalent latitude level by level
    # compute the different levels in parallel with dask
    final_EL = []
    for ilev in range(nlev):
        final_EL.append(dask.delayed(griddata)((flat_pv, flat_theta), flat_EL, (EPV[ilev], PT[ilev])))

    with dask_progress():
        final_EL = np.stack(dask.compute(*final_EL))

    return final_EL


def _format_geosfp_name(product, file_type, levels, date_time, add_subdir=False):
    """
    Create the file name for a GEOS FP or FP-IT file.

    :param product: which GEOS product ('fp' or 'fpit') to use

    :param file_type: which file type ('met' for meteorology, 'chm' for chemistry) to use
    :type file_type: str

    :param levels: which levels ('surf', 'p', or 'eta') to use
    :type levels: str

    :param date_time: the date and time of the desired file. The hour should be a multiple of 3.
    :type date_time: datetime-like

    :param add_subdir: if ``True``, then the correct subdirectory will be prepended.
    :type add_subdir: bool

    :return: the file name
    :rtype: str
    """
    if product in {'it'}:
        return _format_geosit_name(product, file_type, levels, date_time, add_subdir=add_subdir)
    
    product_patterns = {'fp': 'GEOS.fp.asm.inst3_{dim}d_{vars}_{type}.{date_time}.V01.nc4',
                        'fpit': 'GEOS.fpit.asm.inst3_{dim}d_{vars}_{type}.GEOS5124.{date_time}.V01.nc4'}
    level_mapping = {'surf': 'Nx', 'p': 'Np', 'eta': 'Nv'}
    level_dims = {'Np': 3, 'Nx': 2, 'Nv': 3}
    var_types = {'met': 'asm', 'chm': 'chm'}
    try:
        pattern = product_patterns[product]
    except KeyError:
        raise ValueError('product "{}" has not been defined. Allowed values are: {}'
                         .format(product, ', '.join(product_patterns.keys())))

    try:
        levels = level_mapping[levels]
    except KeyError:
        raise ValueError('levels "{}" not recognized. Allowed values are: {}'
                         .format(levels, ', '.join(level_mapping.keys())))

    try:
        dims = level_dims[levels]
    except KeyError:
        raise ValueError('file_type "{}" is not recognized. Allowed values are: {}'
                         .format(file_type, ', '.join(level_dims.keys())))

    date_time = date_time.strftime('%Y%m%d_%H%M')
    fname = pattern.format(dim=dims, type=levels, date_time=date_time, vars=var_types[file_type])
    if add_subdir:
        fname = os.path.join(levels, fname)

    return fname

def _format_geosit_name(product, file_type, levels, date_time, grid='L', add_subdir=False):
    product_patterns = {'it': 'GEOS.it.asm.{kind}_inst_{freq}hr_glo_{grid_key}_{lev}.GEOS5294.{dt:%Y-%m-%d}T{dt:%H%M}.V01.nc4'}
   
    level_descr = {'surf': 'slv', 'eta': 'v72', 'p': 'p42'}
    level_freq = {'surf': 1, 'eta': 3, 'p': 3}
    level_subdirs = {'surf': 'Nx', 'eta': 'Nv', 'p': 'Np'}
    grid_keys = {'L': 'L576x361', 'C': 'C180x180x6'}
    var_types = {'met': 'asm', 'chm': 'chm'}
    
    try:
        pattern = product_patterns[product]
    except KeyError:
        raise ValueError('product "{}" has not been defined. Allowed values are: {}'
                         .format(product, ', '.join(product_patterns.keys())))

    try:
        lev = level_descr[levels]
    except KeyError:
        raise ValueError('levels "{}" not recognized. Allowed values are: {}'
                         .format(levels, ', '.join(level_descr.keys())))

    try:
        freq = level_freq[levels]
    except KeyError:
        raise ValueError('level "{}" is not recognized. Allowed values are: {}'
                         .format(levels, ', '.join(level_freq.keys())))

    try:
        grid_key = grid_keys[grid]
    except KeyError:
        raise ValueError('grid "{}" is not recognized. Allowed values are: {}'
                         .format(grid, ', '.join(grid_keys)))

    try:
        var_type = var_types[file_type]
    except KeyError:
        raise ValueError('file_type "{}" is not recognized. Allowed values are: {}'
                         .format(file_type, ', '.join(var_types)))

    fname = pattern.format(kind=var_type, freq=freq, grid_key=grid_key, lev=lev, dt=date_time)
    if add_subdir:
        fname = os.path.join(level_subdirs[levels], fname)

    return fname


def read_geos_files(start_date, end_date, geos_path, profile_variables, surface_variables, product='fpit',
                    keep_time_dim=True, concatenate_arrays=False, set_mask_to_nan=False):
    """
    Read GEOS FP or FP-IT files between specified dates.

    :param start_date: the first date to read GEOS files from
    :type start_date: datetime-like

    :param end_date: the last date to read GEOS file from (exclusive). Note that if this datetime is not exactly a time
     that GEOS files are produced on, you will lose an extra file. For example, since GEOS files are produced every 3
     hours, if you specify ``end_date = datetime.datetime(2012, 1, 1, 23)``, you will lose the file from
     21:00 UTC 1 Jan 2012,
    :type end_date: datetime-like

    :param geos_path: the path where the GEOS files are stored. Must have subdirectories 'Np' and 'Nx' for the profile
     and surface files, respectively. Currently, each of these subdirectories must be flat, meaning that all GEOS data
     from all times is stored at the top level, not organized into further subdirectories by year/month etc.
    :type geos_path: str

    :param profile_variables: a list of variables to read from the profile (Np) files. 'lon', 'lat', and 'lev' are
     always read.
    :type profile_variables: list(str)

    :param surface_variables: a list of variables to read from the surface (Nx) files. 'lon' and 'lat' are always read.
    :type surface_variables: list(str)

    :param product: one of the strings 'fp' or 'fpit', determine which GEOS product is being read.
    :type product: str

    :param keep_time_dim: Set to ``True`` to keep the time dimension of the variables. This means the profile and
     surface variables will be 4D and 3D respectively. Set to ``False`` to remove it (so they will be 3D and 2D).
    :type keep_time_dim: bool

    :param concatenate_arrays: Set to ``True`` to concatenate the data from different files into a single array. This
     requires ``keep_time_dim`` to be ``True`` since they are concatenated along the time dimension. If ``False``, then
     the variables are left as lists of numpy arrays, where each array comes from a separate file.
    :type concatenate_arrays: bool

    :param set_mask_to_nan: Set to ``True`` turn the masked arrays read from netCDF files by default into regular numpy
     arrays, where the masked values are replaced with NaNs.
    :type set_mask_to_nan: bool

    :return: dictionaries of profile and surface variables, and a Pandas DateTimeIndex of the file dates. The variable
     dictionaries' values' format depends on the value of ``concatenate_arrays``.
    :rtype: dict, dict, DatetimeIndex
    """
    def read_var_helper(nchandle, varname, keep_time=keep_time_dim):
        if keep_time:
            data = nchandle.variables[varname][:]
        else:
            # This is equivalent to doing nchandle.variables[varname][0,:,:,:] for a 4D variable; omitting the trailing
            # colons makes it general for any size array.
            data = nchandle.variables[varname][0]

        if set_mask_to_nan:
            data = data.filled(np.nan)

        return data

    def read_files_helper(file_list, variables, is_profile):
        var_data = {v: [] for v in variables}
        for fidx, fname in enumerate(file_list):

            with ncdf.Dataset(fname, 'r') as nchandle:
                # Always read lat/lon. If reading a profile file, get the levels too for the vertical coordinate.
                lon = read_var_helper(nchandle, 'lon', keep_time=True)
                lat = read_var_helper(nchandle, 'lat', keep_time=True)
                if is_profile:
                    lev = read_var_helper(nchandle, 'lev', keep_time=True)

                # If on the first file, store the coordinate variables. If on a later file, double check that the
                # coordinates are the same. They should be, and that assumption makes the data easier to work with
                # since we don't have to recheck our indices for each file.
                if fidx == 0:
                    var_data['lon'] = lon
                    var_data['lat'] = lat
                    if is_profile:
                        var_data['lev'] = lev
                else:
                    chk = not ma.allclose(var_data['lon'], lon) or not ma.allclose(var_data['lat'], lat)
                    if is_profile:
                        chk = chk or not ma.allclose(var_data['lev'], lev)

                    if chk:
                        # TODO: replace this with proper GEOS error from the backend analysis
                        raise RuntimeError('lat, lon, and/or lev are inconsistent among the GEOS files')

                for var in variables:
                    var_data[var].append(read_var_helper(nchandle, var))

        if concatenate_arrays:
            for var, data in var_data.items():
                if isinstance(data, list):
                    # The lon/lat/lev variables don't need concatenated, we checked that they don't change with time
                    # so there's only one array for them, not a list.
                    var_data[var] = concatenate(data, axis=0)

        return var_data

    # input checking - we concatenate along the time dimension, so we better keep it
    if concatenate_arrays and not keep_time_dim:
        raise ValueError('concatenate_arrays = True requires keep_time_dim = True')

    # If we're converting the default masked arrays to regular arrays, we have to use np.concatenate because
    # ma.concatenate always returns a masked array. If we're not converting, then the reverse applied.
    if set_mask_to_nan:
        concatenate = np.concatenate
    else:
        concatenate = ma.concatenate

    geos_prof_files, file_dates = geosfp_file_names(product, 'met', 'p', start_date, end_date)
    geos_surf_files, surf_file_dates = geosfp_file_names(product, 'met', 'surf', start_date, end_date)

    # Check that the file lists have the same dates
    if len(file_dates) != len(surf_file_dates) or any(file_dates[i] != surf_file_dates[i] for i in range(len(file_dates))):
        raise RuntimeError('Somehow listed different profile and surface files')
    elif concatenate_arrays:
        file_dates = pd.DatetimeIndex(file_dates)

    geos_prof_files = [os.path.join(geos_path, 'Np', f) for f in geos_prof_files]
    geos_surf_files = [os.path.join(geos_path, 'Nx', f) for f in geos_surf_files]
    prof_data = read_files_helper(geos_prof_files, profile_variables, is_profile=True)
    surf_data = read_files_helper(geos_surf_files, surface_variables, is_profile=False)
    return prof_data, surf_data, file_dates


def geosfp_file_names(product, file_type, levels, start_date, end_date=None):
    """
    List all file names for GEOS FP or FP-IT files for the given date(s).

    :param product: which GEOS product ('fp' or 'fpit') to use
    :type product: str

    :param file_type: which file type ('met' for meteorology, 'chm' for chemistry) to use
    :type file_type: str

    :param levels: which levels ('surf', 'p', or 'eta') to use
    :type levels: str

    :param start_date: what date to start listing files for. If ``end_date`` is omitted, only the file for this date
     will be listed. Note that the hour must be a multiple of 3, since GEOS files are produced every three hours.
    :type start_date: datetime-like

    :param end_date: what date to stop list files. This is exclusive, and will not itself be included. Can be omitted
     to just list one file.
    :type end_date: None or datetime-like

    :return: the list of file names and an array of file dates
    :rtype: list, :class:`pandas.DatetimeIndex`
    """
    freq = pd.Timedelta(hours=3)
    if start_date.hour % 3 != 0:
        raise ValueError('The hour of start_date must be a multiple of 3')
    if end_date is None:
        end_date = start_date + freq

    geos_file_dates = pd.date_range(start=start_date, end=end_date - freq, freq=freq)
    geos_file_names = []
    for date in geos_file_dates:
        this_name = _format_geosfp_name(product, file_type, levels, date)
        geos_file_names.append(this_name)

    return geos_file_names, geos_file_dates


def geosfp_file_names_by_day(product, file_type, levels, utc_dates, utc_hours=None, add_subdir=False):
    """
    Create a list of GEOS-FP file names for specified dates

    This differs from :func:`geosfp_file_names` because this function can list files for only specific hours across
    multiple days. For example, if you want only 00:00 UTC FP-IT profile files for all of 2018, you would call this as::

        geosfp_file_names_by_day('fpit', 'Np', pd.date_range('2018-01-01', '2018-12-31', utc_hours=[0])

    :param product: which GEOS-FP product to make names for: "fp" or "fpit"
    :type product: str

    :param file_type: which file type ('met' for meteorology, 'chm' for chemistry) to use
    :type file_type: str

    :param levels: which levels ('surf', 'p', or 'eta') to use
    :type levels: str

    :param utc_dates: Dates (on UTC time) to read files for.
    :type utc_dates: collection(datetime) or collection(datetime-like objects)

    :param utc_hours: Which hours of the day to use (in UTC). If ``None``, then all hours that GEOS is produced on is
     used (every 3 hours). Otherwise, pass a collection of integers to specify a subset of hours to use. (e.g.
     [0, 3, 6, 9] to only use the files from the first half of each day).
    :type utc_hours: None or collection(int)

    :param add_subdir: if ``True``, then the correct subdirectory will be prepended.
    :type add_subdir: bool

    :return: a list of GEOS file names
    :rtype: list(str)
    """
    geos_utc_hours = np.arange(0, 24, 3)
    if utc_hours is not None:
        geos_utc_hours = geos_utc_hours[np.isin(geos_utc_hours, utc_hours)]

    geos_file_names = []
    geos_file_dates = []
    for date in utc_dates:
        for hr in geos_utc_hours:
            date_time = dt.datetime(date.year, date.month, date.day, hr)
            this_name = _format_geosfp_name(product, file_type, levels, date_time, add_subdir=add_subdir)
            geos_file_names.append(this_name)
            geos_file_dates.append(date_time)

    return geos_file_names, geos_file_dates


def datetime_from_geos_filename(geos_filename):
    geos_filename = os.path.basename(geos_filename)
    if geos_filename.startswith('GEOS.it'):
        date_str = re.search(r'\d{4}-\d{2}-\d{2}T\d{4}', geos_filename).group()
        return dt.datetime.strptime(date_str, '%Y-%m-%dT%H%M')
    else:
        date_str = re.search(r'\d{8}_\d{4}', geos_filename).group()
        return dt.datetime.strptime(date_str, '%Y%m%d_%H%M')


def is_geos_on_native_grid(geos_filename):
    with ncdf.Dataset(geos_filename, 'r') as nch:
        return nch['lev'].size == 72
    

def mode_to_product(mode):
    return mode.replace('-eta', '')


def mod_interpolation_legacy(z_grid, z_met, t_met, val_met, interp_mode=1, met_alt_geopotential=True):
    """
    Legacy interpolation for .mod file profiles onto the TCCON grid

    :param z_grid: the altitude levels (in kilometers) to interpolate the values onto
    :type z_grid: :class:`numpy.ndarray`

    :param z_met: the altitude levels (in kilometers) of the input values
    :type z_met: :class:`numpy.ndarray`

    :param t_met: the absolute temperature (in Kelvin) on the same levels as the input values
    :type t_met: :class:`numpy.ndarray`

    :param val_met: the input values to be interpolated to the ``z_grid`` levels
    :type val_met: :class:`numpy.ndarray`

    :param interp_mode: how to do the interpolation. Mode ``1`` (default) is used in the original GGG code for water
     vapor dry-air mole fraction. Recommended mode for anything relating to a concentration. Mode ``0`` is for
     temperature only. Mode ``2`` is for pressure, or more generally, values with an exponential dependence on altitude.
    :type interp_mode: int

    :param met_alt_geopotential: if ``True``, in met altitudes are assumed to be heights based on geopotential, not
     geometry. Therefore internally, the grid altitudes are slightly modified to be compatible. If ``False``, then the
     met altitudes are assumed to be geometric, and the grid altitudes will not be scaled.
    :type met_alt_geopotential: bool

    :return: the input values, interpolated onto the ``z_grid`` altitudes.
    :rtype: :class:`numpy.ndarray`
    """
    if met_alt_geopotential:
        z_grid = z_grid / (1 + z_grid/earth_radius)

    val_grid = np.full(z_grid.shape, np.nan, dtype=val_met.dtype)
    for i, z in enumerate(z_grid):
        # Find the levels in the met data above and below the current grid level. If we're below the first met level,
        # use the first two levels to extrapolate
        i_below = np.argwhere(z_met < z)
        if i_below.size == 0:
            i_below = 0
        else:
            i_below = np.max(i_below)

        i_above = i_below + 1

        # Calculate beta, which is used as the interpolation factor
        lapse_rate = (t_met[i_above] - t_met[i_below]) / (z_met[i_above] - z_met[i_below])
        if lapse_rate * (t_met[i_above] - t_met[i_below]) > 0.01 * t_met[i_below]:
            beta = np.log(1 + lapse_rate * (z - z_grid[i_below])/t_met[i_below]) / np.log(1 + lapse_rate*(z_met[i_below] - z_met[i_below]) / t_met[i_below])
        else:
            beta = (z - z_met[i_below]) / (z_met[i_above] - z_met[i_below])

        # Different interpolation modes that come from the GGG2014 fortran code.
        if interp_mode == 0:
            # interp_mode = 0 is for temperature only, uses lapse rate directly
            val_grid[i] = val_met[i_below] + lapse_rate * (z - z_met[i_below])
        elif interp_mode == 1:
            # interp_mode = 1 is for species concentrations
            val_grid[i] = val_met[i_below] + beta * (val_met[i_above] - val_met[i_below])
        elif interp_mode == 2:
            # interp_mode = 2 is for pressure
            val_grid[i] = val_met[i_below] * (val_met[i_above] / val_met[i_below])**beta
        else:
            raise ValueError('interp_mode = {} is not allowed. Must be 0, 1, or 2'.format(interp_mode))

    return val_grid


def mod_interpolation_new(z_grid, z_met, vals_met, interp_mode='linear'):
    """
    New method to interpolate met data onto the TCCON prior altitude grid

    :param z_grid: the altitude levels to interpolate to.
    :type z_grid: :class:`numpy.ndarray`

    :param z_met: the altitude levels in the meteorology data
    :type z_met: :class:`numpy.ndarray`

    :param vals_met: the values to be interpolated, on the ``z_met`` levels
    :type vals_met: :class:`numpy.ndarray`

    :param interp_mode: how to do the interpolation:

        * ``'linear'`` will do linear interpolation of ``vals_met`` with respect to z.
        * ``'lin-log'`` linearly interpolate ln(``vals_met``) with respect to z.
        * ``'log-lin'`` linearly interpolated ``vals_met`` with respect to ln(z).
        * ``'log-log'`` linearly interpolate ln(``vals_met``) with respect to ln(z).

     For compatibility with `mod_interpolation_legacy`, ``interp_mode`` may also be an integer. ``0`` or ``1`` are
     aliases for ``'linear'`` and ``2`` is the same as ``'lin-log'``
    :type interp_mode: str or int

    :return: the values interpolated to the TCCON grid.
    :rtype: :class:`numpy.ndarray`
    """
    interp_mode_compat_mapping = {0: 'linear', 1: 'linear', 2: 'lin-log', 3: 'log-lin', 4: 'log-log'}
    err_msg = 'interp_mode = {} is invalid. It must be one of the strings "{}" or one of the integers {}'.format(
        interp_mode, '", "'.join(interp_mode_compat_mapping.values()),
        ', '.join([str(k) for k in interp_mode_compat_mapping.keys()])
    )
    if isinstance(interp_mode, int):
        try:
            interp_mode = interp_mode_compat_mapping[interp_mode]
        except KeyError:
            raise ValueError(err_msg)
    elif interp_mode not in interp_mode_compat_mapping.values():
        raise ValueError(err_msg)

    interp_mode = interp_mode.lower()
    do_log_x = re.match('^log-\w{3}', interp_mode)
    do_log_y = re.match('\w{3}-log$', interp_mode)

    if do_log_x:
        z_grid = np.log(z_grid)
        z_met = np.log(z_met)
    if do_log_y:
        vals_met = np.log(vals_met)

    vals_interp = interp1d(z_met, vals_met, fill_value='extrapolate')
    vals_grid = vals_interp(z_grid)

    # If doing logarithmic interpolation for the y variable need to restore the output values. (x not returned so no
    # need to restore.)
    if do_log_y:
        vals_grid = np.exp(vals_grid)

    return vals_grid


def interp_tropopause_height_from_pressure(p_trop_met, p_met, z_met):
    """
    Calculate the tropopause height by interpolating to the tropopause pressure

    :param p_trop_met: the blended tropopause pressure from GEOS Nx files.
    :type p_trop_met: float

    :param p_met: the vector of pressure for this profile. Must be in the same units as ``p_trop_met``.
    :type p_met: array-like

    :param z_met: the vector of altitude levels for this profile.
    :type z_met: array-like

    :return: the tropopause altitude, in the same units as ``z_met``.
    :rtype: float
    """
    # The age-of-air calculation used for the tropospheric trace gas profile calculation needs the tropopause altitude.
    # Previously we'd tried finding this by interpolating to the tropopause potential temperature, in order to be
    # consistent about defining the strat/trop separation by potential temperature. However, potential temperature
    # does not always behave in a manner that makes interpolating to it straightforward (e.g. it crosses the tropopause
    # theta 0 or >1 times) so we just use pressure now.
    z_trop_met = mod_interpolation_new(p_trop_met, p_met, z_met, 'log-lin')
    if z_trop_met < np.nanmin(z_met):
        raise RuntimeError('Tropopause altitude calculated to be below the bottom of the profile. Something has '
                           'gone horribly wrong.')
    return z_trop_met


def calc_wmo_tropopause(temperature, altitude, limit_to=(5., 18.), raise_error=True):
    """
    Find the tropopause altitude using the WMO definition

    The WMO thermal definition of the tropopause is: "the level at which the lapse rate drops to < 2 K/km and the
    average lapse rate between this and all higher levels within 2 km does not exceed 2 K/km".
    (quoted in https://www.atmos-chem-phys.net/8/1483/2008/acp-8-1483-2008.pdf, sect. 2.4).

    :param temperature: the temperature profile, in K
    :type temperature: :class:`numpy.ndarray` (1D)

    :param altitude: the altitude for each level in the temperature profile, in kilometers
    :type altitude: :class:`numpy.ndarray` (1D)

    :param limit_to: the range of altitudes to limit the search for the tropopause to. This both helps avoid erroneous
     results and potentially speed up the analysis.
    :type limit_to: tuple(float, float)

    :param raise_error: If ``True``, this function raises an error if it cannot find the tropopause. If ``False``, it
     returns a NaN in that case.
    :type raise_error: bool

    :return: the tropopause altitude in kilometers
    :rtype: float
    :raises TropopauseError: if ``raise_error`` is ``True`` and this cannot find the tropopause.
    """

    # Calculate the lapse rate on the half levels. By definition, a positive lapse rate is a decrease with altitude, so
    # we need the minus sign.
    lapse = -np.diff(temperature) / np.diff(altitude)
    alt_half = altitude[:-1] + np.diff(altitude)/2.0

    # Cut down the data to just the relevant range of altitudes recommended by
    # https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2003GL018240 (end of sect. 2)
    zz = (alt_half >= np.min(limit_to)) & (alt_half <= np.max(limit_to))
    lapse = lapse[zz]
    alt_half = alt_half[zz]

    # Iterate over the levels. If the lapse rate is < 2 K/km, check that it remains there over the next 2 kilometers
    for k, (gamma, alt) in enumerate(zip(lapse, alt_half)):
        if gamma < 2.0:
            step = 0.1
            test_alt = np.arange(alt, alt+2.0+step, step)
            test_lapse = np.interp(test_alt, alt_half, lapse)
            if np.all(test_lapse < 2.0):
                # Interpolate to the exact tropopause altitude where the lapse rate first crosses the 2 K/km
                # np.interp requires the x-coordinates to be sorted, hence the complicated formula to get k_inds
                k_inds = np.argsort(lapse[[k-1, k]]) + k - 1
                return np.interp(2.0, lapse[k_inds], alt_half[k_inds])

    # If we get here, we failed to find the tropopause, so return a NaN or raise an error
    if raise_error:
        raise TropopauseError('Could not find a level meeting the WMO tropopause condition in the given profile')
    else:
        return np.nan


def number_density_air(p, t):
    """
    Calculate the ideal dry number density of air in molec. cm^-3

    :param p: pressure in hPa
    :type p: float or :class:`numpy.ndarray`

    :param t: temperature in K
    :type t: float or :class:`numpy.ndarray`

    :return: ideal dry number density in molec. cm^-3
    :rtype: float or :class:`numpy.ndarray`
    """
    R = const.gas_const  # gas constant in cm^3 * hPa / (mol * K)
    return p / (R*t) * const.avogadro


def effective_vertical_path(z, zmin, p=None, t=None, nair=None):
    """  
    Calculate the effective vertical path used by GFIT for a given z/P/T grid.

    :param z: altitudes of the vertical levels. May be any unit, but note that the effective paths will be returned in
     the same unit.
    :type z: array-like
    
    :param zmin: minimum altitude that the light ray reaches. This is given as ``zmin`` in the netCDF files and the .ray
     files. Must be in the same unit as ``z``.
    :type zmin: float

    :param p: pressures of the vertical levels. Must be in hPa.
    :type p: array-like

    :param t: temperatures of the vertical levels. Must be in K.
    :type t: array-like

    :return: effective vertical paths in the same units as ``z``
    :rtype: array-like
    """
    def integral(dz_in, lrp_in, sign):
        return dz_in * 0.5 * (1.0 + sign * lrp_in / 3 + lrp_in**2/12 + sign*lrp_in**3/60)

    if nair is not None:
        d = nair 
    elif p is not None and t is not None:
        d = number_density_air(p, t)
    else:
        raise TypeError('Either nair or p & t must be given')
    
    vpath = np.zeros_like(d)
    
    # From gfit/compute_vertical_paths.f, we need to find the first level above zmin
    # If there is no such level (which should not happen for TCCON), we treat the top
    # level this way
    try:
        klev = np.flatnonzero(z > zmin)[0]
    except IndexError:
        klev = np.size(z) - 1
        
    # from gfit/compute_vertical_paths.f, the calculation for level i is
    #   v_i = 0.5 * dz_{i+1} * (1 - l_{i+1}/3 + l_{i+1}**2/12 - l_{i+1}**3/60)
    #       + 0.5 * dz_i * (1 + l_i/3 + l_i**2/12 + l_i**3/60)
    # where
    #   dz_i = z_i - z_{i-1}
    #   l_i  = ln(d_{i-1}/d_i)
    # The top level has no i+1 term. This vector addition duplicates that calculation. The zeros padded to the beginning
    # and end of the difference vectors ensure that when there's no i+1 or i-1 term, it is given a value of 0.
    dz = np.concatenate([[0.0], np.diff(z[klev:]), [0.0]])
    log_rp = np.log(d[klev:-1] / d[klev+1:])
    log_rp = np.concatenate([[0.0], log_rp, [0.0]])
    
    # The indexing is complicated here, but with how dz and log_rp are constructed, this makes sure that, for vpath[klev],
    # the first integral(...) term uses dz = z[klev+1] - z[klev] and log_rp = ln(d[klev]/d[klev+1]) and the second integral
    # term is 0 (as vpath[klev] needs to account for the surface location below). For all other terms, this combines the
    # contributions from the weight above and below each level, with different integration signs to account for how the
    # weights increase from the level below to the current level and decrease from the current level to the level above.
    vpath[klev:] = integral(dz[1:], log_rp[1:], sign=-1) + integral(dz[:-1], log_rp[:-1], sign=1)
       
    # Now handle the surface - I don't fully understand how this is constructed mathematically, but the idea is that both
    # the levels in the prior above and below zmin need to contribute to the column, however that contribution needs to be
    # 0 below zmin. 
    
    dz = z[klev] - z[klev-1]
    xo = (zmin - z[klev-1])/dz
    log_rp = 0.0 if d[klev] <= 0 else np.log(d[klev-1]/d[klev])
    xl = log_rp * (1-xo)
    vpath[klev-1] += dz * (1-xo) * (1-xo-xl*(1+2*xo)/3 + (xl**2)*(1+3*xo)/12 + (xl**3)*(1+4*xo)/60)/2
    vpath[klev] += dz * (1-xo) * (1+xo+xl*(1+2*xo)/3 + (xl**2)*(1+3*xo)/12 - (xl**3)*(1+4*xo)/60)/2

    return vpath


def get_ussa_for_alts(alts):
    """
    Get temperature and pressure from the US standard atmosphere (USSA) for given altitudes.

    Temperature is interpolated to the requested altitudes linearly, assuming that the lapse rate is constant between
    the levels defined by the USSA. Pressure is interpolated exponentially, i.e. ln(p) is interpolated linearly.

    :param alts: altitudes, in kilometers, to calculate T and P for.
    :type alts: float or :class:`numpy.ndarray`

    :return: temperatures (in K) and pressures (in hPa). Arrays will be the same shape as the input altitudes.
    :rtype: float or :class:`numpy.ndarray`, float or :class:`numpy.ndarray`
    """
    # Need to interpolate pressure and temperature to the given altitudes. Will assume that temperature varies linearly
    # with altitude and pressure varies exponentially. Since p = p0 * exp(-z/H) then ln(p) = ln(p0) - z/H, therefore
    # we will linearly interpolate ln(p) w.r.t. altitude.
    z_coord = const.z_ussa
    t_coord = const.t_ussa
    p_coord = np.log(const.p_ussa)

    interp_args = {'left': np.nan, 'right': np.nan}
    t = np.interp(alts, z_coord, t_coord, **interp_args)
    p = np.exp(np.interp(alts, z_coord, p_coord, **interp_args))

    return t, p


def get_ussa_for_pres(pres):
    """
    Get altitude and temperature from the US standard atmosphere (USSA) for given pressures.

    Temperature is interpolated to the requested altitudes linearly, assuming that the lapse rate is constant between
    the levels defined by the USSA. Pressure is interpolated exponentially, i.e. ln(p) is interpolated linearly.

    :param pres: pressures, in hPa, to calculate z and T for.
    :type pres: float or :class:`numpy.ndarray`

    :return: temperatures (in K) and altitudes (in km). Arrays will be the same shape as the input altitudes.
    :rtype: float or :class:`numpy.ndarray`, float or :class:`numpy.ndarray`
    """

    # Since temperature varies linearly with altitude and altitude varies linearly vs. ln(p), interpolate both by the
    # log of pressure
    z_coord = np.flipud(const.z_ussa)
    t_coord = np.flipud(const.t_ussa)
    # must flip - np.interp expects its x-coordinates to be increasing.
    p_coord = np.flipud(np.log(const.p_ussa))

    pres = np.log(pres)

    interp_args = {'left': np.nan, 'right': np.nan}
    t = np.interp(pres, p_coord, t_coord, **interp_args)
    z = np.interp(pres, p_coord, z_coord, **interp_args)

    return t, z


def age_of_air(lat, z, ztrop, ref_lat=45.0):
    """
    Calculate age of air using a function form from GGG 2014.

    :param lat: the latitude(s) to calculate age of air for
    :type lat: float or :class:`numpy.ndarray`

    :param z: the altitude(s) to calculate age of air for. If both ``z`` and ``lat`` given as a vectors, they must be
     the same shape. Must have units of kilometers.
    :type z: float or :class:`numpy.ndarray`

    :param ztrop: the tropopause altitude, in kilometers.
    :type ztrop: float

    :param ref_lat: the reference latitude for the cycle. This is where the exponential in latitude is maximized. 45N
     was chosen as the default as the center of the northern hemisphere, where most anthropogenic emissions are.
    :type ref_lat: float

    :return: age of air, in years, as a numpy array
    :rtype: :class:`numpy.ndarray`
    """
    # Force z to be a numpy array. This allows us to use numpy indexing for the extra (stratospheric) term below and
    # simultaneously ensures aoa is always a numpy array.
    if not isinstance(z, np.ndarray):
        z = np.array([z])

    fl = lat/22.0
    aoa = 0.313 - 0.085 * np.exp(-((lat-ref_lat)/18)**2) - 0.268*np.exp(-1.42 * z / (z+ztrop)) * fl / np.sqrt(1+fl**2)

    # We limit the calculation to z > ztrop here because that avoids a divide-by-0 warning
    # This term is really only kept in for completeness; in practice, it should never be used because we don't use
    # this term in the stratosphere.
    extra_term = 7.0 * (z[z > ztrop]-ztrop)/z[z > ztrop]
    aoa[z > ztrop] += extra_term
    return aoa


def seasonal_cycle_factor(lat, z, ztrop, fyr, species, ref_lat=45.0):
    """
    Calculate a factor to multiply a concentration by to account for the seasonal cycle.

    :param lat: the latitude(s) to calculate age of air for
    :type lat: float or :class:`numpy.ndarray`

    :param z: the altitude(s) to calculate age of air for. If both ``z`` and ``lat`` given as a vectors, they must be
     the same shape. Must have units of kilometers.
    :type z: float or :class:`numpy.ndarray`

    :param ztrop: the tropopause altitude, in kilometers.
    :type ztrop: float

    :param fyr: the fraction of the year that corresponds to this date. You can convert a date time to this value with
     :func:`date_to_frac_year`.
    :type fyr: float

    :param species: a child class of :class:`~tccon_priors.TraceGasTropicsRecord` that defines a gas name and seasonal
     cycle coefficient. May be an instance of the class or the class itself. If the gas name is "co2", then a
     CO2-specific parameterization is used.
    :type species: :class:`~tccon_priors.TraceGasTropicsRecord`

    :param ref_lat: reference latitude for the age of air. Set to 45N as an approximation of where the NH emissions are.
    :type ref_lat: float

    :return: the seasonal cycle factor as a numpy array. Multiply this by a deseasonalized concentration at (lat, z) to
     get the concentration including the seasonal cycle
    """
    if species.gas_seas_cyc_coeff is None:
        raise TypeError('The species record ({}) does not define a seasonal cycle coefficient')

    aoa = age_of_air(lat, z, ztrop, ref_lat=ref_lat)
    if species.gas_name.lower() == 'co2':
        sv = np.sin(2*np.pi *(fyr - 0.834 - aoa))
        svnl = sv + 1.80 * np.exp(-((lat - 74)/41)**2)*(0.5 - sv**2)
        sca = svnl * np.exp(-aoa/0.20)*(1 + 1.33*np.exp(-((lat-76)/48)**2) * (z+6)/(z+1.4))
    else:
        sv = np.sin(2*np.pi * (fyr - 0.78))  # basic seasonal variation
        svl = sv * (lat / 15.0) / np.sqrt(1 + (lat / 15.0)**2.0)  # latitude dependence
        sca = svl * np.exp(-aoa / 0.85)  # altitude dependence

    return 1 + sca * species.gas_seas_cyc_coeff


def hf_ch4_slope_fit(yrs, a, b, c, t0):
    """
    A fitting function appropriate to fit the trend of CH4 vs. HF slopes

    This function has the form:

    ..math::
        a * exp(b*(t - t0)) + c

    where t is given in years.

    :param yrs: t in the above equation.
    :type yrs: :class:`numpy.ndarray`

    :param a, b, c, t0: the fitting parameters in the above equation
    :type a, b, c, t0: float

    :return: the predicted slopes at ``yrs``
    """
    return a * np.exp(b*(yrs - t0)) + c


# from https://stackoverflow.com/a/16562028
def isoutlier(data, m=2):
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else np.zeros_like(data)
    
    return s >= m


def is_tropics(lat, doy, ages):
    return np.abs(lat) < 20.0


def is_vortex(lat, doy, ages):
    if not isinstance(doy, np.ndarray):
        doy = np.full_like(lat, doy)
    xx_vortex = np.zeros_like(lat, dtype=np.bool_)
    xx_vortex[(doy > 140) & (doy < 245) & (lat < -55.0) & (ages > 3.25)] = True
    xx_vortex[(doy > 275) & (doy < 60) & (lat > 55.0) & (ages > 3.25)] = True
    return xx_vortex


def is_midlat(lat, doy, ages):
    return ~is_tropics(lat, doy, ages) & ~is_vortex(lat, doy, ages)


def is_overworld(potential_temp, pressure, trop_pres):
    return (potential_temp >= 380) & (pressure <= trop_pres)


def npdate_to_datetime(numpy_date):
    numpy_date = numpy_date.astype('datetime64[s]')
    ts = (numpy_date - np.datetime64('1970-01-01T00:00:00Z', 's')) / np.timedelta64(1, 's')
    return dt.datetime.utcfromtimestamp(ts)


def date_to_decimal_year(date_in):
    """
    Convert a datetime object to a decimal year.

    A decimal year is e.g. 2018.5, where the part before the decimal is the year itself and after the decimal is how
    much of the year has passed.

    :param date_in: the datetime object to convert
    :type date_in: datetime-like

    :return: the decimal year
    :rtype: float
    """
    if date_in is not None:
        return date_in.year + date_to_frac_year(date_in)
    else:
        return np.nan


def day_of_year(date_in):
    return clams_day_of_year(date_in) - 1


def clams_day_of_year(date_in):
    return float(date_in.strftime('%j'))


def date_to_frac_year(date_in):
    """
    Convert a datetime object to a fraction of a year.

    The fraction is essentially how much of the year has passed, so June 1st (of any year) becomes ~0.416.

    Note: this function assumes 365.25 days per year. This is an imperfect solution, ideally a fractional year should
    really describe the fraction of an orbit the Earth has completed, so only use this for things where +/- a day is an
    insignificant error.

    :param date_in: the datetime object to convert
    :type date_in: datetime-like

    :return: the fractional year
    :rtype: float
    """
    return day_of_year(date_in) / days_per_year  # since there's about and extra quarter of a day per year that gives us leap years


def frac_year_to_doy(yr_in):
    """
    Convert a fractional year to a day of year.

    Internally, this multiplies by 365.25, so see the warning in the docstring for :func:`date_to_frac_year` about
    its precision. Crucially, this is NOT a reliable inverse operation to :func:`date_to_frac_year`.

    :param yr_in: the fractional year to convert
    :type yr_in: float

    :return: the number of days since 1 Jan
    :rtype: float
    """
    return yr_in * days_per_year


def frac_years_to_reldelta(frac_year, allow_nans=True):
    """
    Convert a fractional year to a :class:`relativedelta` from dateutils.

    Note: like the other fraction/decimal year functions, this assumes 365.25 days/year internally. Therefore, this
    should function correctly as an inverse operation to date_to_frac_year when added back to Jan 1 of the year in
    question.

    :param frac_year: the fractional year(s) (e.g 2.5 for 2 and a half years) to convert
    :type frac_year: float or a collection of floats

    :param allow_nans: whether to permit NaNs in the decimal years. If ``True``, then NaNs will be retained in the
     output list. If ``False``, an error is raised if NaNs are found in ``dec_year``.
    :type allow_nans: bool

    :return: a list of dateutils :class:`relativedelta` objects or a single :class:`relativedelta` if a scalar
     ``frac_year`` was given.
    :rtype: :class:`relativedelta` or list(:class:`relativedelta`)
    """
    if isinstance(frac_year, float):
        return_scalar = True
        frac_year = [frac_year]
    else:
        return_scalar = False
    if not allow_nans and np.any(np.isnan(frac_year)):
        raise ValueError('NaNs not permitted in frac_year. Either remove them, or set `allow_nans=True`')
    age_years = np.floor(frac_year)
    age_fracs = np.mod(frac_year, 1)
    rdels = [relativedelta(years=y, days=days_per_year * d) if not (np.isnan(y) or np.isnan(d)) else np.nan for y, d in zip(age_years, age_fracs)]
    if return_scalar:
        rdels = rdels[0]

    return rdels


def timedelta_to_frac_year(timedelta):
    """
    Convert a concrete timedelta to fractional years

    :param timedelta: the timedelta to convert
    :type timedelta: :class:`datetime.timedelta`

    :return: the time delta as a fraction of years, assuming 365.25 days per year.
    :rtype: float
    """

    return timedelta.total_seconds() / (days_per_year * 24 * 3600)


def decimal_year_to_date(dec_year, date_type=dt.datetime):
    """
    Convert decimal year or years (e.g. 2018.5) to a datetime-like object

    :param dec_year: the decimal year or years to convert. May be any kind of collection (if passing multiple values) so
     long as it supports iteration and that iteration returns scalar values (i.e. a 2D numpy array will not work because
     iteration returns rows).

    :param date_type: what type to convert the decimal years in to. May be any time that can be called
     ``date_type(year, month, day)``.

    :return: the converted dates in the type ``date_type``. If a single decimal date was passed in, then a single value
     of type ``date_type`` is returned. If a collection was passed in, then the dates will be returned in a 1D numpy
     array.
    """

    if np.any(np.isnan(dec_year)):
        raise NotImplementedError('NaNs in the input decimal years not implemented')

    try:
        dec_year[0]
    except (TypeError, IndexError):
        dec_year = [dec_year]
        return_as_scalar = True
    else:
        return_as_scalar = False

    years = np.array([int(d) for d in dec_year])
    frac_yrs = np.array([d % 1 for d in dec_year])
    dates = np.array([date_type(y, 1, 1) + frac_years_to_reldelta(fr, allow_nans=False) for y, fr in zip(years, frac_yrs)])

    if return_as_scalar:
        return dates[0]
    else:
        return dates


def start_of_month(date_in, out_type=dt.date):
    """
    Get a date-like object corresponding to the beginning of the month of ``date_in``

    :param date_in: Any date-like object that has attributes ``year`` and ``month``
    :param out_type: A type whose constructor accepts the keyword arguments ``year``, ``month``, and ``day``.
    :return: an instance of ``out_type`` set to day 1, 00:00:00 of the month of ``date_in``.
    """
    return out_type(year=date_in.year, month=date_in.month, day=1)


def relativedelta2string(rdelta):
    parts = ('years', 'months', 'days', 'hours', 'minutes', 'seconds')
    time_parts = []
    for p in parts:
        tp = getattr(rdelta, p)
        if tp > 0:
            time_parts.append('{} {}'.format(tp, p))

    return ', '.join(time_parts)


def gravity(gdlat,altit):
    """
    copy/pasted from fortran routine comments
    This is used to convert

    Input Parameters:
        gdlat       GeoDetric Latitude (degrees)
        altit       Geometric Altitude (km)

    Output Parameter:
        gravity     Effective Gravitational Acceleration (m/s2)
        radius 		Radius of earth at gdlat

    Computes the effective Earth gravity at a given latitude and altitude.
    This is the sum of the gravitational and centripital accelerations.
    These are based on equation I.2.4-(17) in US Standard Atmosphere 1962
    The Earth is assumed to be an oblate ellipsoid, with a ratio of the
    major to minor axes = sqrt(1+con) where con=.006738
    This eccentricity makes the Earth's gravititational field smaller at
    the poles and larger at the equator than if the Earth were a sphere
    of the same mass. [At the equator, more of the mass is directly
    below, whereas at the poles more is off to the sides). This effect
    also makes the local mid-latitude gravity field not point towards
    the center of mass.

    The equation used in this subroutine agrees with the International
    Gravitational Formula of 1967 (Helmert's equation) within 0.005%.

    Interestingly, since the centripital effect of the Earth's rotation
    (-ve at equator, 0 at poles) has almost the opposite shape to the
    second order gravitational field (+ve at equator, -ve at poles),
    their sum is almost constant so that the surface gravity could be
    approximated (.07%) by the simple expression g=0.99746*GM/radius^2,
    the latitude variation coming entirely from the variation of surface
    r with latitude. This simple equation is not used in this subroutine.
    """

    d2r=3.14159265/180.0	# Conversion from degrees to radians
    gm=3.9862216e+14  		# Gravitational constant times Earth's Mass (m3/s2)
    omega=7.292116E-05		# Earth's angular rotational velocity (radians/s)
    con=0.006738       		# (a/b)**2-1 where a & b are equatorial & polar radii
    shc=1.6235e-03  		# 2nd harmonic coefficient of Earth's gravity field
    eqrad=6378178.0   		# Equatorial Radius (meters)

    gclat=arctan(tan(d2r*gdlat)/(1.0+con))  # radians

    radius=1000.0*altit+eqrad/np.sqrt(1.0+con*sin(gclat)**2)
    ff=(radius/eqrad)**2
    hh=radius*omega**2
    ge=gm/eqrad**2                      # = gravity at Re

    gravity=(ge*(1-shc*(3.0*sin(gclat)**2-1)/ff)/ff-hh*cos(gclat)**2)*(1+0.5*(sin(gclat)*cos(gclat)*(hh/ge+2.0*shc/ff**2))**2)

    return gravity, radius


def geopotential_height_to_altitude(gph, lat, alt):
    """
    Convert a geopotential height in m^2 s^-2 to meters

    :param gph: geopotential height in m^2 s^-2.
    :type gph: float

    :param lat: geographic latitude (in degrees, south is negative)
    :type lat: float

    :param alt: altitude of the TCCON site in kilometers. If set to 0, will use gravity at the surface for the given
     latitude.
    :type alt: float

    :return: the geopotential height converted to meters
    """
    gravity_at_site, _ = gravity(lat, alt)
    return gph / gravity_at_site


def to_unix_time(datetime):
    """
    Convert a datetime-like object into Unix time (seconds since midnight, 1 Jan 1970)

    :param datetime: the datetime to convert. May be any type that can have a :class:`datetime.datetime` object
     subtracted from it, and for which the subtraction has a method `total_seconds` that returns the time delta as
     a number of seconds. Both :class:`datetime.datetime` and :class:`pandas.Timestamp` are examples.

    :return: unix time
    :rtype: float
    """
    return (datetime - dt.datetime(1970, 1, 1)).total_seconds()


def from_unix_time(utime, out_type=dt.datetime):
    """
    Convert a unix time into a datetime object.

    :param utime: the unix time (seconds since midnight, 1 Jan 1970)
    :type utime: float

    :param out_type: optional, a type that represents a datetime which has an init method such that
     ``out_type(year, month, day)`` returns a object representing that time and can be added with a
     :class:`datetime.timedelta``.
    :type out_type: type

    :return: a datetime object of the type specified by ``out_type``.
    """
    return out_type(1970, 1, 1) + dt.timedelta(seconds=utime)


def mod_file_name(prefix,date,time_step,site_lat,site_lon_180,ew,ns,mod_path,round_latlon=True,in_utc=True):

    YYYYMMDD = date.strftime('%Y%m%d')
    HH = date.strftime('%H')
    if in_utc:
        HH += 'Z'
    else:
        HH += 'L'

    if round_latlon:
        site_lat = round(abs(site_lat))
        site_lon = round(abs(site_lon_180))
        latlon_precision = 0
    else:
        site_lat = abs(site_lat)
        site_lon = abs(site_lon_180)
        latlon_precision = 2
    if time_step < timedelta(days=1):
        mod_fmt = '{{prefix}}_{{ymd}}{{hh}}_{{lat:0>2.{prec}f}}{{ns:>1}}_{{lon:0>3.{prec}f}}{{ew:>1}}.mod'.format(prec=latlon_precision)
    else:
        mod_fmt = '{{prefix}}_{{ymd}}_{{lat:0>2.{prec}f}}{{ns:>1}}_{{lon:0>3.{prec}f}}{{ew:>1}}.mod'.format(prec=latlon_precision)

    mod_name = mod_fmt.format(prefix=prefix, ymd=YYYYMMDD, hh=HH, lat=site_lat, ns=ns, lon=site_lon, ew=ew)
    return mod_name


def mod_file_name_for_priors(datetime, site_lat, site_lon_180, prefix='FPIT', **kwargs):
    if site_lon_180 > 180:
        site_lon_180 -= 360
    ew = format_lon(site_lon_180)[-1]
    ns = format_lat(site_lat)[-1]
    return mod_file_name(prefix=prefix, date=datetime, time_step=dt.timedelta(hours=3), site_lat=site_lat,
                         site_lon_180=site_lon_180, ew=ew, ns=ns, mod_path='', **kwargs)


def parse_date_range(datestr):
    def parse_date(datestr):
        try:
            date_out = dt.datetime.strptime(datestr, '%Y%m%d')
        except ValueError:
            date_out = dt.datetime.strptime(datestr, '%Y%m%d_%H')
        return date_out

    dates = datestr.split('-')
    start_date = parse_date(dates[0])
    if len(dates) > 1:
        end_date = parse_date(dates[1])
    else:
        end_date = start_date + timedelta(days=1)

    return start_date, end_date


def get_ggg_path(subdir, subdir_name):
    gggpath = os.getenv('GGGPATH')
    if gggpath is None:
        raise GGGPathError('Could not find the GGGPATH environmental variable. Please specify an explicit {}.'.format(subdir_name))
    full_subdir = os.path.join(gggpath, subdir)
    if not os.path.isdir(full_subdir):
        raise GGGPathError('Could not find default {} {}'.format(subdir_name, full_subdir))

    return full_subdir


def check_site_lat_lon_alt(abbrev, lat=None, lon=None, alt=None):
    """
    Verify that the input site abbreviation, latitude, longitude, and altitude are consistent.

    This checks that (1) if any of lat/lon/alt are given, all are, (2) if any inputs are > 1 long all those that are
    are the same length, (3) if any inputs are scalar (None, int, float, or string) they are converted to tuples
    of the proper length, and (4) all longitudes are given in [0, 360)

    :param abbrev: site abbreviation(s), as string or iterable of such
    :type abbrev: str or list(str)

    :param lat: site latitude(s), as float or iterable of such, or None
    :type lat: float or list(float)

    :param lon: site longitude(s), as float or iterable of such, or None
    :type lon: float or list(float)

    :param alt: site altitude(s), as float or iterable of such, or None
    :type alt: float or list(float)

    :return: abbrev, lat, lon, and "alt" converted to tuples of the same length. Scalar values are repeated.
    :rtype: tuples
    """
    def to_tuple(val):
        if isinstance(val, (int, float, str, type(None))):
            val = (val,)
        return val, len(val)

    are_none = [x is None for x in (lat, lon, alt)]
    abb_tuple, _ = to_tuple(abbrev)
    if all(are_none):
        if len(abb_tuple) == 1 and abb_tuple[0] == 'all':
            pass
        elif any(abb not in site_dict for abb in abb_tuple):
            raise ValueError('One or more site abbreviation not a predefined TCCON site, must give lat, lon, alt')
    elif any(are_none):
        if len(abb_tuple) == 1 and abb_tuple[0] == 'all':
            raise ValueError('Running all sites is incompatible with giving custom lat/lons')
        else:
            raise ValueError('If any of lat, lon, or alt are given, all must be given')

    vals = dict()
    n = dict()
    vals['abbrev'], n['abbrev'] = to_tuple(abbrev)
    vals['lat'], n['lat'] = to_tuple(lat)
    vals['lon'], n['lon'] = to_tuple(lon)
    vals['alt'], n['alt'] = to_tuple(alt)

    max_n = max(v for v in n.values())
    for k in vals.keys():
        if n[k] == 1:
            vals[k] *= max_n
        elif n[k] != max_n:
            raise ValueError('{} is not a single value or the same length as the longest input'.format(k))

    # Constrain longitude between 0 and 360
    vals['lon'] = tuple([l + 360 if l is not None and l < 0 else l for l in vals['lon']])
    if any([l < 0 or l >= 360 for l in vals['lon'] if l is not None]):
        raise ValueError('Longitude outside of valid range (-360, +360, exclusive) found')

    return vals['abbrev'], vals['lat'], vals['lon'], vals['alt']


def dry2wet(dmf, h2o_dmf):
    return dmf / (1 + h2o_dmf)


def interp_to_zgrid(profile_dict, zgrid, gas_extrap_method='linear'):
    """
    Interpolate the output profile variables to a desired altitude grid.

    This mimics the behavior in `read_refvmrs` of the Fortran gsetup code. It interpolates the output variables linearly
    with respect to altitude. It will extrapolate linearly towards the surface, but will raise an error if the profile
    does not go high enough.

    :param profile_dict: a dictionary of profile variables. Must include the altitude levels as the key "Height". The
     keys "gas_record_dates" and "gas_date" are treated as dates for interpolation.
    :type profile_dict: dict

    :param zgrid: any specification of a fixed altitude grid accepted by :func:`_setup_zgrid`, i.e. ``None`` to do no
     interpolation, a path to an integral file, or an array directly specifying the altitude grid.
    :type zgrid: None, str, or :class:`numpy.ndarray`.

    :return: the profile dictionary with its variables interpolated/extrapolated to the new altitude grid. It will
     also have been modified in-place.
    :rtype: dict
    """
    # Specify pairs of functions to convert and unconvert the values of the input arrays. Each key must match a key in
    # the profile_dict, and the values must be two element tuples, where the first element is the function to use to
    # convert the values in the dict to ones that can be interpolated and the second does the reverse process.
    met_variables = ('Height', 'Temp', 'Pressure', 'PT', 'EqL')
    dt_converters = (_datetime2float, _float2datetime)
    converters = {'gas_record_dates': dt_converters,
                  'gas_date': dt_converters}

    zgrid = _setup_zgrid(zgrid)
    if zgrid is None:
        return profile_dict

    profile_z = profile_dict['Height'].copy()
    for k, v in profile_dict.items():
        if k in converters:
            v = converters[k][0](v)

        v = xr.DataArray(v, coords=[profile_z], dims=['z'])
        if gas_extrap_method == 'linear' or k in met_variables:
            # we always want to linearly extrapolate the met variables, especially since they often act as vertical
            # coordinates. If more met variables are added, you'll have to update the list.
            fills = 'extrapolate'
        elif gas_extrap_method in ('const', 'constant'):
            fills = (v[0], v[-1])
        else:
            raise ValueError('extrap_method must be one of: "linear", "const", or "constant".')
        v = v.interp(z=zgrid, method='linear', kwargs={'fill_value': fills})

        if k in converters:
            v = converters[k][1](v.data)
        else:
            v = v.data
        profile_dict[k] = v

    return profile_dict


def _setup_zgrid(zgrid):
    """
    Setup a fixed altitude grid
    :param zgrid:
    :return:
    """
    if isinstance(zgrid, str):
        int_info = _read_integral_file(zgrid)
        return int_info['Height']
    elif not isinstance(zgrid, (type(None), np.ndarray, xr.DataArray)):
        raise TypeError('zgrid must be a string, None, numpy array, or xarray DataArray')
    else:
        return zgrid


def _read_integral_file(integral_file, as_dataframes=False):
    """
    Read an integral file that defines an altitude grid for GGG

    :param integral_file: the path to the integral file
    :type integral_file: str

    :param as_dataframes: if ``True``, the information in the file is returned as a single dataframe. If ``False``, it
     is returned as a dict of numpy arrays.
    :type as_dataframes: bool

    :return: the table of altitudes and mean molecular weights.
    :rtype: :class:`pandas.DataFrame` or dict
    """
    df = pd.read_csv(integral_file, sep=r'\s+', header=None, names=['Height', 'mmw'])
    if as_dataframes:
        return df
    else:
        return {k: v.to_numpy() for k, v in df.items()}


def _datetime2float(dtarray):
    return np.array([np.nan if d is None else to_unix_time(d) for d in dtarray])


def _float2datetime(dtarray):
    return np.array([None if np.isnan(d) else from_unix_time(d, pd.Timestamp) for d in dtarray])


def frac_ydh_to_date(year, day, hour):
    """
    Convert a fractional year, day, and hour from a GGG output file to a date

    Some GGG output files record year as :math:`year + (day + hour/24)/366`
    and day as :math:`day + hour/24`. This function converts such values to
    a Pandas DatetimeIndex.

    :param year: array of years, as decimal numbers
    :type year: array(float)

    :param day: array of days-of-year, with Jan 1 = 1, as decimal numbers.
    :type day: array(float)

    :param hour: array of hours of day.
    :type hour: array(float)

    :return: array of datetimes
    :rtype: :class:`pandas.DatetimeIndex`
    """
    # In many output files, the year and day are calculated to represent the 
    # whole date, i.e. year is a decimal number where the decimal part is the
    # day + hour in fraction of a year. Same for day. Normally we could just
    # convert them to an integer and be fine, except there are cases where the
    # day rolls over into the next day, e.g. because we're working on a western
    # hemisphere site that starts measuring in one UTC day and keeps measuring
    # through to the next, so the hour exceeds 24.
    #
    # We account for this by subtracting off the fractional part of the year and
    # day computed from the smaller elements. However, we don't subtract the 
    # whole amount off, because we don't want floating point uncertainty to 
    # take us to less than the correct integer. E.g. if the day is 60.5 and
    # we compute hour/24 = 0.50000000001, then int(day - hour/24) = 59.

    # correct the year
    year_frac = day / 366 
    year = (year - 0.99*year_frac).astype('int')
    
    # correct the day
    day_frac = hour / 24
    day = (day - 0.99*day_frac).astype('int')

    return ydh_to_date(year, day, hour)


def ydh_to_date(year, day, hour):
    dates = []
    for y, d, h in zip(year, day, hour):
        ts = pd.Timestamp(y, 1, 1) + pd.Timedelta(days=d-1, hours=h)
        dates.append(ts)
    return pd.DatetimeIndex(dates)


def _split_outside_group(s, splitchar, open='(', close=')'):
    out = []
    grp_cnt = 0
    start = 0
    for i, c in enumerate(s):
        if c == splitchar and grp_cnt == 0:
            out.append(s[start:i+1])
            start = i+1
        elif c == open:
            grp_cnt += 1
        elif c == close:
            grp_cnt -= 1

        if grp_cnt < 0:
            raise ValueError('Given string has an unmatched {} at position {}'.format(close, i))

    # Group count should be 0. If not, then there were more open characters than closing characters
    if grp_cnt > 0:
        raise ValueError('Given string had an unmatched {}'.format(open))

    # Handle the last piece. Check that we haven't just added something, i.e. that the split character isn't the last
    # one in the string. If it is, then we should add an empty element
    if start == len(s) + 1:
        out.append('')
    else:
        out.append(s[start:])

    return out


def fortran_fmt_to_fwf_tuples(fmt_str):
    fmt_str = fmt_str.strip()
    if not fmt_str.startswith('(') or not fmt_str.endswith(')'):
        raise ValueError('The first and last non-whitespace characters of the format string must be open and close '
                         'parentheses, respectively.')
    fmt_parts = _split_outside_group(fmt_str[1:-1], ',')
    colspecs = []
    types = []

    idx = 0
    for part in fmt_parts:
        if '(' in part:
            match = re.match(r'(\d*)(\(.+\))', part)
            repeats = 1 if len(match.group(1)) == 0 else int(match.group(1))
            subfmt = match.group(2)
            subspecs, subtypes = fortran_fmt_to_fwf_tuples(subfmt)

            for i in range(repeats):
                for start, stop in subspecs:
                    colspecs.append((idx+start, idx+stop))
                types += subtypes
                idx += stop
        elif re.match(r'\d+x', part):
            # In Fortran, specifiers like "1x" mean that n spaces do not contain information and are just spacers
            nspaces = int(re.match(r'\d+', part).group(), re.IGNORECASE)
            idx += nspaces
        else:
            match = re.match(r'(\d*)([a-z])(\d+)', part, re.IGNORECASE)
            repeats = 1 if len(match.group(1)) == 0 else int(match.group(1))
            width = int(match.group(3))

            for i in range(repeats):
                colspecs.append((idx, idx+width))
                types.append(match.group(2))
                idx += width

    return colspecs, types


def make_geos_date_ranges(dates):
    geos_tdel = pd.Timedelta(hours=3)
    geos_dates = dates.round('3H').unique().sort_values()
    date_ranges = []

    range_start = geos_dates[0]
    last_date = geos_dates[0]

    for d in geos_dates[1:]:
        if d - last_date > geos_tdel:
            # If the next time is more than the GEOS time step in the future, then the current date range is done.
            # Append it, remembering that the end date is exclusive
            date_ranges.append((range_start, last_date + geos_tdel))
            range_start = d
        last_date = d

    # There will always be a last date range, even if we added a date range in the last loop (in that case only the last
    # time will go in the final date range)
    date_ranges.append((range_start, last_date+geos_tdel))
    return date_ranges


def get_runlog_geos_date_ranges(rldf):
    rl_dates = ydh_to_date(rldf['Year'], rldf['Day'], rldf['Hour'])
    return make_geos_date_ranges(rl_dates)


def compute_file_checksum(file_path):
    """Compute an MD5 checksum on the given file

    Parameters
    ----------
    file_path : str or Path
        Path to the file

    Returns
    -------
    str
        The hexadecimal representation of the checksum
    """
    checksum = md5()
    with open(file_path, 'rb') as f:
        while True:
            block = f.read(1000000)
            if len(block) == 0:
                break
            checksum.update(block)
    return checksum.hexdigest()


def compute_bytes_checksum(bytes):
    """Compute an MD5 checksum on the given bytes

    Parameters
    ----------
    bytes
        A byte string. To compute the checksum of a string,
        use `.encode()` to convert it to bytes.

    Returns
    -------
    str
        The hexadecimal representation of the checksum
    """
    checksum = md5()
    checksum.update(bytes)
    return checksum.hexdigest()
