from __future__ import print_function, division

import cftime
import datetime as dt
from hashlib import sha1
import netCDF4 as ncdf
import numpy as np
from subprocess import CalledProcessError
import sys

from . import mod_utils


class SmartHandle:
    """An improved handle to standard files

    This solves the problem of opening either a file on disk or the special stdin/stdout file objects. If the filename
    is "-" or `None`, then stdin or stdout is opened (depending on the mode - stdin for read modes, stdout for any other
    mode). Otherwise the file specified is opened as normal.

    Parameters
    ----------
    filename: str or None
        The path to the file to open. If "-" or `None`, then stdin/stdout is "opened"
    mode: str
        The permission mode to open the file under, may be any mode recognized by the :func:`open` builtin.

    Examples
    --------

    You can use this pretty much anywhere in place of :func:`open`. The following would write "This is a log message"
    to the :file:`logfile.txt` file:

    >>> f = SmartHandle('logfile.txt', 'w')
    >>> f.write('This is a log message\\n')
    >>> f.close()

    But we can also use this to write that to stdout:

    >>> f = SmartHandle('-', 'w')  # stdout is chosen because the mode is write
    >>> f.write('This is a log message\\n')
    >>> f.close()  # closing will have no effect and can be included or omitted safely

    This works in `with` blocks as well
    >>> with SmartHandle('-', 'w') as f:
    >>>      f.write('This is a log message\\n')

    This is helpful if, say, you want to print messages to the screen normally but have the ability to redirect them
    to a file. A single `with` block can write to either a file or stdout depending on the value of the filename, which
    helps prevent having to duplicate your code.
    """
    def __init__(self, filename, mode):
        if filename is None or filename == '-':
            if mode.startswith('r'):
                self._handle = sys.stdin
            else:
                self._handle = sys.stdout
            self._do_close = False
        else:
            self._handle = open(filename, mode)
            self._do_close = True

    def __getattr__(self, item):
        # this allows pass through of all read/write methods to the _handle
        return getattr(self._handle, item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the open file.

        This has no effect if it was stdin or stdout opened.
        """
        if self._do_close:
            self._handle.close()


def make_ncdim_helper(nc_handle, dim_name, dim_var, **attrs):
    """
    Create a netCDF dimension and its associated variable simultaneously

    Typically in a netCDF file, each dimension should have a variable with the same name that defines the coordinates
    for that dimension. This function streamlines the process of creating a dimension with its associated variable.

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param dim_name: the name to give both the dimension and its associated variable
    :type dim_name: str

    :param dim_var: the variable to use when defining the dimension's coordinates. The dimensions length will be set
     by the size of this vector. This must be a 1D numpy array or comparable type.
    :type dim_var: :class:`numpy.ndarray`

    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the dimension object
    :rtype: :class:`netCDF4.Dimension`
    """
    if np.ndim(dim_var) != 1:
        raise ValueError('Dimension variables are expected to be 1D')
    dim = nc_handle.createDimension(dim_name, np.size(dim_var))
    var = nc_handle.createVariable(dim_name, dim_var.dtype, dimensions=(dim_name,))
    var[:] = dim_var
    var.setncatts(attrs)
    return dim


def make_nctimedim_helper(nc_handle, dim_name, dim_var, base_date=dt.datetime(1970, 1, 1), time_units='seconds',
                          calendar='gregorian', **attrs):
    """
    Create a CF-style time dimension.

    The Climate and Forecast (CF) Metadata Conventions define standardized conventions for how to represent geophysical
    data. Time is one of the trickiest since there are multiple ways of identifying dates that can be ambiguous. The
    standard representation is to give time in units of seconds/minutes/hours/days since a base time in a particular
    calendar. This function handles creating the a time dimension and associated variable from any array-like object of
    datetime-like object.

    For more information, see:

        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#time-coordinate

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param dim_name: the name to give both the dimension and its associated variable
    :type dim_name: str

    :param dim_var: the variable to use when defining the dimension's coordinates. The dimensions length will be set
     by the size of this vector. This must be a 1D numpy array or comparable type.
    :type dim_var: :class:`numpy.ndarray`

    :param base_date: the date and time to make the time coordinate relative to. The default is midnight, 1 Jan 1970.
    :type base_date: datetime-like object

    :param time_units: a string indicating what unit to use as the count of time between the base date and index date.
     Options are 'seconds', 'minutes', 'hours', or 'days'.  This is more restrictive than the CF convention, but avoids
     the potential pitfalls of using months or years.
    :type time_units: str

    :param calendar: one of the calendar types defined in the CF conventions document (section 4.4.1)
    :type calendar: str

    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the dimension object, and a dictionary describing the units, calendar, and base date of the time dimension
    :rtype: :class:`netCDF4.Dimension`, dict
    """
    allowed_time_units = ('seconds', 'minutes', 'hours', 'days')
    if time_units not in allowed_time_units:
        raise ValueError('time_units must be one of: {}'.format(', '.join(allowed_time_units)))

    units_str = '{} since {}'.format(time_units, base_date.strftime('%Y-%m-%d %H:%M:%S'))
    # date2num requires that the dates be given as basic datetimes. We'll handle converting Pandas timestamps, either
    # as a series or datetime index, but other types will need handled by the user.
    try:
        date_arr = ncdf.date2num(dim_var, units_str, calendar=calendar)
    except TypeError:
        dim_var = [d.to_pydatetime() for d in dim_var]
        date_arr = ncdf.date2num(dim_var, units_str, calendar=calendar)
    base_date = cftime.date2num(base_date, units_str, calendar=calendar)
    attrs.update({'units': units_str, 'calendar': calendar, 'base_date': base_date})
    dim = make_ncdim_helper(nc_handle, dim_name, date_arr, **attrs)
    return dim


def make_ncvar_helper(nc_handle, var_name, var_data, dims, **attrs):
    """
    Create a netCDF variable and store the data for it simultaneously.

    This function combines call to :func:`netCDF4.createVariable` and assigning the variable's data and attributes.

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param var_name: the name to give the variable
    :type var_name: str

    :param var_data: the array to store in the netCDF variable.
    :type var_data: :class:`numpy.ndarray`

    :param dims: the dimensions to associate with this variable. Must be a collection of either dimension names or
     dimension instances. Both types may be mixed. This works well with :func:`make_ncdim_helper`, since it returns the
     dimension instances.
    :type dims: list(:class:`netCDF4.Dimensions` or str)

    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the variable object
    :rtype: :class:`netCDF4.Variable`
    """
    dim_names = tuple([d if isinstance(d, str) else d.name for d in dims])
    var = nc_handle.createVariable(var_name, var_data.dtype, dimensions=dim_names)
    var[:] = var_data
    var.setncatts(attrs)
    return var


def make_creation_info(filename, creation_note=None):
    """
    Create a string describing the creation of a stored file.

    This is intended to create a string attribute that describes how a file containing look up tables or other
    derived information for the GGG retrieval. The string will include the date the file was generated and, provided
    the Mercurial repository is intact, the commit hash that generated the file.

    :param filename: the path to the file that is being created. Its status will be ignored when determining if the
     Mercurial commit is clean or not.
    :type filename: str

    :param creation_note: A short string describing how the file was created. If given, it will be inserted into the
     string as "Created by {note} on {date}". Typically this will indicate what function created the file.
    :type creation_note: str

    :return: the description string
    :rtype: str
    """
    now = dt.datetime.now()
    try:
        git_or_hg = 'git' if mod_utils._is_git_repo() else 'mercurial'
        commit_hash, branch, _ = mod_utils.vcs_commit_info()
        clean_or_dirty = 'clean' if mod_utils.vcs_is_commit_clean(ignore_files=[filename]) else 'dirty'
    except (CalledProcessError, FileNotFoundError):  # if hg is not installed, get a FileNotFoundError instead of CalledProcessError
        if creation_note is not None:
            description = 'Created by {note} on {date} (could not acquire {vcs} commit information)'
        else:
            description = 'Created on {date} (could not acquire {vcs} commit information)'
        description = description.format(note=creation_note, date=now, vcs=git_or_hg)
    else:
        if creation_note is not None:
            description = 'Created by {note} on {date} ({vcs} commit {commit} on branch {branch}, {cleanstate})'
        else:
            description = 'Created with {vcs} commit {commit} on branch {branch} ({cleanstate}) on {date}'

        description = description.format(note=creation_note, date=now, commit=commit_hash, branch=branch,
                                         cleanstate=clean_or_dirty, vcs=git_or_hg)
    return description


def add_creation_info(nc_handle, creation_note=None, creation_att_name='history'):
    """
    Add a creation note to a netCDF file.

    :param nc_handle: a handle to the netCDF4 dataset to add the attribute to.
    :type nc_handle: :class:`netCDF4.Dataset`

    :param creation_note: see the same named parameter in :func:`make_creation_info`.
    :type creation_note: str

    :param creation_att_name: what attribute to store the creation information as.
    :type creation_att_name: str

    :return: None
    """
    description = make_creation_info(nc_handle.filepath(), creation_note=creation_note)
    nc_handle.setncattr(creation_att_name, description)


def make_dependent_file_hash(dependent_file):
    """
    Create an SHA1 hash of a file.

    :param dependent_file: the path to the file to hash.
    :type dependent_file: str

    :return: the SHA1 hash
    :rtype: str
    """
    hashobj = sha1()
    with open(dependent_file, 'rb') as fobj:
        block = fobj.read(4096)
        while block:
            hashobj.update(block)
            block = fobj.read(4096)

    return hashobj.hexdigest()


def add_dependent_file_hash(nc_handle, hash_att_name, dependent_file):
    """
    Add an SHA1 hash of another file as an attribute to a netCDF file.

    This is intended to create an attribute that list the SHA1 hash of a file that the netCDF file being created
    depends on.

    :param nc_handle: a handle to the netCDF4 dataset to add the attribute to.
    :type nc_handle: :class:`netCDF4.Dataset`

    :param hash_att_name: the name to give the attribute that will store the hash. It is recommended to include the
     substring "sha1" so that it is clear what hash function was used.
    :type hash_att_name: str

    :param dependent_file: the file to generate the hash of.
    :type dependent_file: str

    :return: None
    """
    hash_hex = make_dependent_file_hash(dependent_file)
    nc_handle.setncattr(hash_att_name, hash_hex)
