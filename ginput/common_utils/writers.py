import netCDF4 as ncdf
import numpy as np
import os
from warnings import warn

# Allow the ginput.common_utils.writers module to be imported if there's an issue interacting with the C library. This
# can happen if it's not installed on the system, or when called from a Jupyter notebook.
try:
    from cfunits import Units
except (AssertionError, AttributeError, FileNotFoundError) as e:
    warn(f'Could not import cfunits due to a(n) {type(e).__name__}. Will not be able to enforce CF units conventions.')
    cfunits_imported = False
else:
    cfunits_imported = True

from .ggg_logging import logger
from . import mod_utils, mod_constants, ioutils, readers
from ..mod_maker import tccon_sites
from .. import __version__

_map_scale_factors = {'co2': 1e6, 'n2o': 1e9, 'co': 1e9, 'ch4': 1e9, 'hf': 1e12}
_map_text_units = {'Height': 'km', 'Temp': 'K', 'Pressure': 'hPa', 'Density': 'molecules_cm3', 'h2o': 'parts',
                   'hdo': 'parts', 'co2': 'ppm', 'n2o': 'ppb', 'co': 'ppb', 'ch4': 'ppb', 'hf': 'ppt', 'o2': 'parts',
                   'gravity': 'm_s2'}
# For units whose old text names are not recognizable under the CF convention, replace them with ones that are
_map_canonical_units = _map_text_units.copy()
_map_canonical_units.update({'Density': 'molecules/cm^3',
                             'h2o': 'mol/mol',
                             'hdo': 'mol/mol',
                             'o2': 'mol/mol',
                             'gravity': 'm/s^2'})
_map_standard_names = {'Height': 'altitude',
                       'Temp': 'air_temperature',
                       'Pressure': 'air_pressure',
                       'Density': 'air_number_density',
                       'h2o': 'water_{w_or_d}_mole_fraction',
                       'hdo': 'heavy_water_{w_or_d}_mole_fraction',
                       'co2': 'carbon_dioxide_{w_or_d}_mole_fraction',
                       'n2o': 'nitrous_oxide_{w_or_d}_mole_fraction',
                       'co':  'carbon_monoxide_{w_or_d}_mole_fraction',
                       'ch4': 'methane_{w_or_d}_mole_fraction',
                       'hf':  'hydrofluoric_acid_{w_or_d}_mole_fraction',
                       'o2':  'oxygen_{w_or_d}_mole_fraction',
                       'gravity': 'gravitational_acceleration'}

_map_var_mapping = {'Temperature': 'Temp'}
_map_var_order = ('Height', 'Temp', 'Pressure', 'Density', 'h2o', 'hdo', 'co2', 'n2o', 'co', 'ch4', 'hf', 'o2', 'gravity')

_float_fmt = '{:8.3f}'
_exp_fmt = '{:10.3E}'
_map_var_formats = {'Height': _float_fmt, 'Temp': _float_fmt, 'Pressure': _exp_fmt, 'Density': _exp_fmt,
                    'h2o': _exp_fmt, 'hdo': _exp_fmt, 'co2': _float_fmt, 'n2o': _float_fmt, 'co': _exp_fmt,
                    'ch4': '{:7.1f}', 'hf': _float_fmt, 'o2': '{:7.4f}', 'gravity': '{:6.3f}'}

wmf_message = ['NOTE: The gas concentrations (including H2O) are WET MOLE FRACTIONS. If you require dry mole fractions,',
               'you must calculate [H2O]_dry = (1/[H2O]_wet - 1)^-1 and then [gas]_dry = [gas]_wet * (1 + [H2O]_dry).']


class CFUnitsError(Exception):
    def __init__(self, unit_string):
        msg = '"{}" is not a CF-compliant unit'.format(unit_string)
        super(CFUnitsError, self).__init__(msg)


def _cfunits(unit_string, no_cfunits=False):
    """
    Convert a units string to a CF-compliant one.
    :param unit_string: the unit string to convert
    :param no_cfunits: if True, then will not format unit strings if CFUnits failed to import. Has no effect if CFUnits
     did import successfully.
    :return: the converted unit string
    :raises CFUnitsError: if the string cannot be made CF-compliant
    """
    if not cfunits_imported:
        if no_cfunits:
            return unit_string
        else:
            raise ImportError('cfunits.Unit was not successfully imported. Use the no_cfunits keyword to ignore that '
                              'problem and skip making the units CF compliant.')

    units = Units(unit_string).formatted()
    if units is None:
        raise CFUnitsError(unit_string)
    else:
        return units


def write_map_from_vmr_mod(vmr_file, mod_file, map_output_dir, fmt='txt', wet_or_dry='wet', site_abbrev='xx',
                           no_cfunits=False):
    """
    Create a .map file from a .vmr and .mod file

    :param vmr_file: the path to the .vmr file to read the gas concentrations from
    :param mod_file: the path to the .mod file to read the met variables from
    :param map_output_dir: the directory to write the .map file to. It will automatically be given the correct name.
    :param fmt: what format to write the .map files in, either "txt" for the original text files or "nc" for the new
     netCDF files.
    :param wet_or_dry: whether to write wet or dry mole fractions.
    :param site_abbrev: the site abbreviation to go in the file name and netCDF attributes.
    :param no_cfunits: if True, then will not format unit strings if CFUnits failed to import. Has no effect if CFUnits
     did import successfully.
    :return: none, writes .map or .map.nc file.
    """
    if not os.path.isfile(vmr_file):
        raise OSError('vmr_file "{}" does not exist'.format(vmr_file))
    if not os.path.isdir(map_output_dir):
        raise OSError('map_output_dir "{}" is not a directory'.format(map_output_dir))
    if wet_or_dry not in ('wet', 'dry'):
        raise ValueError('wet_or_dry must be "wet" or "dry"')

    map_name = mod_utils.map_file_name_from_mod_vmr_files(site_abbrev, mod_file, vmr_file, fmt)
    map_name = os.path.join(map_output_dir, map_name)
    mapdat, obs_lat = _merge_and_convert_mod_vmr(vmr_file, mod_file, wet_or_dry=wet_or_dry)

    if fmt in {'txt', 'text'}:
        _write_text_map_file(mapdat=mapdat, obs_lat=obs_lat, map_file=map_name, wet_or_dry=wet_or_dry)
    elif fmt in {'nc', 'netcdf'}:
        moddat = readers.read_mod_file(mod_file)
        _write_ncdf_map_file(mapdat=mapdat, obs_lat=obs_lat, obs_date=moddat['file']['datetime'], obs_site=site_abbrev,
                             file_lat=moddat['file']['lat'], file_lon=moddat['file']['lon'],
                             map_file=map_name, wet_or_dry=wet_or_dry, no_cfunits=no_cfunits)


def _merge_and_convert_mod_vmr(vmr_file, mod_file, vmr_vars=('h2o', 'hdo', 'co2', 'n2o', 'co', 'ch4', 'hf', 'o2'),
                               mod_vars=('Height', 'Temperature', 'Pressure', 'Density', 'gravity'), wet_or_dry='wet'):
    vmrdat = readers.read_vmr_file(vmr_file)
    moddat = readers.read_mod_file(mod_file)
    mapdat = dict()

    # put the .mod variables (always on the GEOS native grid) on the same grid as the .vmr file (whatever that is).
    obs_lat = moddat['constants']['obs_lat']
    zgrid = vmrdat['profile']['altitude']
    for mvar in mod_vars:
        if mvar == 'gravity':
            lat = np.broadcast_to([obs_lat], moddat['profile']['Height'].shape)
            mapdat['gravity'], _ = mod_utils.gravity(lat, moddat['profile']['Height'])
        elif mvar == 'Density':
            mapdat['Density'] = mod_utils.number_density_air(moddat['profile']['Pressure'], moddat['profile']['Temperature'])
        else:
            mout = _map_var_mapping[mvar] if mvar in _map_var_mapping else mvar
            scale = _map_scale_factors[mout] if mout in _map_scale_factors else 1
            mapdat[mout] = moddat['profile'][mvar] * scale

    mapdat = mod_utils.interp_to_zgrid(mapdat, zgrid=zgrid)

    # the .vmr variables should be on the desired zgrid already since we took the zgrid from that file. However, we
    # may need to change them from dry to wet VMRs. Setting the H2O dry mole fraction to 0 makes the mole fractions dry
    # b/c there's no water, though it doesn't matter because of the `if wet_or_dry` in the loop.
    h2o_dmf = vmrdat['profile']['h2o'] if wet_or_dry == 'wet' else 0

    for vvar in vmr_vars:
        scale = _map_scale_factors[vvar] if vvar in _map_scale_factors else 1
        gas_dmf = vmrdat['profile'][vvar] * scale
        if wet_or_dry == 'wet':
            gas_dmf = mod_utils.dry2wet(gas_dmf, h2o_dmf)
        mapdat[vvar] = gas_dmf

    return mapdat, obs_lat


def _write_text_map_file(mapdat, obs_lat, map_file, wet_or_dry):
    def iter_values_formats(irow):
        for varname in _map_var_order:
            yield mapdat[varname][irow], _map_var_formats[varname]
    # First build up the header. A GGG2014 map file contains in the header:
    #   nheader/ncols
    #   its basename
    #   program versions
    #   reference to the TCCON wiki
    #   avogadro constant
    #   mass dry air
    #   mass H2O
    #   latitude

    header = [os.path.basename(map_file),
              '{:25} Version {:9} JLL, SR, MK'.format('GINPUT', __version__),
              'Please see https://tccon-wiki.caltech.edu for a complete description of this file and its usage.']

    # Usage and warnings
    if wet_or_dry == 'wet':
        header += wmf_message

    # Constants
    header.append('Avogadro (molecules/mole): {}'.format(mod_constants.avogadro))
    header.append('Mass_Dry_Air (kg/mole): {}'.format(mod_constants.mass_dry_air))
    header.append('Mass_H2O (kg/mole): {}'.format(mod_constants.mass_h2o))
    header.append('Latitude (degrees): {}'.format(obs_lat))

    # Column headers
    header.append(','.join(_map_var_order))
    header.append(','.join(_map_text_units[v] for v in _map_var_order))

    # Now prepend the number of header rows and data columns
    header.insert(0, '{} {}'.format(len(header)+1, len(_map_var_order)))

    # Begin writing
    with open(map_file, 'w') as wobj:
        for line in header:
            wobj.write(line + '\n')
        for i in range(mapdat['Height'].size):
            line = ','.join(fmt.format(value) for value, fmt in iter_values_formats(i))
            wobj.write(line + '\n')


def _write_ncdf_map_file(mapdat, obs_lat, obs_date, file_lat, file_lon, obs_site, map_file, wet_or_dry,
                         no_cfunits=False):
    with ncdf.Dataset(map_file, 'w') as wobj:
        alt_human_units = _map_canonical_units['Height']
        alt_units = _cfunits(alt_human_units, no_cfunits=no_cfunits)
        altdim = ioutils.make_ncdim_helper(wobj, 'altitude', mapdat['Height'],
                                           units=alt_units, full_units=alt_human_units, long_name='altitude',
                                           tccon_name='height')

        # not actually used, but makes more sense to write these as variables/dimensions than attributes
        ioutils.make_nctimedim_helper(wobj, 'time', np.array([obs_date]), time_units='hours', long_name='time')
        ioutils.make_ncdim_helper(wobj, 'lat', np.array([obs_lat]), units='degrees_north', long_name='latitude')

        for varname in _map_var_order:
            if varname == 'Height':
                # already defined as a coordinate
                continue

            human_units = _map_canonical_units[varname]
            cf_units = _cfunits(human_units, no_cfunits=no_cfunits)
            std_name = _map_standard_names[varname].format(w_or_d=wet_or_dry)
            ioutils.make_ncvar_helper(wobj, varname.lower(), mapdat[varname], dims=[altdim],
                                      units=cf_units, full_units=human_units, long_name=std_name)

        # finally the file attributes, including ginput version, constants used, WMF message, etc. Global CF attributes
        # to include are "comment", "Conventions" (?), "history", "institution", "references", "source", and "title"
        wobj.comment_full_units = 'The full_units attribute provides a human-readable counterpart to the ' \
                                  'CF-compliant units attribute'
        if wet_or_dry == 'wet':
            wobj.comment_wet_mole_fractions = ' '.join(wmf_message)
        wobj.comment_file_lat_lon = 'These are the latitude/longitude recorded in the input .mod file name. They ' \
                                    'may be rounded to the nearest degree.'
        wobj.contact = 'Joshua Laughner (jlaugh@caltech.edu)'
        wobj.Conventions = 'CF-1.7'
        wobj.institution = 'California Institute of Technology, Pasadena, CA, USA'
        wobj.references = 'https://tccon-wiki.caltech.edu'
        wobj.source = 'ginput version {}'.format(__version__)
        wobj.title = 'GGG2020 TCCON prior profiles'

        creation_note = 'ginput (commit {})'.format(mod_utils.vcs_commit_info()[0])
        ioutils.add_creation_info(wobj, creation_note, creation_att_name='history')

        # ggg-specific attributes
        wobj.file_latitude = file_lat
        wobj.file_longitude = file_lon
        wobj.file_datetime = obs_date.strftime('%Y-%m-%d %H:%M:%S UTC')
        wobj.tccon_site = obs_site
        wobj.tccon_site_full_name = tccon_sites.site_dict[obs_site]['name'] if obs_site in tccon_sites.site_dict else 'N/A'
        wobj.constant_avogadros_number = mod_constants.avogadro
        wobj.constant_avogadros_number_units = 'molecules.mole-1'  # CF convention would be '1.66053878316273e-24 1' which is just ugly
        wobj.constant_mass_dry_air = mod_constants.mass_dry_air
        wobj.constant_mass_dry_air_units = _cfunits('kg/mol', no_cfunits=no_cfunits)
        wobj.constant_mass_h2o = mod_constants.mass_h2o
        wobj.constant_mass_h2o_units = _cfunits('kg/mol', no_cfunits=no_cfunits)


def write_vmr_file(vmr_file, tropopause_alt, profile_date, profile_lat, profile_alt, profile_gases, gas_name_order=None,
                   extra_header_info=None):
    """
    Write a new-style .vmr file (without seasonal cycle, secular trends, and latitudinal gradients

    :param vmr_file: the path to write the .vmr file ar
    :type vmr_file: str

    :param tropopause_alt: the altitude of the tropopause, in kilometers
    :type tropopause_alt: float

    :param profile_date: the date of the profile
    :type profile_date: datetime-like

    :param profile_lat: the latitude of the profile (south is negative)
    :type profile_lat: float

    :param profile_alt: the altitude levels that the profiles are defined on, in kilometers
    :type profile_alt: array-like

    :param profile_gases: a dictionary of the prior profiles to write to the .vmr file.
    :type profile_gases: dict(array)

    :param gas_name_order: optional, a list/tuple specifying what order the gases are to be written in. If not given,
     they will be written in whatever order the iteration through ``profile_gases`` defaults to. If given, then an
     error is raised if any of the gas names listed here are not present in ``profile_gases`` (comparison is case-
     insensitive). Any gases not listed here that are in ``profile_gases`` are skipped.
    :type gas_name_order: list(str)

    :param extra_header_info: optional, if given, must be a dictionary or list of lines to include at the end of the
     header in the .vmr. If a list, must be a list of strings. If a dict, each line will be formatted as "key: value"
     and the keys/values may be any type.

    :return: none, writes the .vmr file.
    """

    if np.ndim(profile_alt) != 1:
        raise ValueError('profile_alt must be 1D')

    if gas_name_order is None:
        gas_name_order = [k for k in profile_gases.keys()]

    if extra_header_info is None:
        extra_header_info = []
    elif isinstance(extra_header_info, dict):
        extra_header_info = ['{}: {}'.format(k, v) for k, v in extra_header_info.items()]

    gas_name_order_lower = [name.lower() for name in gas_name_order]
    gas_name_mapping = {k: None for k in gas_name_order}

    # Check that all the gases in the profile_gases dict are expected to be written.
    for gas_name, gas_data in profile_gases.items():
        if gas_name.lower() not in gas_name_order_lower:
            logger.warning('Gas "{}" was not listed in the gas name order and will not be written to the .vmr '
                           'file'.format(gas_name))
        elif np.shape(gas_data) != np.shape(profile_alt):
            raise ValueError('Gas "{}" has a different shape ({}) than the altitude data ({})'.format(
                gas_name, np.shape(gas_data), np.shape(profile_alt)
            ))
        elif np.ndim(gas_data) != 1:
            raise ValueError('Gas "{}" is not 1D'.format(gas_name))
        else:
            idx = gas_name_order_lower.index(gas_name.lower())
            gas_name_mapping[gas_name_order[idx]] = gas_name

    # Write the header, which starts with the number of header lines and data columns, then has the tropopause altitude,
    # profile date as a decimal year, and profile latitude. I'm going to skip the secular trends, seasonal cycle, and
    # latitude gradient because those are not necessary.
    alt_fmt = '{:9.3f} '
    gas_fmt = '{:.3E}  '
    table_header = ['Altitude'] + ['{:10}'.format(name) for name in gas_name_order]
    header_lines = [' GINPUT_VERSION: {}'.format(__version__),
                    ' ZTROP_VMR: {:.1f}'.format(tropopause_alt),
                    ' DATE_VMR: {:.3f}'.format(mod_utils.date_to_decimal_year(profile_date)),
                    ' LAT_VMR: {:.2f}'.format(profile_lat)] \
                + [' ' + l for l in extra_header_info] \
                + [' '.join(table_header)]

    with open(vmr_file, 'w') as fobj:
        _write_header(fobj, header_lines, len(gas_name_order) + 1)
        for i in range(np.size(profile_alt)):
            fobj.write(alt_fmt.format(profile_alt[i]))
            for gas_name in gas_name_order:
                if gas_name_mapping[gas_name] is not None:
                    gas_conc = profile_gases[gas_name_mapping[gas_name]][i]
                else:
                    gas_conc = 0.0
                fobj.write(gas_fmt.format(gas_conc))
            fobj.write('\n')


def _write_header(fobj, header_lines, n_data_columns):
    """
    Write a standard header to a GGG file.

    A typical header in a GGG file begins with a line with two numbers: the number of header lines and the number of
    data columns. This automatically formats that line, then writes the given lines.

    :param fobj: File handle (i.e. object returned by :func:`open`)
    :type fobj: :class:`_io.TextIO`

    :param header_lines: a list of header lines (after the first) to write. Should be strings, each may or may not end
     in newlines (both cases are handled).
    :type header_lines: list(str)

    :param n_data_columns: number of data columns in the main part of the file
    :type n_data_columns: int

    :return: none, writes to the file object.
    """
    line1 = ' {} {}\n'.format(len(header_lines)+1, n_data_columns)
    fobj.write(line1)
    header_lines = [l if l.endswith('\n') else l + '\n' for l in header_lines]
    fobj.writelines(header_lines)



