"""
This module serves as the interface between the OCO/GOSAT code and the TCCON prior generator.

Usual GGG users will not need to concern themselves with this file. GGG developers will want to ensure that this module
is still able to generate equivalent latitudes and CO2 priors from the TCCON prior and mod_maker code.

If you wish to call this code from within other python code, the function :func:`acos_interface_main` is the entry
point. A command line interface is also provided.
"""

from __future__ import print_function, division

import argparse
from contextlib import contextmanager
import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import lru_cache
import h5py
from itertools import repeat, product
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os
import sys
import traceback

from ..common_utils import mod_utils, mod_constants
from ..common_utils.sat_utils import time_weight, datetime2datenum
from ..common_utils.ggg_logging import logger, setup_logger
from ..mod_maker import mod_maker
from . import tccon_priors, fo2_prep
from .. import __version__

__acos_int_version__ = '1.2.3'

_acos_tstring_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
# Values lower than this will be replaced with NaNs when reading in the resampled met data
_fill_val_threshold = -9e5
# NaNs will be replaced with this value when writing the HDF5 file
_fill_val = -999999
# This will be used as a fill value for strings. It must be bytes b/c HDF5s do not accept fixed length unicode strings
# so we use fixed length ASCII strings.
_string_fill = b'N/A'


class ErrorHandler(object):
    _err_codes = {'bad_time_str': 1,
                  'eqlat_failure': 2,
                  'prior_failure': 3,
                  'met_qual_flag': 4,
                  'out_of_range_date': 5,
                  'no_data': 6,
                  'cannot_find_geos': 7}

    _err_descriptions = {'bad_time_str': 'The OCO/ACOS time string could not be parsed',
                         'eqlat_failure': 'A problem occurred while calculating equivalent latitude',
                         'prior_failure': 'A problem occurred while calculating the prior itself',
                         'met_qual_flag': 'The quality flag in the met data was nonzero',
                         'out_of_range_date': 'The sounding date was out of the expected range',
                         'no_data': 'The met data for altitude was all fill values',
                         'cannot_find_geos': 'Could not find a GEOS FP-IT file bracketing the sounding time. '}

    def __init__(self, suppress_error):
        self.suppress_error = suppress_error

    def handle_err(self, err, err_code_name, flags, inds):
        """
        Handle an error

        If this class has ``suppress_error == False``, then the error will be raised as normal. Otherwise, the error
        will be printed via the logger but will not halt execution. A full traceback will only be printed if the
        logger's level is set to DEBUG or higher.

        :param err: the exception object (caught by the try block) that represents the error.
        :type err: :class:`Exception`

        :param err_code_name: must be one of the keys in the class attribute ``_err_codes``. This will be used to
         determine the numerical error code stored in the flags array.
        :type err_code_name: str

        :param flags: an integer array into which the error code will be stored. If None, then the error will always be
         raised as normal, regardless of the value of ``self.suppress_error``.
        :type flags: :class:`numpy.ndarray`

        :param inds: any form of indexing that will assign the error code to the correct place in the flags array, i.e.
         that is valid for ``flags[inds] = code``.

        :return: None
        """
        if not self.suppress_error or flags is None:
            raise err

        self.set_flag(err_code_name, flags, inds)

        if logger.level <= logging.DEBUG:
            logger.exception(err)
        else:
            last_call = traceback.format_stack()[-2].strip()
            msg = '{}: {}'.format(type(err).__name__, err.args[0].strip())
            logger.error(
                'Error at indices = {} in {}. Error was: "{}". (Use -vv at the command line for full traceback.)'
                .format(inds, last_call, msg))

    def set_flag(self, err_code_name, flags, inds):
        try:
            ecode = self._err_codes[err_code_name]
        except KeyError:
            raise ValueError('"{}" is not a valid error type'.format(err_code_name))

        flags[inds] = ecode

    def get_error_descriptions(self):
        descriptions = []
        for k in self._err_codes:
            descriptions.append((self._err_codes[k], self._err_descriptions[k]))

        # Ensure descriptions are ascending by the error code
        descriptions.sort(key=lambda el: el[0])

        descript_strs = []
        for code, meaning in descriptions:
            descript_strs.append('{}: {}'.format(code, meaning))

        return 'Value meanings:  ' + '; '.join(descript_strs)


_def_errh = ErrorHandler(suppress_error=False)


def acos_interface_main(instrument, met_resampled_file, geos_files, output_file, mlo_co2_file=None, smo_co2_file=None,
                        use_trop_eqlat=False, cache_strat_lut=False, truncate_mlo_smo_by=0, nprocs=0, interp_pickle_dir='.',
                        fo2_file=fo2_prep.DEFAULT_FO2_FILE, auto_update_fo2_file=False,
                        error_handler=_def_errh):
    """
    The primary interface to create CO2 priors for the ACOS algorithm

    :param met_resampled_file: the path to the HDF5 file containing the met data resampled to the satellite footprints
    :type met_resampled_file: str

    :param geos_files: the list of GEOS FP or FP-IT pressure level met files that bracket the satellite observations.
     That is, if the first satellite observation in ``met_resampled_file`` is on 2019-05-22 02:30:00Z and the last is
     at 2019-05-22 03:30:00Z, then the GEOS files for 0, 3, and 6 UTC for 2019-05-22 must be listed.
    :type geos_files: list(str)

    :param output_file: the path to where the priors (and related data) should be written as an HDF5 file.
    :type output_file: str

    :param mlo_co2_file: optional, the path to the Mauna Loa monthly CO2 flask data file. If not given (None), the
     default file included in the repository is used.
    :type mlo_co2_file: str or None

    :param smo_co2_file: optional, the path to the American Samoa monthly CO2 flask data file. If not given (None), the
     default file included in the repository is used.
    :type smo_co2_file: str

    :param cache_strat_lut: optional, set to ``True`` to enable the CO2 record instance to cache its stratospheric
     concentration lookup table to avoid having to recalculate it each time. When ``True``, the CO2 record class will
     determine if the LUT needs updated based on the state of the dependencies. When ``False`` (default), the LUT will
     always be calculated from the MLO/SMO record and the age spectra and will never be saved.
    :type cache_strat_lut: bool

    :param truncate_mlo_smo_by: optional, number of months before the latest date in the ``met_resampled_file`` to
     truncate the MLO/SMO data to (and extrapolate beyond). The default is 2, e.g. if the latest date in the met file
     is in Aug 2021, the MLO/SMO data will only be used up to June 2021 - but they *must* include data up to that
     month, or an error is raised.

    :param fo2_file: Path to a file containing a timeseries of O2 DMFs. This is typically produced by the "fo2_prep"
     subcommand of ginput, or it can be created automatically by setting ``auto_update_fo2_file = True``.
    :type fo2_file: os.PathLike

    :param auto_update_fo2_file: set to ``True`` to enable this function to automatically create or update the O2
     DMF input file at the path specified by ``fo2_file``. This will by default do nothing if the ``fo2_file`` has
     been updated within the last 7 days. Otherwise it will download the latest Scripps and NOAA data needed to create
     the file and check if there is new data in them that requires an update to the ``fo2_file``. If ``fo2_file`` does
     not exist, the outputs are downloaded and created. Note that outputs will always be downloaded to the default directory.
     using this option. If you want more control, please use the "update_fo2" subcommand of ``run_ginput.py``.

    :return: None, writes results to the HDF5 ``output_file``.
    """

    def check_mlo_smo_gas_file(gas_file, mlo_or_smo):
        if gas_file is not None and '{gas}' not in gas_file:
            raise ValueError('The {inst} instrument generates multiple gases; if passing an explicit path '
                             'to the {mlosmo} file, that path must include {{gas}} which will be replaced '
                             'with the lower case forumla (e.g. "co2") for each gas that needs it (co2, '
                             'n2o, or ch4)'.format(inst=instrument, mlosmo=mlo_or_smo))

    def fmt_gas_file(gas_file, this_gas):
        if gas_file is None:
            return gas_file
        else:
            return gas_file.format(gas=this_gas)

    if mlo_co2_file is not None and not os.path.exists(mlo_co2_file):
        raise IOError('Given path for mlo_co2_file ({}) does not exist'.format(mlo_co2_file))
    if smo_co2_file is not None and not os.path.exists(smo_co2_file):
        raise IOError('Given path for smo_co2_file ({}) does not exist'.format(smo_co2_file))

    if instrument == 'oco':
        met_data, prior_flags = read_oco_resampled_met(met_resampled_file, error_handler=error_handler)
        gases = ('co2',)
        prior_group = 'priors'
        record_group = 'mlo_smo_record'
    elif instrument == 'gosat':
        met_data, prior_flags = read_gosat_resampled_met(met_resampled_file, error_handler=error_handler)
        gases = ('co2',)
        prior_group = 'priors'
        record_group = 'mlo_smo_record'
    elif instrument == 'geocarb':
        met_data, prior_flags = read_geocarb_resampled_met(met_resampled_file, error_handler=error_handler)
        gases = ('co', 'ch4', 'co2')
        prior_group = '{}_priors'
        record_group = 'mlo_smo_{}_record'
        check_mlo_smo_gas_file(mlo_co2_file, 'MLO')
        check_mlo_smo_gas_file(smo_co2_file, 'SMO')
    else:
        raise ValueError('instrument must be "oco", "gosat", or "geocarb"')

    # Reshape the met data from (soundings, footprints, levels) to (profiles, level)
    orig_shape = met_data['pv'].shape
    flags_orig_shape = prior_flags.shape

    nlevels = orig_shape[-1]
    pv_array = met_data['pv'].reshape(-1, nlevels)
    theta_array = met_data['theta'].reshape(-1, nlevels)

    datenum_array = met_data['datenums'].reshape(-1)
    qflag_array = met_data['quality_flags'].reshape(-1)

    prior_flags = prior_flags.reshape(-1)

    # Before we spent a lot of time calculating the equivalent latitude, let's determine
    # the truncation date
    max_date = np.max(met_data['dates'])
    # Also fix the end date of the MLO/SMO record relative to the data date, rather than
    # the execution date. Requested by SDOS on 14 Oct 2021.
    record_end_date = mod_utils.start_of_month(max_date) + relativedelta(years=2, months=1)

    if truncate_mlo_smo_by is not None:
        check_date = dt.datetime.today() + dt.timedelta(days=30)
        if max_date > check_date:
            # If there's some crazy fill value in the met file that causes the maximum date
            # to be in the future, that would cause our truncation to misbehave. 
            # So far, I've only seen fill values be ~1993, so this shouldn't happen.
            raise ValueError('A date in met file "{}" is more than 30 days in the future from today. This breaks the MLO/SMO data truncation!'.format(met_resampled_file))

        truncate_mlo_smo_date = mod_utils.start_of_month(max_date) - relativedelta(months=truncate_mlo_smo_by)
        truncate_mlo_smo_date = dt.datetime(truncate_mlo_smo_date.year, truncate_mlo_smo_date.month, truncate_mlo_smo_date.day)
    else:
        truncate_mlo_smo_date = None


    eqlat_array = compute_sounding_equivalent_latitudes(sounding_pv=pv_array, sounding_theta=theta_array,
                                                        sounding_datenums=datenum_array, sounding_qflags=qflag_array,
                                                        geos_files=geos_files, nprocs=nprocs, prior_flags=prior_flags,
                                                        eqlat_pickle_dir=interp_pickle_dir,
                                                        error_handler=error_handler)

    met_data['el'] = eqlat_array.reshape(orig_shape)
    prior_flags = prior_flags.reshape(flags_orig_shape)
    o2_dmfs = _make_o2_dmf_array(met_data['dates'], fo2_file=fo2_file, auto_update_fo2_file=auto_update_fo2_file)

    # Create the priors
    if cache_strat_lut:
        # If we're not truncating the MLO/SMO data, we can rely on the internal code
        # to determine if we need to regenerate the LUT. If we are truncating the
        # data, then we should force it to regenerate (even though the internal logic
        # should handle that).
        regen_lut = None if truncate_mlo_smo_by is None else True
        save_lut = True
    else:
        regen_lut = True
        save_lut = False

    unit_scales = {'ppm': 1e-6, 'ppb': 1e-9}

    init_prior_h5(output_file, geos_files, met_resampled_file)

    for gas in gases:
        gas_field = '{}_prior'.format(gas)
        gas_mlo_file = fmt_gas_file(mlo_co2_file, gas)
        gas_smo_file = fmt_gas_file(smo_co2_file, gas)

        record_kws = dict(mlo_file=gas_mlo_file, 
                          smo_file=gas_smo_file, 
                          recalculate_strat_lut=regen_lut, 
                          save_strat=save_lut, 
                          truncate_date=truncate_mlo_smo_date,
                          last_date=record_end_date)
        if gas == 'co':
            record_kws = dict()
        record_class = tccon_priors.gas_records[gas]
        gas_record = record_class(**record_kws)
        gas_prior_flags = prior_flags.copy()

        # The keys here define the variable names that will be used in the HDF file. The values define the corresponding
        # keys in the output dictionaries from tccon_priors.generate_single_tccon_prior.
        var_mapping = {gas_field: gas_record.gas_name, 'gas_record_latency': 'mean_latency', 'equivalent_latitude': 'EqL',
                       'gas_record_date': 'gas_date', 'atmospheric_stratum': 'atm_stratum', 'age_of_air': 'strat_age_of_air',
                       'altitude': 'Height', 'pressure': 'Pressure'}
        # This dictionary defines extra type information to create the output arrays. _make_output_profiles_dict uses it.
        # The keys should match those in var_mapping; any key from var_mapping that isn't in this one gets the default
        # output array (with shape orig_shape and fill value np.nan). Each value in this dict must be a two-element tuple;
        # the first is the desired shape, the second the fill value (which also sets the type). Any values of -1 in the
        # shape get replaced with the corresponding value from orig_shape.
        var_type_info = {'gas_record_date': (orig_shape, None)}

        if nprocs == 0:
            profiles, units = _prior_serial(orig_shape=orig_shape, var_mapping=var_mapping, var_type_info=var_type_info,
                                            met_data=met_data, gas_record=gas_record, prior_flags=gas_prior_flags,
                                            use_trop_eqlat=use_trop_eqlat, fo2_file=fo2_file, error_handler=error_handler)
        else:
            profiles, units = _prior_parallel(orig_shape=orig_shape, var_mapping=var_mapping, var_type_info=var_type_info,
                                              met_data=met_data, gas_record=gas_record, prior_flags=gas_prior_flags, nprocs=nprocs,
                                              use_trop_eqlat=use_trop_eqlat, fo2_file=fo2_file, error_handler=error_handler)

        # Add latitude, longitude, and flags to the priors file
        profiles['sounding_longitude'] = met_data['longitude']
        units['sounding_longitude'] = 'degrees_east'
        profiles['sounding_latitude'] = met_data['latitude']
        units['sounding_latitude'] = 'degrees_north'
        profiles['prior_failure_flags'] = gas_prior_flags
        units['prior_failure_flags'] = error_handler.get_error_descriptions()

        # Convert the from ppm/ppb to dry mole fraction
        profiles[gas_field] *= unit_scales[units[gas_field]]
        units[gas_field] = 'dmf'

        # Also need to convert the entry dates into decimal years to write to HDF
        gas_date_dec_years = [mod_utils.date_to_decimal_year(d) for d in profiles['gas_record_date'].flat]
        profiles['gas_record_date'] = np.array(gas_date_dec_years).reshape(profiles['gas_record_date'].shape)
        units['gas_record_date'] = 'Date as decimal year (decimal part = 0-based day-of-year / {})'.format(mod_constants.days_per_year)

        # And convert the stratum to a short integer and update the unit to be more descriptive
        profiles['atmospheric_stratum'] = profiles['atmospheric_stratum'].astype(np.uint8)
        units['atmospheric_stratum'] = 'flag (1 = troposphere, 2 = middleworld, 3 = overworld)'

        # If running for GOSAT, we have an extra dimension between the exposure and level which is just 1 long and was only
        # a placeholder to provide compatibility with the loops over sounding group/sounding for OCO. Remove those singleton
        # dimensions to keep the GOSAT files clean
        if instrument == 'gosat':
            for key, value in profiles.items():
                profiles[key] = value.squeeze()
                logger.debug('GOSAT array "{}" squeezed from {} to {}'.format(key, value.shape, profiles[key].shape))

        # Write the priors to the file requested.
        write_o2_h5(output_file, o2_dmfs)
        write_prior_h5(output_file, profiles, units, prior_group=prior_group.format(gas))
        if gas != 'co':
            insitu_record_to_h5_group(output_file, gas_record.conc_seasonal, gas, record_group=record_group.format(gas))



def _prior_helper(i_sounding, i_foot, qflag, mod_data, gas_record, var_mapping, var_type_info, use_trop_eqlat=False,
                  fo2_file=fo2_prep.DEFAULT_FO2_FILE, prior_flags=None, error_handler=_def_errh):
    """
    Underlying function that generates individual prior profiles in serial and parallel mode

    :param i_sounding: the sounding group index (first dimension) within the data array that's being generated.
    :type i_sounding: int

    :param i_foot: the footprint index (second dimension).
    :type i_foot: int

    :param qflag: the integer quality flag for this sounding
    :type qflag: int

    :param mod_data: the dictionary representing the profile's met data, structure as a model data dictionary for the
     TCCON code.
    :type mod_data: dict

    :param gas_record: the MLO/SMO record class instance
    :type gas_record: :class:`ggg_inputs.priors.tccon_priors.MloSmoTraceGasRecord`

    :param var_mapping: a dictionary mapping the TCCON variable names in the output dict from the TCCON code to the
     variable names desired for the ACOS code. The ACOS names are the keys, the TCCON names the values.
    :type var_mapping: dict

    :param var_type_info: a dictionary with extra information about the type of the output arrays. See
     :func:`_make_output_profiles_dict` for a full description.
    :type var_type_info: dict

    :return: dictionaries of profiles (and ancillary data) and units. Profiles will hold the data arrays of the ACOS
     variables and units will hold strings defining the units of those arrays, as returned by the TCCON code. Both will
     have the same keys as ``var_mapping``
    :rtype: dict, dict
    """
    obs_date = mod_data['file']['datetime']

    if i_foot == 0:
        logger.info('Processing set of soundings {}'.format(i_sounding + 1))

    profiles = dict()
    for k in var_mapping.keys():
        fill_val = var_type_info[k][1] if k in var_type_info else np.nan
        profiles[k] = np.full(mod_data['profile']['Height'].shape, fill_val)
    units = {k: '' for k in var_mapping.keys()}

    if qflag != 0:
        logger.info('Quality flag != 0 for sounding group/footprint {}/{}. Skipping prior calculation'
                    .format(i_sounding + 1, i_foot + 1))
        error_handler.set_flag(err_code_name='met_qual_flag', flags=prior_flags, inds=(i_sounding, i_foot))
        return profiles, None, prior_flags[i_sounding, i_foot]
    elif prior_flags is not None and prior_flags[i_sounding, i_foot] != 0:
        logger.info('Prior flag != 0 for sounding group/footprint {}/{}. Skipping prior calculation'
                    .format(i_sounding + 1, i_foot + 1))
        return profiles, None, prior_flags[i_sounding, i_foot]
    elif obs_date < dt.datetime(1993, 1, 1):
        # In the test met file I was given, one set of soundings had a date set to 20 Dec 1992, while the rest
        # where on 14 May 2017. Since the 1992 date is close to 999999 seconds before 1 Jan 1993 and the dates
        # are in TAI93 time, I assume these are fill values.
        logger.important('Date before 1993 ({}) found, assuming this is a fill value and skipping '
                         '(sounding group/footprint {}/{})'.format(obs_date, i_sounding + 1, i_foot + 1))
        error_handler.set_flag(err_code_name='out_of_range_date', flags=prior_flags, inds=(i_sounding, i_foot))
        return profiles, None, prior_flags[i_sounding, i_foot]
    elif np.all(np.isnan(mod_data['profile']['Height'])):
        logger.important('Profile at sounding group/footprint {}/{} is all fill values, not calculating prior'
                         .format(i_sounding + 1, i_foot + 1))
        error_handler.set_flag(err_code_name='no_data', flags=prior_flags, inds=(i_sounding, i_foot))
        return profiles, None, prior_flags[i_sounding, i_foot]

    try:
        # 2021-11-08: adjusting the altitude grid caused undesired behavior in granule 210902202558s.
        # Specifially, altitude grids in one part of the world had their first level set to altitude = 0,
        # which significantly overestimated the seasonal drawdown. The simple fix for now is to turn off
        # that adjustment.
        priors_dict, priors_units, priors_constants = tccon_priors.generate_single_tccon_prior(
            mod_data, dt.timedelta(hours=0), gas_record, use_eqlat_trop=use_trop_eqlat, use_adjusted_zgrid=False,
            o2_mole_fraction_file=fo2_file
        )
    except Exception as err:
        new_err = err.__class__(err.args[0] + ' Occurred at sounding = {}, footprint = {}'.format(i_sounding+1, i_foot+1))
        error_handler.handle_err(new_err, err_code_name='prior_failure', flags=prior_flags, inds=(i_sounding, i_foot))
        return profiles, None, prior_flags[i_sounding, i_foot]

    for h5_var, tccon_var in var_mapping.items():
        # The TCCON code returns profiles ordered surface-to-space. ACOS expects space-to-surface
        profiles[h5_var] = np.flipud(priors_dict[tccon_var])
        # Yes this will get set every time but only needs set once. Can optimize later if it's slow.
        units[h5_var] = priors_units[tccon_var]

    return profiles, units, prior_flags[i_sounding, i_foot]


def _make_o2_dmf_array(datetimes: np.array, fo2_file: os.PathLike = fo2_prep.DEFAULT_FO2_FILE, auto_update_fo2_file: bool = False) -> np.ndarray:
    output_shape = datetimes.shape
    datetimes = pd.to_datetime(datetimes.ravel())
    o2_rec = tccon_priors.O2MeanMoleFractionRecord(
        o2_mole_fraction_file=fo2_file,
        auto_update_fo2_file=auto_update_fo2_file,
    )
    # Timestamp fill values seem to be in 1992 (-999999 from 1 Jan 1993 in seconds probably)
    tt = datetimes > pd.Timestamp(1993,1,1)
    o2_dmfs = np.full(datetimes.size, np.nan)
    o2_dmfs[tt] = o2_rec.get_many_o2_mole_fractions(datetimes[tt])
    return o2_dmfs.reshape(output_shape)


def _make_output_profiles_dict(orig_shape, var_mapping, var_type_info):
    """
    Initialize the dictionary to store the output ACOS variables.

    :param orig_shape: The original shape of the 3D ACOS variables (sounding group, footprints, levels).
    :type orig_shape: tuple(int)

    :param var_mapping: a dictionary mapping the TCCON variable names in the output dict from the TCCON code to the
     variable names desired for the ACOS code. The ACOS names are the keys, the TCCON names the values.
    :type var_mapping: dict

    :param var_type_info: a dictionary with extra information about the type of the output arrays. Its keys must also
     be keys in ``var_mapping``. Each value is a two element tuple:

        * The first element must be a tuple or list that describes the desired shape of the output array for this
          variable. Any elements that are -1 are replaced by the corresponding element in ``orig_shape``.
        * The second element must be the fill value to initialize the array with. This will also implicitly set the
          array's data type.

    .. caution::
       At present, the shapes defined in ``var_type_info`` may have fewer dimensions than ``orig_shape`` but not more.
       Attempting to pass such a shape will result in a ``NotImplementedError`` being raised. This has not been
       implemented simply because it has not yet been necessary.

    :type var_type_info: dict

    :return: the profiles dictionary, with arrays initialized with their respective fill values.
    :rtype: dict
    """
    prof_dict = dict()
    unit_dict = dict()
    for k in var_mapping:
        if k not in var_type_info:
            shape = orig_shape
            fill_val = np.nan
        else:
            new_shape, fill_val = var_type_info[k]
            if len(new_shape) > len(orig_shape):
                raise NotImplementedError('Shapes in var_type_info with more dimensions that in orig_shape are not '
                                          'yet implemented.')
            shape = [o if n == -1 else n for o, n in zip(orig_shape, new_shape)]

        prof_dict[k] = np.full(shape, fill_val)
        unit_dict[k] = ''
    return prof_dict, unit_dict


def _prior_serial(orig_shape, var_mapping, var_type_info, met_data, gas_record, prior_flags=None, use_trop_eqlat=False,
                  fo2_file=fo2_prep.DEFAULT_FO2_FILE, error_handler=_def_errh):
    """
    Generate the priors, running in serial mode.

    See :func:`_prior_helper` for help on all other inputs not listed here.

    :param met_data: the dictionary of met data read from the resampler .h5 file with equivalent latitude added as the
     "el" variable.
    :type met_data: dict

    :return: profiles and units dictionaries; profiles contains the actual data, units strings describing the units of
     each array.
    :rtype: dict, dict
    """
    profiles, units = _make_output_profiles_dict(orig_shape, var_mapping, var_type_info)
    units_set = False

    for i_sounding in range(orig_shape[0]):
        for i_foot in range(orig_shape[1]):

            mod_data = _construct_mod_dict(met_data, i_sounding, i_foot)
            qflag = met_data['quality_flags'][i_sounding, i_foot]

            this_profiles, this_units, _ = _prior_helper(i_sounding, i_foot, qflag, mod_data, gas_record,
                                                         var_mapping, var_type_info, prior_flags=prior_flags,
                                                         use_trop_eqlat=use_trop_eqlat, fo2_file=fo2_file, error_handler=error_handler)
            for h5_var, h5_array in profiles.items():
                h5_array[i_sounding, i_foot, :] = this_profiles[h5_var]
            if not units_set and this_units is not None:
                units = this_units
                units_set = True

    return profiles, units


def _prior_parallel(orig_shape, var_mapping, var_type_info, met_data, gas_record, nprocs, prior_flags=None,
                    fo2_file=fo2_prep.DEFAULT_FO2_FILE, use_trop_eqlat=False, error_handler=_def_errh):
    """
    Generate the priors, running in parallel mode.

    See :func:`_prior_helper` for help on all other inputs not listed here.

    :param met_data: the dictionary of met data read from the resampler .h5 file with equivalent latitude added as the
     "el" variable.
    :type met_data: dict

    :param nprocs: the number of processors to use to run the code.
    :type nprocs: int

    :return: profiles and units dictionaries; profiles contains the actual data, units strings describing the units of
     each array.
    :rtype: dict, dict
    """
    logger.info('Running {} prior calculation in parallel with {} processes'.format(gas_record.gas_name.upper(), nprocs))

    # Need to prepare iterators of the sounding and footprint indices, as well as the individual met dictionaries
    # and observation dates. We only want to pass the individual dictionary and date to each worker, not the whole
    # met data, because that would probably be slow due to overhead. (Not tested however.)
    sounding_inds, footprint_inds = [x for x in zip(*product(range(orig_shape[0]), range(orig_shape[1])))]
    mod_dicts = map(_construct_mod_dict, repeat(met_data), sounding_inds, footprint_inds)
    qflags = [met_data['quality_flags'][isound, ifoot] for isound, ifoot in zip(sounding_inds, footprint_inds)]

    with Pool(processes=nprocs) as pool:
        result = pool.starmap(_prior_helper, zip(sounding_inds, footprint_inds, qflags, mod_dicts,
                                                 repeat(gas_record), repeat(var_mapping), repeat(var_type_info),
                                                 repeat(use_trop_eqlat), repeat(fo2_file), repeat(prior_flags),
                                                 repeat(error_handler)))

    # At this point, result will be a list of tuples of pairs of dicts, the first dict the profiles dict, the second
    # the units dict or None if the prior calculation did not run. We need to combine the profiles into one array per
    # variable and get one valid units dict
    profiles, units = _make_output_profiles_dict(orig_shape, var_mapping, var_type_info)
    units_set = False
    for (these_profs, these_units, retflag), i_sounding, i_foot in zip(result, sounding_inds, footprint_inds):
        prior_flags[i_sounding, i_foot] = retflag
        if not units_set and these_units is not None:
            units = these_units
            units_set = True
        for h5var, h5array in profiles.items():
            h5array[i_sounding, i_foot, :] = these_profs[h5var]

    return profiles, units


def compute_sounding_equivalent_latitudes(sounding_pv, sounding_theta, sounding_datenums, sounding_qflags, geos_files,
                                          nprocs=0, prior_flags=None, eqlat_pickle_dir='.', error_handler=_def_errh):
    """
    Compute equivalent latitudes for a collection of OCO soundings

    :param sounding_pv: potential vorticity in units of PVU (1e-6 K * m2 * kg^-1 * s^-1). Must be an array with
     dimensions (profiles, levels). That is, if the data read from the resampled met files have dimensions (soundings,
     footprints, levels), these must be reshaped so that the first two dimensions get collapsed into one.
    :type sounding_pv: :class:`numpy.ndarray`

    :param sounding_theta: potential temperature in units of K. Same shape as ``sounding_pv`` required.
    :type sounding_theta: :class:`numpy.ndarray`

    :param sounding_datenums: date and time of each profile as a date number (a numpy :class:`~numpy.datetime64` value
     converted to a float type, see :func:`datetime2datenum` in this module). Must be a vector with length equal to the
     first dimension of ``sounding_pv``.
    :type sounding_datenums: :class:`numpy.ndarray`

    :param geos_files: a list of the GEOS 3D met files that bracket the times of all the soundings. Need not be absolute
     paths, but must be paths that resolve correctly from the current working directory.
    :type geos_files: list(str)

    :param nprocs: number of processors to use to compute the equivalent latitudes. 0 will run in serial mode, anything
     greater will use parallel mode.
    :type nprocs: int

    :param prior_flags: an integer array used to store numeric codes indicating why a particular prior failed to
     generate.
    :type prior_flags: :class:`numpy.ndarray`

    :param error_handler: an ErrorHandler instance that determines how errors during the eq. lat. computation are caught
     and handled.
    :type error_handler: :class:`ErrorHandler`

    :return: an array of equivalent latitudes with dimensions (profiles, levels)
    :rtype: :class:`numpy.ndarray`
    """
    # Create interpolators for each of the GEOS FP files provided. The resulting dictionary will have the files'
    # datetimes as keys
    geos_utc_times = [mod_utils.datetime_from_geos_filename(f) for f in geos_files]
    geos_datenums = np.array([datetime2datenum(d) for d in geos_utc_times])

    on_native_grid = [mod_utils.is_geos_on_native_grid(f) for f in geos_files]
    if all(on_native_grid):
        eqlat_fxns = mod_maker.equivalent_latitude_functions_from_native_geos_files(geos_files, geos_utc_times)
    elif not any(on_native_grid):
        eqlat_fxns = mod_maker.equivalent_latitude_functions_from_geos_files(geos_files, geos_utc_times)
    else:
        raise RuntimeError('Received a mixture of GEOS files on native 72 level grid and non-native grid. This '
                           'is not supported.')
    # it will be easier to work with this as a list of the interpolators in the right order.
    eqlat_fxns = [eqlat_fxns[k] for k in geos_utc_times]

    # This part is going to be slow. We need to use the interpolators to get equivalent latitude profiles for each
    # sounding for the two times on either side of the sounding time, then do a further linear interpolation to
    # the actual sounding time.

    if nprocs == 0:
        return _eqlat_serial(sounding_pv, sounding_theta, sounding_datenums, sounding_qflags, geos_datenums, eqlat_fxns,
                             prior_flags=prior_flags, error_handler=error_handler)
    else:
        return _eqlat_parallel(sounding_pv, sounding_theta, sounding_datenums, sounding_qflags, geos_datenums, eqlat_fxns, 
                               prior_flags=prior_flags, error_handler=error_handler, nprocs=nprocs, eqlat_pickle_dir=eqlat_pickle_dir)


def _eqlat_helper(idx, pv_vec, theta_vec, datenum, quality_flag, eqlat_fxns, geos_datenums, prior_flags=None,
                  error_handler=_def_errh):
    """
    Internal function that carries out the equivalent latitude calculation while running in either serial or parallel

    :param idx: the sounding index (assuming the 3D met arrays were reshaped to soundings-by-levels)
    :type idx: int

    :param pv_vec: the potential vorticity vector for this sounding, in PVU (10^-6 K m^2 kg^-1 s^-1)
    :type pv_vec: 1D :class:`numpy.ndarray`

    :param theta_vec: the potential temperature vector for this sounding, in K
    :type theta_vec: 1D :class:`numpy.ndarray`

    :param datenum: a numeric representation of the date/time of this sounding, typically created by
     :func:`datetime2datenum`.
    :type datenum: int or float

    :param quality_flag: a scalar number that is the quality flag for this profile.
    :type quality_flag: int

    :param eqlat_fxns: the collection of equivalent latitude interpolators, must be in the same order as
     ``geos_datenums``.
    :type eqlat_fxns: list(:class:`scipy.interpolate.interpolate.interp2d`)

    :param geos_datenums: the date numbers (see ``datenum``) for the GEOS FP files that bracket this sounding. Should
     be >= 2 and must be ordered the same as ``eqlat_fxns``, so that ``eqlat_fxns[0]`` the the equivalent latitude
     interpolator for ``geos_datenums[0]`` and so on.
    :type geos_datenums: list(float)

    :return: the equivalent latitude profile, interpolated in time and space to the sounding
    :rtype: :class:`numpy.ndarray`
    """
    default_return = np.full_like(pv_vec, np.nan)

    if quality_flag != 0:
        logger.info('Sounding {}: quality flag != 0. Skipping eq. lat. calculation.'.format(idx))
        error_handler.set_flag(err_code_name='met_qual_flag', flags=prior_flags, inds=idx)
        return default_return, prior_flags[idx]
    elif prior_flags is not None and prior_flags[idx] != 0:
        logger.info('Sounding {}: prior flag != 0. Skipping eq. lat. calculation.'.format(idx))
        return default_return, prior_flags[idx]

    # Because of a limit on the size of objects that can be sent between threads in Python 3.6,
    # we need a workaround for big interpolators. If we got strings/paths, then we had to write
    # those interpolators to files and pass the paths to those files instead. This block can
    # be removed when we no longer need to support Python 3.6.
    eqlat_fxns_in = eqlat_fxns
    eqlat_fxns = []
    for fxn in eqlat_fxns_in:
        if isinstance(fxn, (str, Path)):
            eqlat_fxns.append(_eqlat_pickle_load_cached(fxn))
        else:
            eqlat_fxns.append(fxn)

    logger.debug('Calculating eq. lat. {}'.format(idx))
    try:
        i_last_geos = _find_helper(geos_datenums <= datenum, order='last')
        i_next_geos = _find_helper(geos_datenums > datenum, order='first')
    except IndexError as err:
        logger.important('Sounding {}: could not find GEOS file by time. Assuming fill value for time'.format(idx))
        error_handler.set_flag(err_code_name='cannot_find_geos', flags=prior_flags, inds=idx)
        return default_return, prior_flags[idx]

    try:
        last_el_profile = _make_el_profile(pv_vec, theta_vec, eqlat_fxns[i_last_geos])
        next_el_profile = _make_el_profile(pv_vec, theta_vec, eqlat_fxns[i_next_geos])
    except Exception as err:
        error_handler.handle_err(err, err_code_name='eqlat_failure', flags=prior_flags, inds=idx)
        return default_return, prior_flags[idx]

    # Interpolate between the two times by calculating a weighted average of the two profiles based on the sounding
    # time. This avoids another for loop over all levels.
    weight = time_weight(datenum, geos_datenums[i_last_geos], geos_datenums[i_next_geos])
    # Need to return the flags for parallel mode
    return weight * last_el_profile + (1 - weight) * next_el_profile, prior_flags[idx]


def _eqlat_clip(el):
    """
    Restrict equivalent latitude to [-90, 90]

    :param el: the equivalent latitude array.
    :type el: :class:`numpy.ndarray`

    :return: None. Modifies ``el`` in-place.
    """
    def trim(bool_ind, replacement_val):
        if np.any(bool_ind):
            # nanmax can't take an empty array so we have to make sure that
            # bool_ind has at least one true value
            maxdiff = np.nanmax(np.abs(el[bool_ind] - replacement_val))
            el[bool_ind] = replacement_val
            return maxdiff
        else:
            return 0.0

    xx1 = el < -90.0
    xx2 = el > 90.0

    max_below = trim(xx1, -90.0)
    max_above = trim(xx2, 90.0)
    n_outside = xx1.sum() + xx2.sum()

    if n_outside > 0:
        logger.warning('{} equivalent latitudes were outside the range [-90,90] (max difference {}). They have been clipped to [-90,90].'
                       .format(n_outside, max(max_below, max_above))) 


def _eqlat_serial(sounding_pv, sounding_theta, sounding_datenums, sounding_qflags, geos_datenums, eqlat_fxns,
                  prior_flags=None, error_handler=_def_errh):
    """
    Calculate equivalent latitude running in serial mode.

    :param sounding_pv: the array of potential vorticity in PVU (10^-6 K m^2 kg^-1 s^-1). Must have dimensions
     soundings-by-levels.
    :type sounding_pv: :class:`numpy.ndarray`

    :param sounding_theta: the array of potential temperature in K. Must have same dimensions as ``sounding_theta``.
    :type sounding_theta: :class:`numpy.ndarray`

    :param sounding_datenums: the vector of date numbers for each sounding. Must be 1D with length equal to the number
     of soundings. Datenumbers may be any representation of date as an in or float, as long as it is consistent with
     ``geos_datenums``, however typically this is the result of :func:`datetime2datenum`.
    :type sounding_datenums: 1D :class:`numpy.ndarray` or equivalent

    :param geos_datenums: a vector of date numbers corresponding to the GEOS files that provided the ``eqlat_fxns``.
     Must have the same order as ``eqlat_fxns``.
    :type geos_datenums: 1D :class:`numpy.ndarray` or equivalent.

    :param eqlat_fxns: a list of equivalent latitude interpolators for the date/times specified by ``geos_datenums``.
    :type eqlat_fxns: list(:class:`scipy.interpolate.interpolate.interp2d`)

    :return: an array of equivalent latitudes for the soundings (dimensions soundings-by-levels).
    :rtype: :class:`numpy.ndarray`
    """
    sounding_eqlat = np.full_like(sounding_pv, np.nan)

    # This part is going to be slow. We need to use the interpolators to get equivalent latitude profiles for each
    # sounding for the two times on either side of the sounding time, then do a further linear interpolation to
    # the actual sounding time.
    logger.info('Running eq. lat. calculation in serial')
    for idx, (pv_vec, theta_vec, datenum, qflag) in enumerate(zip(sounding_pv, sounding_theta, sounding_datenums, sounding_qflags)):
        sounding_eqlat[idx], _ = _eqlat_helper(idx, pv_vec, theta_vec, datenum, qflag, eqlat_fxns, geos_datenums,
                                               prior_flags=prior_flags, error_handler=error_handler)

    _eqlat_clip(sounding_eqlat)
    return sounding_eqlat


def _eqlat_parallel(sounding_pv, sounding_theta, sounding_datenums, sounding_qflags, geos_datenums, eqlat_fxns, nprocs,
                    prior_flags=None, error_handler=_def_errh, eqlat_pickle_dir='.'):
    """
    Calculate equivalent latitude running in parallel mode.

    :param nprocs: the number of processors to use.
    :type nprocs: int

    See :func:`_eqlat_serial` for the other parameters and return type.
    """
    logger.info('Running eq. lat. calculation in parallel with {} processes'.format(nprocs))
    with Pool(processes=nprocs) as pool, _eqlat_pickle_manager(eqlat_fxns, eqlat_pickle_dir) as eqlat_pickles:
        result = pool.starmap(_eqlat_helper, zip(range(sounding_pv.shape[0]), sounding_pv, sounding_theta,
                                                 sounding_datenums, sounding_qflags,
                                                 repeat(eqlat_pickles), repeat(geos_datenums), repeat(prior_flags),
                                                 repeat(error_handler)))

    eqlats, flags = zip(*result)
    sounding_eqlat = np.array(eqlats)
    _eqlat_clip(sounding_eqlat)
    # Copy the flag values from each parallel call into the array
    for i, flag in enumerate(flags):
        prior_flags[i] = flag
    return sounding_eqlat


@contextmanager
def _eqlat_pickle_manager(eqlat_fxns, pickle_dir='.'):
    if sys.version_info.major <= 3 and sys.version_info.minor >= 10:
        logger.info("Running under Python 3.10 or greater, assuming support for large inter-thread objects. Not writing pickle files for eqlat interpolators.")
        yield eqlat_fxns
        return 

    logger.info("Running under Python 3.9 or less, not assuming there is support for large inter-thread objects. Writing pickle files for eqlat interpolators.")
    mypid = os.getpid()
    eqlat_pickle_files = []

    for i, fxn in enumerate(eqlat_fxns):
        pickle_file = os.path.join(pickle_dir, f'ginput_eqlat_interpolator_pickle.{mypid}-{i}.pkl')
        if os.path.exists(pickle_file):
            raise IOError(f'Cannot create eqlat interpolator pickle file {pickle_file}, file already exists')
        eqlat_pickle_files.append(pickle_file)
        with open(pickle_file, 'wb') as f:
            pickle.dump(fxn, f)
    file_basenames = ', '.join(os.path.basename(fname) for fname in eqlat_pickle_files)
    logger.info(f'Created {len(eqlat_pickle_files)} pickle files in {os.path.abspath(pickle_dir)}: {file_basenames}')

    try:
        yield eqlat_pickle_files
    finally:
        for pickle_file in eqlat_pickle_files:
            if os.path.exists(pickle_file):
                os.remove(pickle_file)
                logger.debug(f'Removed eqlat interpolator pickle file {os.path.abspath(pickle_file)}')
        logger.info(f'Removed {len(eqlat_pickle_files)} temporary pickle files')


@lru_cache(maxsize=4)
def _eqlat_pickle_load_cached(pickle_file):
    # It's important to have this as a cached function rather than loading the pickle directly in _eqlat_helper.
    # If we do that, then it needs to load the pickles for each sounding it operates on, which is slow.
    # At least in my testing, using this cache function means that the first sounding in a given thread will
    # load the pickle, but then it is cached, so all the remaining soundings can access it directly from memory.
    logger.debug(f'Loading eqlat interpolator from file {pickle_file}')
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def read_oco_resampled_met(met_file, error_handler=_def_errh):
    met_group = 'Meteorology'
    sounding_group = 'SoundingGeometry'
    var_dict = {'pv': [met_group, 'epv_profile_met'],
                'temperature': [met_group, 'temperature_profile_met'],
                'pressure': [met_group, 'vector_pressure_levels_met'],
                'date_strings': [sounding_group, 'sounding_time_string'],
                'altitude': [met_group, 'height_profile_met'],
                'latitude': [sounding_group, 'sounding_latitude'],
                'longitude': [sounding_group, 'sounding_longitude'],
                'trop_pressure': [met_group, 'blended_tropopause_pressure_met'],
                'trop_temperature': [met_group, 'tropopause_temperature_met'],
                'surf_gph': [met_group, 'gph_met'],
                'quality_flags': [sounding_group, 'sounding_qual_flag']
                }

    return read_resampled_met(met_file, var_dict, error_handler=error_handler)


def read_gosat_resampled_met(met_file, error_handler=_def_errh):
    met_group = 'Meteorology'
    sounding_group = 'SoundingGeometry'
    sounding_header = 'SoundingHeader'
    var_dict = {'pv': [met_group, 'epv_profile_met'],
                'temperature': [met_group, 'temperature_profile_met'],
                'pressure': [met_group, 'vector_pressure_levels_met'],
                'date_strings': [sounding_header, 'sounding_time_string'],
                'altitude': [met_group, 'height_profile_met'],
                'latitude': [sounding_group, 'sounding_latitude'],
                'longitude': [sounding_group, 'sounding_longitude'],
                'trop_pressure': [met_group, 'blended_tropopause_pressure_met'],
                'trop_temperature': [met_group, 'tropopause_temperature_met'],
                'surf_gph': [met_group, 'gph_met'],
                'quality_flags': [sounding_header, 'sounding_qual_flag']
                }

    # To be compatible with the OCO code, the arrays need to have at most 3 dimensions: sounding group, footprint,
    # level. GOSAT arrays have exposure, band, polarization, and level. To ensure compatibility, make them have
    # [exposure, 1, level] or [exposure, 1] if they should be 2D.
    data, flags = read_resampled_met(met_file, var_dict, error_handler=error_handler)
    for name, arr in data.items():
        data[name] = _gosat_normalize_shape(arr, name)
    flags = _gosat_normalize_shape(flags, 'prior_flags')
    return data, flags


def read_geocarb_resampled_met(met_file, error_handler=_def_errh):
    met_group = 'Meteorology'
    sounding_group = 'SoundingGeometry'
    var_dict = {'pv': [met_group, 'epv_profile_met'],
                'temperature': [met_group, 'temperature_profile_met'],
                'pressure': [met_group, 'vector_pressure_levels_met'],
                'date_strings': [sounding_group, 'sounding_time_string'],
                'altitude': [met_group, 'height_profile_met'],
                'latitude': [sounding_group, 'sounding_latitude'],
                'longitude': [sounding_group, 'sounding_longitude'],
                'trop_pressure': [met_group, 'blended_tropopause_pressure_met'],
                'trop_temperature': [met_group, 'tropopause_temperature_met'],
                'surf_gph': [met_group, 'gph_met'],
                'co': [met_group, 'co_profile_met'],
                'quality_flags': [sounding_group, 'sounding_qual_flag']
                }

    return read_resampled_met(met_file, var_dict, error_handler=error_handler)


def _gosat_normalize_shape(arr, name):
    def check_band_pol(a):
        diffs = np.diff(a, axis=1)
        reldiffs = np.abs(diffs / arr[0, 0:1])
        reldiffs[np.isnan(reldiffs)] = 0
        max_allowed_rdiff = 1e-6
        if np.any(reldiffs) > max_allowed_rdiff:
            logger.warning('Relative differences in "{}" exceed 1 part in {} along the band/polarization '
                           'dimensions (max relative difference = {}).'.format(name, 1/max_allowed_rdiff, np.max(reldiffs)))
    nlev = 72
    nband = 3
    npol = 2
    if arr.ndim == 4:
        if arr.shape[1:] != (nband, npol, nlev):
            raise NotImplementedError('4D GOSAT array "{}" does not appear to have dimensions '
                                      'exposure x band x polarization x level'.format(name))

        logger.debug('Collapsing "{}" from {} to ({}, 1, {})'.format(name, arr.shape, arr.shape[0], nlev))
        arr = arr.reshape(-1, nband*npol, nlev)
        check_band_pol(arr)
        return arr[:, 0:1, :]

    elif arr.ndim == 3:
        if arr.shape[1:] != (nband, npol):
            raise NotImplementedError('3D GOSAT array "{}" does not appear to have dimensions '
                                      'exposure x band x polarization'.format(name))

        logger.debug('Collapsing "{}" from {} to ({}, 1)'.format(name, arr.shape, arr.shape[0]))
        arr = arr.reshape(-1, nband*npol)
        check_band_pol(arr)
        return arr[:, 0:1]

    elif arr.ndim == 1:
        logger.debug('Expanding "{}" from {} to ({}, 1)'.format(name, arr.shape, arr.shape[0]))
        return arr.reshape(-1, 1)

    else:
        raise NotImplementedError('Do not know how to handle a {}D GOSAT array'.format(arr.ndim))


def read_resampled_met(met_file, var_dict, error_handler=_def_errh):
    """
    Read the required data from the HDF5 file containing the resampled met data.

    :param met_file: the path to the met file
    :type met_file: str

    :param var_dict: a dictionary mapping the output variables to the datasets in the HDF5 file. Keys must be the
     output variables (listed below) and the values must be two-element lists, the first element gives the group name
     and the second the dataset name within that group to read.
    :type var_dict: dict

    :return: a dictionary with variables both read directly from the met file and derived from those values. Keys are:

        * "pv" - potential vorticity in PVU
        * "theta" - potential temperature in K
        * "temperature" - temperature profiles in K
        * "pressure" - pressure profiles in hPa
        * "date_strings" - the sounding date/time as a string
        * "dates" - the sounding date/time as a Python :class:`datetime.datetime` object
        * "datenums" - the sounding date/time as a floating point number (see :func:`datetime2datenum`)
        * "altitude" - the altitude profiles in km
        * "latitude" - the sounding latitudes in degrees (south is negative)
        * "longitude" - the sounding longitudes in degrees (west is negative)
        * "trop_pressure" - the blended tropopause pressure in hPa
        * "trop_temperature" - the blended tropopause temperature in K
        * "surf_alt" - the surface altitude, derived from surface geopotential, in km
        * "quality_flags" - the sounding quality flag. Equal to 0 for good soundings.

    :rtype: dict
    """
    data_dict = dict()
    with h5py.File(met_file, 'r') as h5obj:
        for out_var, (group_name, var_name) in var_dict.items():
            logger.debug('Reading {}/{}'.format(group_name, var_name))
            tmp_data = h5obj[group_name][var_name][:]
            if np.issubdtype(tmp_data.dtype, np.floating):
                tmp_data[tmp_data < _fill_val_threshold] = np.nan
            data_dict[out_var] = tmp_data

    # Potential temperature needs to be calculated, the date strings need to be converted, and the potential temperature
    # needs scaled to units of PVU

    # pressure in the met file is in Pa, need hPa for the potential temperature calculation
    flags = np.zeros(data_dict['quality_flags'].shape, dtype=np.int16)
    data_dict['pressure'] *= 0.01  # convert from Pa to hPa
    data_dict['theta'] = mod_utils.calculate_potential_temperature(data_dict['pressure'], data_dict['temperature'])
    data_dict['dates'] = _convert_acos_time_strings(data_dict['date_strings'], format='datetime', flag_array=flags,
                                                    error_handler=error_handler)
    data_dict['datenums'] = _convert_acos_time_strings(data_dict['date_strings'], format='datenum', flag_array=flags,
                                                       error_handler=error_handler)
    data_dict['pv'] *= 1e6

    data_dict['altitude'] *= 1e-3  # in meters, need kilometers
    data_dict['trop_pressure'] *= 1e-2  # in Pa, need hPa

    # surf_gph is height derived from geopotential by divided by g0 = 9.80665 m/s^2 according to Chris O'Dell on
    # 21 May 2019. We can use this as the surface altitude, just need to convert from meters to kilometers.
    data_dict['surf_alt'] = 1e-3 * data_dict.pop('surf_gph')

    return data_dict, flags


def init_prior_h5(output_file, geos_files, resampler_file):
    """
    Initialize the output .h5 file. If it already exists, it will be overwritten.

    Initialize the output .h5 file. If it already exists, it will be overwritten.
    :type output_file: str

    :param geos_files: a list/tuple of GEOS-FP(IT) files from which the equivalent 
     latitude was computed.
    :type geos_files: Sequence[str]

    :param resampler_file: the path to the resampled met file that the regular met
     variables were read from.
    :type resampler_file: str

    :return: none, writes to disk
    """
    with h5py.File(output_file, 'w') as h5obj:
        h5obj.attrs['geos_files'] = ','.join(os.path.abspath(f) for f in geos_files)
        h5obj.attrs['resampler_file'] = os.path.abspath(resampler_file)
        h5obj.attrs['ginput_version'] = __version__
        h5obj.attrs['interface_version'] = __acos_int_version__


def write_prior_h5(output_file, profile_variables, units, prior_group='priors'):
    """
    Write the priors to an HDF5 file.

    :param output_file: the path to the output file.
    :type output_file: str

    :param profile_variables: a dictionary containing the variables to write. The keys will be used as the variable
     names.
    :type profile_variables: dict

    :param units: a dictionary defining the units each variable is in. Must have the same keys as ``profile_variables``.
    :type units: dict(str)

    :return: none, writes to file on disk.
    """

    with h5py.File(output_file, 'a') as h5obj:
        h5grp = h5obj.create_group(prior_group)
        for var_name, var_data in profile_variables.items():
            # Replace NaNs with numeric fill values
            if np.issubdtype(var_data.dtype, np.number):
                filled_data = var_data.copy()
                filled_data[np.isnan(filled_data)] = _fill_val
                this_fill_val = _fill_val
            elif np.issubdtype(var_data.dtype, np.string_):
                filled_data = var_data.copy()
                this_fill_val = _string_fill
            else:
                raise NotImplementedError('No method to handle fill values for data type "{}" implemented'.format(var_data.dtype))

            # Write the data
            var_unit = units[var_name]
            dset = h5grp.create_dataset(var_name, data=filled_data, fillvalue=this_fill_val)
            dset.attrs['units'] = var_unit


def write_o2_h5(output_file, o2_dmfs, h5_group='o2_record'):
    with h5py.File(output_file, 'a') as h5obj:
        grp = h5obj.create_group(h5_group)
        dset = grp.create_dataset('o2_global_dmf', data=o2_dmfs, compression='gzip', compression_opts=9, shuffle=True)
        dset.attrs['description'] = ('Per-sounding global mean O2 dry mole fraction calculated from Scripps O2/N2 data and NOAA global mean CO2 data. '
                                     'This will vary slightly over the granule from the interpolation to different times for each sounding. '
                                     'No seasonal cycle is included, only the year-over-year change.')
        dset.attrs['units'] = 'mol mol^-1'
        dset.attrs['note'] = 'Only pixels with a fill value for their time (a time before 1 Jan 1993) will have a NaN, all other receive a value.'

        dset = grp.create_dataset('granule_mean_o2_global_dmf', data=np.nanmean(o2_dmfs))
        dset.attrs['description'] = 'Mean of non-fill values from "o2_global_dmf" for applications that prefer a single value per granule.'
        dset.attrs['units'] = 'mol mol^-1'


def insitu_record_to_h5_group(output_file, df, gas, record_group='mlo_smo_record'):
    def get_unit(col):
        if col == 'dmf_mean':
            return {'co2': 'ppm', 'ch4': 'ppb', 'co': 'ppb'}[gas.lower()]
        if col == 'interp_flag':
            return '0 = not interpolated/extrapolated; 1 = interpolated; 2 = extrapolated.'
        if col == 'latency':
            return 'years'

        return ''

    with h5py.File(output_file, 'a') as h5obj:
        grp = h5obj.create_group(record_group)
        dates = _convert_to_acos_time_strings(df.index.to_pydatetime())
        dset = grp.create_dataset('date', data=dates, fillvalue=_string_fill)
        dset.attrs['description'] = 'Date associated with the MLO/SMO record'


        for colname, column in df.iteritems():
            column = column.to_numpy().copy()
            column[np.isnan(column)] = _fill_val
            dset = grp.create_dataset(colname, data=column, fillvalue=_fill_val)
            dset.attrs['units'] = get_unit(colname)


def _convert_acos_time_strings(time_string_array, format='datetime', flag_array=None, error_handler=_def_errh):
    """
    Convert an array of time strings in format yyyy-mm-ddTHH:MM:SS.dddZ into an array of python datetimes

    :param time_string_array: the array of input strings
    :type: :class:`numpy.ndarray`

    :param format: controls what format the time data are returned in. Options are:

     * "datetime" - returns :class:`datetime.datetime` objects
     * "datenum" - return dates as a linear number. This should be in units of seconds since 1 Jan 1970; however, this
       unit is not guaranteed, so implementations that rely on that unit should be avoided if possible.  See
       :func:`datetime2datenum` for more information.

    :type format: str

    :return: an array, the same size as ``time_string_array``, that contains the times as datetimes.
    """

    # Start with a flat output array for ease of iteration. Reshape to the same shape as the input at the end.
    if format == 'datetime':
        init_val = None
    elif format == 'datenum':
        init_val = np.nan
    else:
        raise NotImplementedError('No initialization value defined for format == "{}"'.format(format))

    output_array = np.full([time_string_array.size], init_val)
    for idx, time_str in enumerate(time_string_array.flat):
        try:
            time_str = time_str.decode('utf8')
            datetime_obj = dt.datetime.strptime(time_str, _acos_tstring_fmt)
            if format == 'datetime':
                output_array[idx] = datetime_obj
            elif format == 'datenum':
                output_array[idx] = datetime2datenum(datetime_obj)
            else:
                raise NotImplementedError('No conversion method defined for format == "{}"'.format(format))
        except Exception as err:
            # Must use ravel to get a flat view into the same memory; flatten returns a copy
            error_handler.handle_err(err, err_code_name='bad_time_str', flags=flag_array.ravel(), inds=idx)

    return np.reshape(output_array, time_string_array.shape)


def _convert_to_acos_time_strings(datetime_array):
    """
    Convert datetimes to ACOS-style time strings

    :param datetime_array: an array of datetime-like objects
    :type datetime_array: :class:`numpy.ndarray`

    :return: an array of datetime strings
    :rtype: :class:`numpy.ndarray`
    """
    # h5py doesn't accept fixed length unicode strings. So we need to convert 
    # the default unicode literal returned by strftime into a bytes string that
    # numpy will interpret as a simple ASCII type.
    datestring_array = np.array([_string_fill if d is None else d.strftime(_acos_tstring_fmt).encode('utf8') for d in datetime_array.flat])
    return datestring_array.reshape(datetime_array.shape)


def _find_helper(bool_vec, order='all'):
    """
    Helper function to find indices of true values within a vector of booleans

    :param bool_vec: the vector to find indices of ``True`` values in.
    :type bool_vec: :class:`numpy.ndarray`

    :param order: a string indicating which indices to return. "all" returns all indices, "first" returns just the
     index of the first true value, and "last" returns the index of the last true value.
    :type order: str

    :return: the index or indices of the true values
    :rtype: int or ndarray(int)
    """
    order = order.lower()
    inds = np.flatnonzero(bool_vec)
    if order == 'last':
        return inds[-1]
    elif order == 'first':
        return inds[0]
    elif order == 'all':
        return inds
    else:
        raise ValueError('"{}" is not an allowed value for order'.format(order))


def _make_el_profile(pv, theta, interpolator):
    """
    Create an equivalent latitude profile from profiles of PV, theta, and one of the eq. lat. intepolators

    This function will create each level of the eq. lat. profile separately. This is safer than calling the interpolator
    with the PV and theta vectors because the latter returns a 2D array of eq. lat., and occasionally the profile is
    not the diagonal of that array. Since I have not figured out under what conditions that happens, I find this
    approach of calculating each level in a loop, safer.

    :param pv: the profile of potential vorticity in PVU (1e-6 K * m2 * kg^-1 * s^-1).
    :type pv: 1D :class:`numpy.ndarray`

    :param theta: the profile of potential temperature in K
    :type theta: 1D :class:`numpy.ndarray`

    :param interpolator: one of the interpolators returned by :func:`mod_utils.equivalent_latitude_functions_from_geos_files`
     that interpolates equivalent latitude to given PV and theta.
    :type interpolator: :class:`scipy.interpolate.interp2d`

    :return: the equivalent latitude profile
    :rtype: 1D :class:`numpy.ndarray`
    """
    el = np.full_like(pv, np.nan)
    for i in range(el.size):
        el[i] = interpolator(pv[i], theta[i])
    return el


def _construct_mod_dict(acos_data_dict, i_sounding, i_foot):
    """
    Create a dictionary akin that mimics a TCCON .mod file from a single sounding's data.

    :param acos_data_dict: A dictionary containing all the data from the ACOS met resample file (or calculated from it)
     necessary for the creation of TCCON CO2 priors. This dictionary must have the following entries:

        * "el" - equivalent latitude profile in degrees
        * "temperature" - temperature profile in K
        * "pressure" - pressure profile in hPa
        * "theta" - potential temperature profile in K
        * "altitude" - altitude profile in km
        * "latitude" - scalar value defining the sounding latitude
        * "trop_temperature" - the scalar temperature of the tropopause in K
        * "trop_pressure" - the scalar pressure at the tropopause in hPa

     Note that this function also assumes that all these arrays have either dimensions (soundings, footprints, levels)
     or (soundings, footprints).
    :type acos_data_dict: dict

    :param i_sounding: the 0-based index for the sounding (the first dimension in the arrays of ``acos_data_dict``).
    :type i_sounding: int

    :param i_foot: the 0-based index for the footprint (the second dimension in the arrays of ``acos_data_dict``).
    :type i_foot: int

    :return: a dictionary suitable for the first argument of :func:`tccon_priors.generate_single_tccon_prior`
    :rtype: dict
    """

    # This list maps the variables names in the acos_data_dict (the keys) to the keys in the mod-like dict. The list
    # must be a list of tuples. In each tuple, the first element defines the key in acos_data_dict. The second element
    # is itself a list where the first element is the group in the mod_dict ("file", "constants", "scalar", or
    # "profile") and the second the actual variable name.
    var_mapping = [('el', ['profile', 'EqL']),
                   ('temperature', ['profile', 'Temperature']),
                   ('pressure', ['profile', 'Pressure']),
                   ('theta', ['profile', 'PT']),
                   ('altitude', ['profile', 'Height']),
                   ('surf_alt', ['scalar', 'Height']),  # need to read in
                   ('latitude', ['constants', 'obs_lat']),
                   ('latitude', ['file', 'lat']),
                   ('longitude', ['file', 'lon']),
                   ('dates', ['file', 'datetime']),
                   ('trop_temperature', ['scalar', 'TROPT']),  # need to read in
                   ('trop_pressure', ['scalar', 'TROPPB']),
                   ('co', ['profile', 'CO'])]  # need to read in

    subgroups = set([l[1][0] for l in var_mapping])
    mod_dict = {k: dict() for k in subgroups}

    for acos_var, (mod_group, mod_var) in var_mapping:
        if acos_var not in acos_data_dict:
            logger.debug('No {} in the sat data, omitting from .mod dict'.format(acos_var))
            continue

        # For 3D vars this slicing will create a vector. For 2D vars, it will create a scalar
        tmp_val = acos_data_dict[acos_var][i_sounding, i_foot]
        # The profile variables need flipped b/c for ACOS they are arranged space-to-surface,
        # but the TCCON code expects surface-to-space
        if mod_group == 'profile':
            tmp_val = np.flipud(tmp_val)
        mod_dict[mod_group][mod_var] = tmp_val
    return mod_dict


def parse_args(parser=None, instrument=None):
    def comma_list(argin):
        return tuple([a.strip() for a in argin.split(',')])

    description = 'Command line interface to generate CO2 priors for the ACOS algorithm'
    if parser is None:
        i_am_main = True
        parser = argparse.ArgumentParser(description=description)
    else:
        i_am_main = False
        parser.description = description

    parser.add_argument('geos_files', type=comma_list, help='Comma-separated list of paths to the GEOS FP or FP-IT '
                                                            'files that cover the times of the soundings. For example, '
                                                            'if the soundings span 0100Z to 0200Z on 2018-01-01, '
                                                            'then the GEOS files for 2018-01-01 0000Z and 0300Z must '
                                                            'be listed. If the soundings span 0230Z to 0330Z, then the '
                                                            'GEOS files for 2018-01-01 0000Z, 0300Z, and 0600Z must '
                                                            'be listed.')
    parser.add_argument('met_resampled_file', help='The path to the HDF5 file containing the GEOS meteorology sampled '
                                                   'at the satellite soundings')
    parser.add_argument('output_file', help='The filename to give the output HDF5 file containing the CO2 profiles and '
                                            'any additional variables. Note that this path will be overwritten without '
                                            'any warning.')
    parser.add_argument('--use-trop-eqlat', action='store_true',
                        help='Turn on using a theta-derived equivalent latitude in the troposphere.')
    parser.add_argument('--cache-strat-lut', action='store_true',
                        help='Give this flag to turn on the ability of the code to cache the stratospheric CO2 lookup '
                             'table rather than recalculating it each time this program is launched. Even when cached, '
                             'the table will be recalculated if the code detects that the dependencies of the table '
                             'have changed. If this is set, then the code must have permission to write to {}'
                             .format(tccon_priors._data_dir))
    parser.add_argument('--mlo-co2-file', default=None, help='Path to the Mauna Loa CO2 monthly flask file. Must be '
                                                             'a file formatted with "# f_header_lines: n" as the first '
                                                             'line (where n is some number) and the data as a space '
                                                             'separated table where each line has four entries: site, '
                                                             'year, month, and CO2 concentration in ppm. If this is '
                                                             'given, it overrides --record-dir for the Mauna Loa file.')
    parser.add_argument('--smo-co2-file', default=None, help='Path to the American Samoa CO2 monthly flask file. Same '
                                                             'behavior and requirements as the Mauna Loa file.')
    parser.add_argument('--truncate-mlo-smo-by', default=0, type=int, 
                        help='Number of months to truncate the MLO and SMO data by relative to the latest month '
                             'in the met_resampled_file. Note that the MLO and SMO data must have data up to and '
                             'including the month that they will be truncated at. Default is %(default)d.')
    parser.add_argument('--no-truncate-mlo-smo', action='store_const', const=None, dest='truncate_mlo_smo_by',
                        help='Override the default to truncate MLO/SMO data.')
    parser.add_argument('--fo2-file', default=str(fo2_prep.DEFAULT_FO2_FILE),
                        help='Path to the file containing the O2 global mean DMF trend created by "update_fo2". '
                             'Default is %(default)s.')
    parser.add_argument('--auto-update-fo2-file', action='store_true',
                        help='Including this flag will activate logic to automatically check if the file pointed to by '
                             '--fo2-file needs created or updated. If it does not exist or has not been updated recently, '
                             'this will download Scripps and NOAA data to the "data" subdirectory within the ginput package '
                             'and create or update the O2 file if needed. If you need more control over where the data is '
                             'downloaded, you will need to use the "update_fo2" subcommand of run_ginput.py')
    parser.add_argument('-v', '--verbose', dest='log_level', default=0, action='count',
                        help='Increase logging verbosity')
    parser.add_argument('-q', '--quiet', dest='log_level', const=-1, action='store_const',
                        help='Silence all logging except warnings and critical messages. Note: some messages that do '
                             'not use the standard logger will also not be silenced.')
    parser.add_argument('-n', '--nprocs', default=0, type=int, help='Number of processors to use in parallelization')
    parser.add_argument('--interp-pickle-dir', default='.',
                        help='Directory in which to write temporary files containing pickles of the EqL interpolators if needed. '
                             'These files will only be written if running on Python 3.9 or less, and will be cleaned up automatically. '
                             'Default is "%(default)s"')
    parser.add_argument('--raise-errors', action='store_true', help='Raise errors normally rather than suppressing and '
                                                                    'logging them.')

    parser.epilog = 'A note on error handling: by default, most errors will be caught and, rather than halt the ' \
                    'execution of this program, will result in a non-zero flag value being stored. A short version ' \
                    'of the error message will also be printed via the logger. A full traceback can be printed to ' \
                    'the logger by increasing the verbosity to full (-vvvv). Alternately, normal error behavior can ' \
                    'be restored with the --raise-errors flag. If you see messages similar to ' \
                    '"pthread_create: Resource temporarily unavailable", this may be because numpy is trying to acquire ' \
                    'multiple threads for each process. To avoid this, set OMP_NUM_THREADS or the equivalent environmental ' \
                    'variable for your C library to limit the number of threads per process.'

    if i_am_main:
        parser.set_defaults(instrument='oco')
        return vars(parser.parse_args())
    else:
        # if not main, no need to return, modified in-place. Will be called by exterior command line interface function.
        parser.set_defaults(instrument=instrument)
        parser.set_defaults(driver_fxn=cl_driver)


def cl_driver(**args):
    log_level = args.pop('log_level')
    setup_logger(level=log_level)

    raise_errors = args.pop('raise_errors')
    handler = ErrorHandler(suppress_error=not raise_errors)

    acos_interface_main(**args, error_handler=handler)


def main():
    args = parse_args()
    cl_driver(**args)


if __name__ == '__main__':
    main()
