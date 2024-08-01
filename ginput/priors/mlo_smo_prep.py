from abc import ABC, abstractmethod, abstractclassmethod
from argparse import ArgumentParser
from enum import Enum
from logging import warning
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import pandas as pd
import re
import xarray as xr

from ..common_utils import mod_utils
from ..common_utils.ioutils import make_dependent_file_hash
from ..common_utils.ggg_logging import logger, setup_logger

from typing import Tuple, Optional

__version__ = '1.1.1'

class MloPrelimMode(Enum):
    TIME_STRICT_DIFF_EITHER = 0
    TIME_STRICT_DIFF_BOTH = 1
    TIME_RELAXED_DIFF_EITHER = 2
    TIME_RELAXED_DIFF_BOTH = 3
    
    
class MloBackgroundMode(Enum):
    TIME_AND_SIGMA = 0
    TIME_AND_PRELIM = 1


class InsituProcessingError(Exception):
    pass

    
MLO_UTC_OFFSET = pd.Timedelta(hours=-10)
SMO_LON = -170.5644
SMO_LAT = 14.2474
DEFAULT_LAST_MONTH = pd.Timestamp.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0) - relativedelta(months=1)


class RunSettings:
    def __init__(self, 
                 save_missing_geos_to=None, 
                 last_month=DEFAULT_LAST_MONTH, 
                 allow_missing_hourly_times=False, 
                 allow_missing_creation_date=False,
                 limit_by_avail_data=True,
                 **kwargs):
        self.save_missing_geos_to = save_missing_geos_to
        self.last_month = last_month
        self.allow_missing_hourly_times = allow_missing_hourly_times
        self.allow_missing_creation_date = allow_missing_creation_date
        self.limit_by_avail_data = limit_by_avail_data



def read_surface_file(surface_file: str, datetime_index: Optional[Tuple[str, str]]=('year', 'month')) -> pd.DataFrame:
    """Read a text file with NOAA surface data, either a monthly average or hourly file

    Parameters
    ----------
    surface_file
        Path to the surface file to read

    datetime_index
        If not `None`, then must be a tuple giving the names of the year and month columns
        in the file, to be converted to a monthly datetime index. (Only supported for monthly
        files.)

    Returns
    -------
    pd.DataFrame
        The data from the surface file as a dataframe. If `datetime_index` was not `None`, the
        index will be a :class:`pandas.DatetimeIndex`.
    """
    with open(surface_file) as f:
        line1 = f.readline()
        nhead = int(line1.split(':')[1])
        for _ in range(nhead-1):
            line = f.readline()
        columns = line.split(':')[1].split()
    
    df = pd.read_csv(surface_file, sep=r'\s+', skiprows=nhead, header=None)
    df.columns = columns
    if datetime_index is not None:
        yf, mf = datetime_index
        df.index = pd.DatetimeIndex([pd.Timestamp(int(r[yf]), int(r[mf]), 1) for _, r in df.iterrows()])
    return df


def read_hourly_insitu(hourly_file: str) -> pd.DataFrame:
    """Read and standardize an hourly in situ file

    Parameters
    ----------
    hourly_file
        Path to the NOAA hourly file

    Returns
    -------
    pd.DataFrame
        A dataframe with the hourly data, a :class:`pandas.DatetimeIndex`, and its columns
        standardized to "site", "year", "month", "day", "hour", "minute", "value", "uncertainty",
        and "flag".
    """
    df = read_surface_file(hourly_file, datetime_index=None)
    df = _standardize_rapid_df(df, minute_col=False, unc_col='std_dev')
    return df


def _standardize_rapid_df(df, site_col=None, year_col=None, month_col=None, day_col=None, hour_col=None, minute_col=None, 
                         value_col=None, unc_col=None, flag_col=None, time_prefix=''):
    """Convert a hourly dataframe into one with standardized column names and index
    """
    site_col = _find_column(df, 'site', site_col)
    year_col = _find_column(df, f'{time_prefix}year', year_col)
    month_col = _find_column(df, f'{time_prefix}month', month_col)
    day_col = _find_column(df, f'{time_prefix}day', day_col)
    hour_col = _find_column(df, f'{time_prefix}hour', hour_col)
    minute_col = _find_column(df, f'{time_prefix}minute', minute_col)
    value_col = _find_column(df, 'value', value_col)
    unc_col = _find_column(df, 'uncertainty', unc_col)
    flag_col = _find_column(df, 'flag', flag_col)
    
    if minute_col is False:
        df['minute'] = 0
        minute_col = 'minute'
    # xx = df[flag_col] == '...'
    # df = df.loc[xx, [year_col, month_col, day_col, hour_col, minute_col, value_col, unc_col, flag_col]].copy()
    df = df.loc[:, [site_col, year_col, month_col, day_col, hour_col, minute_col, value_col, unc_col, flag_col]].copy()
    df.rename(columns={site_col: 'site', year_col: 'year', month_col: 'month', day_col: 'day', hour_col: 'hour', minute_col: 'minute', value_col: 'value', unc_col: 'uncertainty', flag_col: 'flag'}, inplace=True)
    df.index = _dtindex_from_columns(df)
    # assume large negative values are fills
    # df.loc[df['value'] < -90, 'value'] = np.nan
    # df.loc[df['uncertainty'] < -90, 'uncertainty'] = np.nan
    return df


def _filter_rapid_df(df):
    """Filter the NOAA hourly dataframe, removing fill values and flagged data points.
    """

    # Only check the first two columns of the flag, as the third one is "extra information"
    # and will be set to "P" for preliminary data. Since we're always working with preliminary
    # data, need to ignore that. If there's other flags there, for now I'm going to ignore them
    # and assume that they do not impact the data significantly given the other selection both
    # NOAA and this code does.
    xx_flag = df['flag'].apply(lambda x: x[:2] == '..')
    xx_fills = (df['value'] < -90) | (df['uncertainty'] < -90)

    n_points = df.shape[0]
    n_flagged_out = np.sum(~xx_flag)
    n_missing_data = np.sum((df['flag'].str[0] == 'I') | xx_fills)
    percent_missing = n_missing_data / n_points * 100
    first_date = df.index.min().strftime('%Y-%m-%d %H:%M')
    last_date = df.index.max().strftime('%Y-%m-%d %H:%M')
    msg = '{flagged} of {n} data points between {start} and {end} removed by flags. {nmiss} ({pmiss:.2f}%) data points are missing data.'.format(
        flagged=n_flagged_out, n=n_points, start=first_date, end=last_date, nmiss=n_missing_data, pmiss=percent_missing
    )
    if percent_missing > 50:
        logger.warn(msg)
    else:
        logger.info(msg)

    df = df.loc[xx_flag, :].copy()
    
    df.loc[xx_fills, 'value'] = np.nan
    df.loc[xx_fills, 'uncertainty'] = np.nan
    return df
    
    
def _find_column(df, column_name, given_column=None):
    """Find a column in a NOAA dataframe containing the substring `column_name`
    """
    if given_column is not None:
        return given_column
    elif given_column is False:
        return False
    
    matching_columns = [c for c in df.columns if column_name in c]
    if len(matching_columns) != 1:
        matches = ', '.join(matching_columns)
        raise TypeError(f'Cannot identify unique {column_name} field, found {matches}')
        
    return matching_columns[0]


def _dtindex_from_columns(df, year_col=None, month_col=None, day_col=None, hour_col=None, minute_col=None):
    """Create a datetime index from component time columns
    """
    year_col = _find_column(df, 'year', year_col)
    month_col = _find_column(df, 'month', month_col)
    day_col = _find_column(df, 'day', day_col)
    hour_col = _find_column(df, 'hour', hour_col)
    minute_col = _find_column(df, 'minute', minute_col)
    
    return pd.to_datetime(df[[year_col, month_col, day_col, hour_col, minute_col]])
    

def noaa_prelim_flagging(noaa_df: pd.DataFrame, hr_std_dev_max: float = 0.2, hr2hr_diff_max: float = 0.25, 
                        mode: MloPrelimMode = MloPrelimMode.TIME_RELAXED_DIFF_EITHER, full_output: bool = False) -> pd.DataFrame:
    """Do preliminary selection of background data for NOAA houly in situ data

    Parameters
    ----------
    noaa_df
        A NOAA hourly dataframe read by :func:`read_hourly_insitu`.

    hr_std_dev_max
        The maximum standard deviation allowed within one hourly data point for it to be kept.

    hr2hr_diff_max
        Maximum allowed difference between adjacent hourly points for them to be retained. How this
        is interpreted depends on `mode`.

    mode:
        Controls how the hour-to-hour differences are used to reject data points. The different
        :class:`MloPrelimMode` variants mean:

        * `TIME_RELAXED_DIFF_EITHER`: Keep points when the difference with either adjacent point
          is less than `hr2hr_diff_max`. If the time difference is >1 hour (due to the standard
          deviation or flagging), do not count that VMR difference.
        * `TIME_RELAXED_DIFF_BOTH`: Only keep points where the VMR differences are smaller than
          `hr2hr_diff_max` *or* the time difference is >1 on both sides.
        * `TIME_STRICT_DIFF_EITHER`: Keep a point only if it has at least one neighbor with a
          VMR difference smaller than `hr2hr_diff_max` and a time difference < 1 hr.
        * `TIME_STRICT_DIFF_BOTH`: Keep a point only if the VMR difference with both neighbors
          is smaller than `hr2hr_diff_max` and both time differences are < 1 hr.

    full_output
        Set to `True` to output two additional logical vectors that indicate which points pass the 
        hourly standard deviation and hour-to-hour difference criteria. If `False`, only the dataframe
        limited to good points is returned.
    """
    # The first condition - remove points with standard deviation above some
    # value - is easy. I will also omit times with NaNs
    noaa_df = noaa_df.dropna()
    
    xx_sd = noaa_df['uncertainty'] <= hr_std_dev_max
    noaa_df = noaa_df.loc[xx_sd, :]
    
    # The next one is more complicated, as we want to retain times when the 
    # hour to hour difference is within X ppm. We need to check that both
    # (a) the time difference is 1 hour and (b) the value difference is
    # less than X
    td_diff = (noaa_df.index[1:] - noaa_df.index[:-1]) - pd.Timedelta(hours=1)
    xx_tdiff = (td_diff >= pd.Timedelta(minutes=-5)) & (td_diff <= pd.Timedelta(minutes=5))
    values = noaa_df['value'].to_numpy()
    xx_vdiff = np.abs(values[1:] - values[:-1]) <= hr2hr_diff_max
    
    # Want xx_diff to be true for differences that DO NOT exclude
    if mode in {MloPrelimMode.TIME_RELAXED_DIFF_BOTH, MloPrelimMode.TIME_RELAXED_DIFF_EITHER}:
        # These modes mean that when the time difference is greater than an hour we ignore
        # that difference in DMF values, as a time difference > 1 means we don't know what
        # the typical DMF change should be. 
        #
        # That is, keep if the DMF differences is small enough OR the time difference is too large
        xx_diff = xx_vdiff | ~xx_tdiff
    elif mode in {MloPrelimMode.TIME_STRICT_DIFF_BOTH, MloPrelimMode.TIME_STRICT_DIFF_EITHER}:
        # These modes mean that a time difference of >1 hour is the same as a DMF difference
        # above the threshold. 
        #
        # That is, keep if the DMF difference is small enough AND the time difference is small enough
        xx_diff = xx_vdiff & xx_tdiff
    else:
        raise TypeError('Unknown mode')
    
    xx_hr2hr = np.zeros(noaa_df.shape[0], dtype=np.bool_)
    if mode in {MloPrelimMode.TIME_RELAXED_DIFF_EITHER, MloPrelimMode.TIME_STRICT_DIFF_EITHER}:
        # In these modes, a point is kept as long as the difference on at least one side is
        # small enough
        xx_hr2hr[1:-1] = xx_diff[:-1] | xx_diff[1:]
    elif mode in {MloPrelimMode.TIME_RELAXED_DIFF_BOTH, MloPrelimMode.TIME_STRICT_DIFF_BOTH}:
        # In these modes, a point is kept only if the differences on BOTH sides are small enough.
        xx_hr2hr[1:-1] = xx_diff[:-1] & xx_diff[1:]
    else:
        raise TypeError(f'Unknown `mode` "{mode}"')
        
    # For the first and last points, there's only one difference to consider
    if np.size(xx_hr2hr) > 0 and np.size(xx_diff) > 0:
        xx_hr2hr[0] = xx_diff[0]
        xx_hr2hr[-1] = xx_diff[-1]
    elif np.size(xx_hr2hr) == 0 and np.size(xx_diff) == 0:
        logger.info('There are no data points to flag by hour to hour differences')
    else:
        raise NotImplementedError('A case occurred where of two arrays that should both be size 0 or not, one had size 0 and the other did not. This case is not handled.')
    
    if full_output:
        xx_hr2hr_full = np.zeros_like(xx_sd)
        xx_hr2hr_full[xx_sd] = xx_hr2hr
        return noaa_df.loc[xx_hr2hr, :], xx_sd, xx_hr2hr_full
    else:
        return noaa_df.loc[xx_hr2hr, :]
    
    
def mlo_background_selection(mlo_df: pd.DataFrame, method: MloBackgroundMode) -> pd.DataFrame:
    """Limit a Mauna Loa hourly dataframe to background data.

    Parameters
    ----------
    mlo_df
        The MLO hourly dataframe.

    method:
        How to do the background selection. The two enum variants are:

        * `TIME_AND_SIGMA`: limit to midnight to 7a local time and where the standard
          deviation is less than 0.3 ppm.
        * `TIME_AND_PRELIM`: limit to midnight to 7a local time and where 
          :func:`noaa_prelim_flagging` would keep the data.

    Returns
    -------
    pd.DataFrame
        `mlo_df` with non-background rows removed.
    """
    if method == MloBackgroundMode.TIME_AND_SIGMA:
        return _mlo_background_time_sigma(mlo_df)
    elif method == MloBackgroundMode.TIME_AND_PRELIM:
        return _mlo_background_time_prelim(mlo_df)
    else:
        raise NotImplementedError(f'Unimplemented method "{method.name}"')
        
        
def _mlo_background_time_sigma(mlo_df: pd.DataFrame) -> pd.DataFrame:
    """Limit MLO data to background by time and hourly std. dev. only
    """
    local_times = mlo_df.index + MLO_UTC_OFFSET
    xx = (local_times.hour >= 0) & (local_times.hour < 7) & (mlo_df['uncertainty'] < 0.3)
    return mlo_df.loc[xx,:]


def _mlo_background_time_prelim(mlo_df: pd.DataFrame) -> pd.DataFrame:
    """Limit MLO data to background by time and :func:`noaa_prelim_flagging`.
    """
    mlo_df = noaa_prelim_flagging(mlo_df)
    local_times = mlo_df.index + MLO_UTC_OFFSET
    xx = (local_times.hour >= 0) & (local_times.hour < 7)
    return mlo_df.loc[xx,:]


def compute_wind_for_times(wind_file: str, times: pd.DatetimeIndex, wind_alt: int = 10, run_settings: RunSettings = RunSettings(), allow_missing_geos_files: bool = False) -> pd.DataFrame:
    """Compute winds for specific times from a file already interpolated to a specific lat/lon
    
    Parameters
    ----------
    wind_file
        Either:

        1. A file containing a list of GEOS FP-IT surface files that span the times
           in the `times` input, or
        2. A file summarizing the GEOS FP-IT surface variables at the SMO lat/lon.
           It must have `UxM` and `VxM` variables, where "x" is the wind altitude 
           (see `wind_alt`)
        
    times
        Times to interpolate to.
        
    wind_alt
        Which surface wind altitude (2, 10, or 50 meters usually) to use. This will 
        look for variables named e.g. U10M and V10M in the GEOS file(s), with the 
        number changing based on the altitude.

    run_settings
        A :class:`RunSettings` instance that carries configuration.

    allow_missing_geos_files
        Set to `True` to allow this function to complete if any of the expected 
        GEOS files were missing. By default an error is raised.
    
        
    Returns
    -------
    pd.DataFrame
        Data frame with the U and V wind vectors, wind velocity, and wind direction
        indexed by time. The vectors and velocity will have the same units as in the
        `winds_file` (usually meters/second) and the wind direction uses the convention
        of what direction the wind is coming FROM in degrees clockwise from north.
        
    Notes
    -----
    10 is the default `wind_alt` because Waterman et al. 1989 (JGR, vol. 94, pp. 14817--14829)
    indicates in the "Air intake and topography" section that sampling heights between 6 and
    18 meters were suitable.
    """
    wind_dataset = get_smo_winds_from_file(wind_file, wind_alt=wind_alt)
    _check_geos_times(times, wind_dataset, run_settings=run_settings, allow_missing_geos_files=allow_missing_geos_files)
    
    u = wind_dataset['u'][:]
    v = wind_dataset['v'][:]
        
    u = u.interp(time=times)
    v = v.interp(time=times)
    
    # Velocity is easy - just the magnitude of the combined vector
    velocity = np.sqrt(u**2 + v**2)
    
    # Direction, in the convention of giving the direction the wind
    # goes towards in deg. CCW from east, is just the inverse tangent
    # (accounting for quadrant)
    #
    # However, this needs converted to the convention of giving the
    # direction the wind is coming from in deg. CW from north. That means:
    #   0 -> 270  or (1,0) -> (0,-1)
    #   45 -> 225 or (1,1) -> (-1,-1)
    #   90 -> 180 or (0,1) -> (-1, 0)
    #   135 -> 135 or (-1,1) -> (-1,1)
    #   180 -> 90 or (-1,0) -> (0,1)
    #   225 -> 45 or (-1,-1) -> (1,1)
    #   270 -> 0 or (0,-1) -> (1,0)
    #   280 -> 350 
    #   305 -> 325
    #   315 -> 315 or (1,-1) -> (1,-1)
    #   325 -> 305 
    #
    # where the pairs after the "or" are the x,y vectors that, if put
    # through arctan2, give their respective directions. Therefore, 
    # we just need a u = -v transformation
    direction = np.rad2deg(np.arctan2(-u, -v))
    direction[direction < 0] += 360
    
    return pd.DataFrame({'velocity': velocity, 'direction': direction, 'u': u, 'v': v}, index=times)


def _check_geos_times(data_times, geos_dataset, run_settings: RunSettings = RunSettings(), allow_missing_geos_files: bool = False):
    """Check that all 3-hourly GEOS times needed for `data_times` are included in `geos_dataset`.
    """
    data_times = pd.DatetimeIndex(data_times)
    data_start = data_times.min().floor('D')
    data_end = data_times.max().ceil('D')
    geos_times = set(pd.DatetimeIndex(geos_dataset.time.data))
    expected_times = set(pd.date_range(data_start, data_end, freq='3H'))
    missing_times = expected_times.difference(geos_times)
    n_missing = len(missing_times)
    if run_settings.save_missing_geos_to:
        with open(run_settings.save_missing_geos_to, 'w') as f:
            for mtime in sorted(missing_times):
                f.write('{}\n'.format(mtime))
    if len(missing_times) > 0:
        msg = 'Missing data from {n} GEOS times between {start} and {end} (inclusive).'.format(n=n_missing, start=data_start, end=data_end)
        if allow_missing_geos_files:
            logger.warning(msg)
        else:
            raise InsituProcessingError(msg)



def merge_insitu_with_wind(insitu_df: pd.DataFrame, wind_file: str, wind_alt: float = 10, run_settings: RunSettings = RunSettings(), allow_missing_geos_files: bool = False) -> pd.DataFrame:
    """Merge an in situ hourly dataframe with SMO data with GEOS surface winds

    Parameters
    ----------
    insitu_df
        The SMO hourly dataframe.

    wind_file
        Path to a file that lists GEOS surface files, one per line. All GEOS files between the 
        day floor and day ceiling of the times in `insitu_df` must be included. That is, if
        `insitu_df` has data from 2021-08-02 16:00 to 2021-08-28 19:00 UTC, then GEOS files
        from 2021-08-02 00:00 to 2021-08-29 00:00 UTC are required.

    wind_alt
        Which altitude above the surface to draw winds from. 10 meters is the default as that is
        close to tha altitude of the NOAA sampling intake (see :func:`compute_wind_for_times`)

    run_settings
        A :class:`RunSettings` instance that carries configuration for extra outputs.

    Returns
    -------
    pd.DataFrame
        The `insitu_df` with columns for wind speed, velocity, u-component, and v-component
        added, each interpolated to the times in the `insitu_df`.
    """
    wind_df = compute_wind_for_times(wind_file, times=insitu_df.index, wind_alt=wind_alt, run_settings=run_settings, allow_missing_geos_files=allow_missing_geos_files)
    return insitu_df.merge(wind_df, left_index=True, right_index=True)


def smo_wind_filter(smo_df: pd.DataFrame, first_wind_dir: float = 330.0, last_wind_dir: float = 160.0, min_wind_speed: float = 2.0) -> pd.DataFrame:
    """Subset an SMO CO2 dataframe to just rows with specific wind conditions
    
    Parameters
    ----------
    smo_df
        The dataframe of SMO CO2 DMFs with wind data included (see :func:`merge_insitu_with_wind`)
        
    first_wind_dir
    last_wind_dir
        These set the range of wind directions permitted; only data with a wind direction in the
        clockwise slice between `first_wind_dir` and `last_wind_dir` are retained.
        
    min_wind_speed
        The slowest wind speed allowed; only rows with a wind speed greater than or equal to this
        are retained.
        
    Returns
    -------
    pd.DataFrame
        A data frame that has a subset of the rows in `smo_df`.
        
    Notes
    -----
    The default wind limits come from  Waterman et al. 1989 (JGR, vol. 94, pp. 14817--14829).
    In the section "Data Processing," they give two different criteria for wind direction. Although
    they found that the looser constrains kept much more data and did not introduce significant numbers
    of non-background measurements, I am using the stricter criteria, since I am filtering on
    GEOS FP-IT winds, which likely have some error compared to the surface winds measured at SMO.
    """
    if first_wind_dir < last_wind_dir:
        xx = (smo_df['direction'] >= first_wind_dir) & (smo_df['direction'] <= last_wind_dir)
    else:
        xx = (smo_df['direction'] >= first_wind_dir) | (smo_df['direction'] <= last_wind_dir)
        
    xx &= smo_df['velocity'] >= min_wind_speed
    return smo_df.loc[xx,:]


def monthly_avg_rapid_data(df: pd.DataFrame, year_field: Optional[str] = None, month_field: Optional[str] = None) -> pd.DataFrame:
    """Compute monthly averages from an hourly dataframe

    Parameters
    ----------

    df
        The hourly dataframe to compute from

    year_field
        Which column in the dataframe gives the year. If this is `None`, will try
        to find a column containing "year".

    month_field
        Which column in the dataframe gives the month. If this is `None`, will try
        to find a column containing "month".

    Returns
    -------
    pd.DataFrame
        The input dataframe averaged to months, with the index set to datetimes at the start of each month.
    """
    year_field = _find_column(df, 'year', year_field)
    month_field = _find_column(df, 'month', month_field)

    
    monthly_df = df.groupby([year_field, month_field]).mean().reset_index()
    monthly_df.index = pd.DatetimeIndex(pd.Timestamp(int(r[year_field]),int(r[month_field]),1) for _,r in monthly_df.iterrows())
    return monthly_df


# --------------- #
# WIND RESAMPLING #
# --------------- #

def get_smo_winds_from_file(winds_file: str, wind_alt: int = 10) -> xr.Dataset:
    """Get surface winds interpolated to SMO lat/lon from GEOS surface files.

    Parameters
    ----------
    wind_file
        Path to a file that lists GEOS surface files, one per line. All GEOS files between the 
        day floor and day ceiling of the times in `insitu_df` must be included. That is, if
        `insitu_df` has data from 2021-08-02 16:00 to 2021-08-28 19:00 UTC, then GEOS files
        from 2021-08-02 00:00 to 2021-08-29 00:00 UTC are required.

    wind_alt
        Which altitude above the surface to draw winds from. 10 meters is the default as that is
        close to tha altitude of the NOAA sampling intake (see :func:`compute_wind_for_times`)

    Returns
    -------
    xr.Dataset
        An xarray dataset containing 'u' and 'v' variables index by time.
    """
    if winds_file.endswith('.nc'):
        raise NotImplementedError('Reading from a pre-interpolated GEOS summary is not supported')
    else:
        return _resample_winds_from_geos_file_list(winds_file, wind_alt=wind_alt)


def _resample_winds_from_geos_file_list(winds_file, wind_alt=10, lon=SMO_LON, lat=SMO_LAT):
    """Interpolate surface winds from a list of GEOS surface files.
    """
    with open(winds_file) as f:
        geos_files = np.array(f.read().splitlines())

    geos_data = None
    u_var = 'U{}M'.format(wind_alt)
    v_var = 'V{}M'.format(wind_alt)

    nfiles = len(geos_files)
    logger.info('Interpolating data from {} surface geos files to lon={}, lat={}'.format(nfiles, lon, lat))
    for ifile, gf in enumerate(geos_files):
        if ifile % 100 == 0 and ifile > 0:
            logger.info('  * Done with {i} of {n} files'.format(i=ifile, n=nfiles))

        with xr.open_dataset(gf) as ds:
            if geos_data is None:
                # Ensure we have the right datatype or the datetime gets messed up
                geos_data = {
                    'times': np.full(geos_files.shape, 0, dtype=ds.time.dtype),
                    'u': np.full(geos_files.shape, np.nan),
                    'v': np.full(geos_files.shape, np.nan)
                }
            
            geos_data['times'][ifile] = ds.time.item()
            geos_data['u'][ifile] = ds[u_var].interp(lon=lon, lat=lat).item()
            geos_data['v'][ifile] = ds[v_var].interp(lon=lon, lat=lat).item()

    logger.info('Done with {n} of {n} GEOS files'.format(n=nfiles))

    geos_times = geos_data.pop('times')
    geos_data = {k: xr.DataArray(v, coords=[geos_times], dims=['time']) for k, v in geos_data.items()}
    return xr.Dataset(geos_data, coords={'lon': lon, 'lat': lat})


def get_hourly_file_creation_date(hourly_file: str) -> pd.Timestamp:
    creation_date = None
    with open(hourly_file) as f:
        for line in f:
            m = re.match(r'#\s+description_creation-time:\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if m is not None:
                creation_date = pd.to_datetime(m.group(1))

    if creation_date is None:
        msg = 'Could not find "description_creation-time" in file "{}", cannot enforce that the file contains data through the last expected month'.format(hourly_file)
        raise InsituProcessingError(msg)
    else:
        return creation_date


# -------------- #
# DRIVER CLASSES #
# -------------- #


class InsituMonthlyAverager(ABC):
    """Abstract class that handles most of the logic for updating MLO or SMO monthly average files.

    To implement a site-specific concrete averager, only two methods need implemented: :meth:`class_site`
    (a class method that returns the site code, e.g. "MLO" or "SMO") and :meth:`select_background`, which
    takes in an hourly dataframe and returns one with only background data left as rows.

    To update a monthly file, use the :meth:`convert` method.
    """
    @abstractclassmethod
    def class_site(cls):
        """Returns the site code (e.g. "MLO") for the class.
        """
        pass

    @abstractmethod
    def select_background(self, hourly_df: pd.DataFrame) -> pd.DataFrame:
        """Select only background data from an hourly dataframe.

        Parameters
        ----------
        hourly_df
            The dataframe with all good-quality hourly data

        Returns
        -------
        pd.DataFrame
            A dataframe with only background data kept.
        """
        pass

    def __init__(self, clobber: bool = False, run_settings: RunSettings = RunSettings()) -> None:
        self._clobber = clobber
        self._run_settings = run_settings

    @staticmethod
    def get_new_hourly_data(monthly_df: pd.DataFrame, hourly_df: pd.DataFrame, last_expected_month: pd.Timestamp,
                            allow_missing_times: bool = False, creation_month: Optional[pd.Timestamp] = None,
                            limit_to_avail_data: bool = True) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Get the subset of `hourly_df` that has new data to append to the end of `monthly_df`

        Parameters
        ----------
        monthly_df
            A dataframe of monthly-averaged data from the previous monthly-average file that is being updated.

        hourly_df
            A dataframe of new hourly data, not yet filtered for good-quality data. It must contain all hours
            for the months it is to add.

        last_expected_month
            The last month required to have data in the hourly file. Note that this data may be all fill values,
            it only requires that this month and all months after the end of the previous monthly data be present.

        allow_missing_times
            By default, an error is raised if any of the expected data is not present in the hourly file. Setting
            this to `True` reduces that error to a warning.

        Returns
        -------
        pd.DataFrame
            The subset of `hourly_df` that is new and good quality.

        pd.DatetimeIndex
            A sequence of dates that are the first of the month for every month to be added by the first return
            value.

        Raises
        ------
        :class:`InsituProcessingError` if `hourly_df` does not contain the required data between the end of the
          previous monthly data and the last expected month. If `allow_missing_times` is `True`, then this is not
          raised and a warning is logged instead.
        """
        if not last_expected_month.is_month_start:
            raise ValueError('last_expected_month must be a timestamp at the start of a month')
        if creation_month is not None and not creation_month.is_month_start:
            raise ValueError('creation_month must be a timestamp at the start of a month, if given')
        if creation_month is not None and creation_month <= last_expected_month and limit_to_avail_data:
            cs = creation_month.strftime('%b %Y')
            ls = last_expected_month.strftime('%b %Y')
            raise InsituProcessingError('Extending the monthly average file to {} is not allowed by default because the hourly file was created in {} (hourly file creation must be at least one month later than last month requested). Use --no-limit-by-avail-data to bypass this check.'.format(ls, cs))

        # First verify that all expected months are complete. This assumes that the input dataframe has not
        # been filtered and has a value for every hour. If that's not the case, `allow_missing_times` can be
        # set to permit missing times.
        first_month = monthly_df.index.max() + relativedelta(months=1)
        curr_month = first_month
        hourly_df_timestamps = set(hourly_df.index)

        while curr_month <= last_expected_month:
            next_month = curr_month + relativedelta(months=1)
            expected_timestamps = set(pd.date_range(curr_month, next_month, freq='H', closed='left'))
            
            # Do we have all the time stamps for the current month?
            intersection = expected_timestamps.intersection(hourly_df_timestamps)
            if intersection != expected_timestamps:
                n_found = len(intersection)
                n_expected = len(expected_timestamps)
                msg = 'Found {found} of {expected} expected hourly timestamps for {month}, month is incomplete. (Note that the hourly file truncated to the creation time, unless that attribute cannot be found.)'.format(
                    found=n_found, expected=n_expected, month=curr_month.strftime('%B %Y')
                )

                if allow_missing_times or (creation_month is not None and curr_month >= creation_month and not limit_to_avail_data):
                    logger.warning(msg)
                else:
                    raise InsituProcessingError(msg)
                
            curr_month = next_month

        cutoff_date = last_expected_month + relativedelta(months=1)
        if limit_to_avail_data and creation_month < cutoff_date:
            if creation_month is None:
                raise TypeError('If `limit_to_avail_data` is `True`, `creation_month` must be provided')

            last_month_str = (creation_month - relativedelta(months=1)).strftime('%b %Y')
            logger.warning('The requested final month is later than can be accomodated by the hourly file due to its creation date, the last month that will be added is {}'.format(last_month_str))
            cutoff_date = creation_month
            
        tt = (hourly_df.index >= first_month) & (hourly_df.index < cutoff_date)
        hourly_df = hourly_df.loc[tt, :].copy()
        if hourly_df.shape[0] == 0:
            raise InsituProcessingError('No new hourly data to add given specified last month and/or creation date of the hourly file')
        hourly_df = _filter_rapid_df(hourly_df)
        return hourly_df, pd.date_range(first_month, cutoff_date, freq='MS', closed='left')

    @staticmethod
    def check_hourly_file_creation_date(hourly_file: str, last_expected_month: pd.Timestamp = DEFAULT_LAST_MONTH) -> None:
        cutoff_date = last_expected_month + relativedelta(months=1)
        creation_date = get_hourly_file_creation_date(hourly_file)
        if creation_date < cutoff_date:
            raise InsituProcessingError('Hourly file creation time ({}) is before the last cutoff time ({} = last expected date + 1 month).'.format(creation_date, cutoff_date))

    @classmethod
    def write_monthly_insitu(cls, output_file: str, monthly_df: pd.DataFrame, previous_monthly_file: str, new_hourly_file: str, 
                             new_months: pd.DatetimeIndex, clobber: bool = False) -> None:
        """Write a new monthly average file
        
        Parameters
        ----------
        output_file
            Path to write to

        monthly_df
            Dataframe with the monthly data to write; must have four columns: site, year, month, value.

        previous_monthly_file
            Path to the previous monthly file

        new_hourly_file
            Path to the hourly file used for this update

        new_months
            A sequence of datetimes giving the first of each month added

        clobber
            Whether to allow overwriting the output file if it already exists.
        """
        if monthly_df.shape[1] != 4:
            raise InsituProcessingError('The monthly dataframe must have 4 columns (site, year, month, value)')
        if not clobber and os.path.exists(output_file):
            raise IOError('Output file already exists')

        new_header = cls._make_monthly_header(previous_monthly_file=previous_monthly_file, new_hourly_file=new_hourly_file, new_months=new_months)
        with open(output_file, 'w') as outf:
            outf.write('\n'.join(new_header) + '\n')
            monthly_df.to_string(outf, index=False, header=False)


    @staticmethod
    def _make_monthly_header(previous_monthly_file, new_hourly_file, new_months):
        """Make the header for the new monthly file; copies the old header and adds the source for the new months added.
        """
        if len(new_months) != 2:
            raise TypeError('Expected `new_months` to be a length-2 sequence')

        header = []
        added_history = False
        with open(previous_monthly_file, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break

                if line.startswith('# END HISTORY'):
                    if added_history:
                        raise InsituProcessingError('Error copying header from {}: "# END HISTORY" found multiple times'.format(previous_monthly_file))
                    first_new_month = new_months[0].strftime('%Y-%m')
                    last_new_month = new_months[1].strftime('%Y-%m')
                    history_lines = [
                        '#    Added {} to {} using mlo_smo_prep version {}:'.format(first_new_month, last_new_month, __version__),
                        '#        - Previous monthly file: {} (SHA1 = {})'.format(previous_monthly_file, make_dependent_file_hash(previous_monthly_file)),
                        '#        - New hourly file: {} (SHA1 = {})'.format(new_hourly_file, make_dependent_file_hash(new_hourly_file))
                    ]

                    header.extend(history_lines)
                    added_history = True

                # Remove whitespace (including newlines) from the end of header lines - easier to 
                # add it back in consistently when writing
                header.append(line.rstrip()) 
            
        # Check that the first and last lines of the header are what we expect and that we added the history
        if not header[0].startswith('# header_lines'):
            raise InsituProcessingError('Error copying header from {}: first header line does not contain the number of header lines'.format(previous_monthly_file))
        if not header[-1].startswith('# data_fields'):
            raise InsituProcessingError('Error copying header from {}: last header line does not contain the data fields'.format(previous_monthly_file))
        if not added_history:
            raise InsituProcessingError('Error copying header from {}: did not find where to insert the history'.format(previous_monthly_file))

        # Update the number of header lines
        header[0] = '# header_lines: {}'.format(len(header))
        return header

    @staticmethod
    def _check_output_clobber(output_file, previous_file, clobber):
        """Check whether it is okay to write the output file
        """
        if os.path.exists(output_file) and not clobber:
            raise InsituProcessingError('Cannot overwrite existing output file without `clobber=True` (the --clobber flag on the command line)')
        elif os.path.exists(output_file) and os.path.samefile(output_file, previous_file):
            raise InsituProcessingError('Cannot directly overwrite the previous monthly output file, even with clobber set')

    @classmethod
    def _check_site(cls, monthly_df, hourly_df, site, allow_alt_sites=False):
        """Check that the monthly and hourly dataframes are for the same NOAA site
        """
        def inner_check(df):
            df_sites = df['site'].unique()
            ok = df_sites.size == 1
            if not allow_alt_sites:
                ok = ok and df_sites[0] == site
            return ok, df_sites

        if allow_alt_sites:
            hourly_ok, hourly_sites = inner_check(hourly_df)
            if not hourly_ok:
                raise InsituProcessingError('Given hourly file contains multiple NOAA site ID: {}'.format(', '.join(hourly_sites)))
            else:
                logger.info('Allowing alternate NOAA site ID %s, as requested', hourly_sites.item())
        else:
            hourly_ok, hourly_sites = inner_check(hourly_df)
            if not hourly_ok:
                raise InsituProcessingError('Given hourly file does not contain only {site} data. Sites present: {all_sites}'.format(site=site, all_sites=', '.join(hourly_sites)))

        # Once multiple sites have been ingested into a monthly file, the check for one site in the monthly file will always fail, so there's not much point in
        # enforcing that requirement. But change the severity of the message so that if the --allow-alt-sites flag is passed, it's just informational, but if
        # that flag was not present, we get a more noticeable warning (in case the mixed sites were accidental).
        monthly_ok, monthly_sites = inner_check(monthly_df)
        if not monthly_ok:
            if allow_alt_sites:
                logger.info('Note: given input monthly file contains multiple sites: %s', ', '.join(monthly_sites))
            else:
                logger.warning('Caution: given input monthly file contains sites other than %s or multiple sites: %s. This is okay if a previous version of the monthly file was generated with alternate site data.',
                               cls.class_site(),  ', '.join(monthly_sites))

        return hourly_sites.item()


    def convert(self, noaa_hourly_file: str, previous_monthly_file: str, output_monthly_file: str, allow_alt_sites: bool = False, site_id_override: Optional[str] = None) -> None:
        """Convert a NOAA hourly file to monthly averages and append to the end of an existing file.

        Parameters
        ----------
        noaa_hourly_file
            Path to the NOAA hourly file to use to update `previous_monthly_file`.

        previous_monthly_file
            Path to the previous monthly averages file, the output file will copy its existing data
            and append new monthly average(s) to the end.

        output_monthly_file
            Path to write the output file

        allow_alt_sites
            Set to ``True`` to allow the input hourly file to contain data with a site ID different
            from the one defined by ``self.class_id()``. Otherwise, that would raise an :class:`InsituProcessingError`.
            Changing to ``True`` also changes how mismatches between the input monthly file an the
            ``self.class_id()`` value are reported, but such mismatches do not raise an error in
            either case.

        site_id_override
            If given, then this value will be used as the site ID for the new row(s) added to the output
            monthly file. When this is given, ``allow_alt_sites`` has no effect. Mismatches between the
            ID in the input hourly file and the ID given to this argument are reported, but do not
            raise an exception.

            .. note::
               Passing a string with more than 3 characters for ``site_id_override`` *should* work, but 
               is not officially tested/supported.

        Returns
        -------
        None
            Writes to `output_monthly_file`
        """
        self._check_output_clobber(output_monthly_file, previous_monthly_file, clobber=self._clobber)

        logger.info('Reading previous {} monthly file ({})'.format(self.class_site(), previous_monthly_file))
        monthly_df = read_surface_file(previous_monthly_file)

        logger.info('Reading hourly {} in situ file ({})'.format(self.class_site(), noaa_hourly_file))
        hourly_df = read_hourly_insitu(noaa_hourly_file)
        try:
            hourly_creation_date = get_hourly_file_creation_date(noaa_hourly_file)
        except InsituProcessingError as err:
            if self._run_settings.allow_missing_creation_date:
                logger.warning(str(err))
                # pretend that the creation date is late enough that all rows in the dataframe
                # are potentially valid, since we don't know any better
                hourly_creation_date = hourly_df.index.max() + relativedelta(years=10)
            else:
                raise

        # Always restrict the monthly averages to full months - this avoids using non QA/QC'ed data
        # from NOAA, since a file created in month N will only have QC'ed data through month N-1; 
        # data in month N is probably not QC'ed.
        tt = hourly_df.index < hourly_creation_date
        hourly_df = hourly_df.loc[tt, :].copy()
        hourly_creation_month = pd.Timestamp(hourly_creation_date.year, hourly_creation_date.month, 1)

        # Handle checks for the site IDs in the input hourly file
        if site_id_override is None:
            # The default behavior is to require that *all* site IDs in the input file be "MLO" or "SMO" for each 
            # site's class, respectively. If not, then this raises an exception. However, setting allow_alt_sites to
            # ``True`` will allow different site IDs *as long as* all rows of the input file have the *same* ID.
            # Note that starting from ginput v1.1.5d, mismatches in the monthly file only ever log a warning. 
            noaa_site_id = self._check_site(monthly_df, hourly_df, self.class_site(), allow_alt_sites=allow_alt_sites)
        else:
            # We can also provide our own site ID. This is mainly useful if the input hourly file has a mix of site IDs,
            # since that always raises an exception no matter what `allow_alt_sites` is. But we could also use this if
            # we need to specify our own site ID for a different reason.
            try:
                hourly_site_id = self._check_site(monthly_df, hourly_df, site_id_override, allow_alt_sites=True)
            except InsituProcessingError:
                # With allow_alt_sites = True, this error indicates that the input file had multiple site IDs
                logger.info('Multiple site IDs were detected in the input hourly file, using specified override "%s" instead', site_id_override)
            else:
                # Otherwise tell the user what site ID they're replacing
                logger.info('Using site ID override "%s" in place of site ID "%s" detected in the input hourly file', site_id_override, hourly_site_id)

            noaa_site_id = site_id_override

        hourly_df, new_month_index = self.get_new_hourly_data(
            monthly_df=monthly_df, 
            hourly_df=hourly_df, 
            last_expected_month=self._run_settings.last_month,
            allow_missing_times=self._run_settings.allow_missing_hourly_times,
            creation_month=hourly_creation_month,
            limit_to_avail_data=self._run_settings.limit_by_avail_data
        )
        
        logger.info('Doing background selection')
        hourly_df = self.select_background(hourly_df)
        
        logger.info('Doing monthly averaging')
        new_monthly_df = monthly_avg_rapid_data(hourly_df)
        new_monthly_df = new_monthly_df.reindex(new_month_index)  # ensure that all months are represented, even if some have no data
        new_monthly_df['year'] = new_month_index.year
        new_monthly_df['month'] = new_month_index.month
        
        logger.info('Writing to {}'.format(output_monthly_file))
        new_monthly_df['site'] = noaa_site_id
        new_monthly_df = new_monthly_df[['site', 'year', 'month', 'value']]

        first_new_date = new_monthly_df.index.min()
        last_new_date = new_monthly_df.index.max()
        new_monthly_df = pd.concat([monthly_df, new_monthly_df], axis=0)
        
        if output_monthly_file is None:
            return new_monthly_df
        else:
            self.write_monthly_insitu(
                output_file=output_monthly_file, 
                monthly_df=new_monthly_df, 
                previous_monthly_file=previous_monthly_file,
                new_hourly_file=noaa_hourly_file, 
                new_months=(first_new_date, last_new_date),
                clobber=self._clobber
            )
            logger.info('New monthly averages written to {}'.format(output_monthly_file))


class MloMonthlyAverager(InsituMonthlyAverager):
    def __init__(self, background_method=MloBackgroundMode.TIME_AND_PRELIM, clobber=False, run_settings: RunSettings = RunSettings()):
        self._background_method = background_method
        # self._clobber = clobber
        super().__init__(clobber=clobber, run_settings=run_settings)

    @classmethod
    def class_site(cls):
        return 'MLO'

    def select_background(self, hourly_df):
        return mlo_background_selection(hourly_df, method=self._background_method)
        
        
class SmoMonthlyAverager(InsituMonthlyAverager):
    def __init__(self, smo_wind_file: str, clobber: bool = False, run_settings: RunSettings = RunSettings(), allow_missing_geos_files: bool = False):
        self._smo_wind_file = smo_wind_file
        self._allow_missing_geos_files = allow_missing_geos_files
        # self._clobber = clobber
        super().__init__(clobber=clobber, run_settings=run_settings)

    @classmethod
    def class_site(cls):
        return 'SMO'

    def select_background(self, hourly_df):
        hourly_df = noaa_prelim_flagging(hourly_df, hr_std_dev_max=0.3)
        if hourly_df.shape[0] == 0:
            logger.warn('No hourly data passed initial flagging, skipping filter on wind direction')
        else:
            hourly_df = merge_insitu_with_wind(hourly_df, self._smo_wind_file, run_settings=self._run_settings, allow_missing_geos_files=self._allow_missing_geos_files)
            hourly_df = smo_wind_filter(hourly_df)
        return hourly_df


def driver(site, previous_monthly_file, hourly_insitu_file, output_monthly_file, last_month=DEFAULT_LAST_MONTH, geos_2d_file_list=None, 
           allow_missing_geos_files=False, allow_missing_hourly_times=False, allow_missing_creation_date=False, limit_by_avail_data=True,
           clobber=False, save_missing_geos_to=None, allow_alt_sites=False, site_id_override=None):
    run_settings = RunSettings(
        save_missing_geos_to=save_missing_geos_to, 
        last_month=last_month, 
        allow_missing_hourly_times=allow_missing_hourly_times,
        allow_missing_creation_date=allow_missing_creation_date,
        limit_by_avail_data=limit_by_avail_data
    )

    conversion_classes = {
        'mlo': MloMonthlyAverager(clobber=clobber, run_settings=run_settings), 
        'smo': SmoMonthlyAverager(smo_wind_file=geos_2d_file_list, clobber=clobber, run_settings=run_settings, allow_missing_geos_files=allow_missing_geos_files)
    }
    if site not in conversion_classes:
        raise InsituProcessingError('Unknown site: {}. Options are: {}'.format(site, ', '.join(conversion_classes.keys())))
    elif site == 'smo' and geos_2d_file_list is None:
        raise InsituProcessingError('When processing with site == "smo", geos_2d_file_list must be provided')

    converter = conversion_classes[site]
    converter.convert(hourly_insitu_file, previous_monthly_file, output_monthly_file, allow_alt_sites=allow_alt_sites, site_id_override=site_id_override)



def parse_args(p: ArgumentParser):
    def month(s):
        m = re.match(r'(\d{4})(\d{2})', s)
        if not m:
            raise ValueError('Bad string representation for last month: "{}". Must be in YYYY-MM format.'.format(s))
        year = int(m.group(1))
        mon = int(m.group(2))

        return pd.Timestamp(year, mon, 1)

        
    if p is None:
        p = ArgumentParser()
        is_main = True
    else:
        is_main = False

    last_month_str = DEFAULT_LAST_MONTH.strftime('%Y%m')

    p.description = 'Update monthly average surface CO2 files from new MLO/SMO hourly data'
    p.add_argument('site', choices=('mlo', 'smo'), help='Which site is being processed.')
    p.add_argument('previous_monthly_file', help='Path to the previous monthly average CO2 file for this site.')
    p.add_argument('hourly_insitu_file', help='Path to the latest file of hourly in situ analyzer data from NOAA')
    p.add_argument('output_monthly_file', help='Path to write the updated monthly averages to. Cannot be the same as the '
                                               'previous_monthly_file')
    p.add_argument('-l', '--last-month', default=last_month_str, type=month,
                   help='The last month that should be added to the output monthly file, given in YYYYMM format. The default is %(default)s. '
                        'The default value is based on today\'s date; it is set to last month.')
    p.add_argument('--allow-missing-hourly-times', action='store_true',
                   help='By default, if the input hourly file is missing any single hour between the end of the '
                        'previous monthly file and the end of the month specified by --last-month, an error is raised. '
                        'Setting this flag reduces that error to a warning.')
    p.add_argument('--allow-missing-creation-date', action='store_true',
                   help='By default, the hourly file is searched for a line containing "description_creation-time" that '
                        'gives the time that the hourly file was created. If this line cannot be found, an error is raised. '
                        'Setting this flag reduces that error to a warning, but means that *if* the creation time is missing, '
                        'the code will not be able to distinguish between data that will always be missing and data missing '
                        'because it has not been acquired yet.')
    p.add_argument('--no-limit-by-avail-data', dest='limit_by_avail_data', action='store_false',
                   help='By default, data in the hourly file will only be used up to the start of the month it was created in. '
                        'Set this flag to ignore the limits of the data in the hourly file, either based on the creation date or '
                        'the actual end of data in the file. Use this flag with caution as it can easily result in data from '
                        'the current month that has not undergone QA/QC by NOAA being used!')
    p.add_argument('-g', '--geos-2d-file-list', help='Path to a file containing a list of paths to GEOS surface files, '
                                                     'one per line. This is required when processing SMO data. There must '
                                                     'be one GEOS file for every 3 hours in every month being added. That is, '
                                                     'if adding data for Aug 2021, all 737 GEOS 2D files from 2021-08-01 00:00 '
                                                     'to 2021-09-01 00:00 (inclusive) must be listed.')
    p.add_argument('--save-missing-geos-to', default=None,
                   help='Write times for missing GEOS surface files to a file. If none are missing, the file will be empty.')
    p.add_argument('--allow-missing-geos-files', action='store_true', help='Prevent this program from erroring if GEOS files '
                                                                           'are missing, will still issue a warning and list '
                                                                           'the missing files if --save-missing-geos-to is given.')

    site_ex_args = p.add_mutually_exclusive_group()
    site_ex_args.add_argument('--allow-alt-noaa-site', dest='allow_alt_sites', action='store_true',
                              help='Override the check that the input monthly and hourly files contain data from the correct NOAA site.')
    site_ex_args.add_argument('--site-id-override', help='Override the site ID written to the new row(s) in the output monthly file with '
                                                         'the value given to this option, e.g. --site-id-override=XXX will use "XXX" in the '
                                                         'output file. This permits ingesting hourly files with mixed site IDs. If you only '
                                                         'need to allow ingesting an hourly file for a site with a different ID than given '
                                                         'by the SITE argument, prefer --allow-alt-noaa-site over this option.')
    p.add_argument('-c', '--clobber', action='store_true', help='By default, the output file will not overwrite any existing '
                                                                'file at that path. Setting this flag will cause it to overwrite '
                                                                'existing files UNLESS that file is the previous monthly file. '
                                                                'Overwriting that file is always forbidden.')

    p.set_defaults(driver_fxn=driver)
    if is_main:
        return vars(p.parse_args())
