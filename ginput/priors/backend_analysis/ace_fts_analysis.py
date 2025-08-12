from __future__ import print_function, division, absolute_import

from collections import OrderedDict
import datetime as dt
from glob import glob
from itertools import repeat
from matplotlib import pyplot as plt
from multiprocessing import Pool
import netCDF4 as ncdf
import numpy as np
import pandas as pd
import os
import re
from scipy.optimize import curve_fit
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import TukeyBiweight
import xarray as xr

from ...common_utils import mod_utils, ioutils, sat_utils
from ...common_utils.ggg_logging import logger
from .. import tccon_priors
from ...mod_maker import mod_maker as mm
from .backend_utils import read_ace_var, read_ace_date, read_ace_theta, read_ace_latlon

_mydir = os.path.abspath(os.path.dirname(__file__))
_tccon_top_alt = 70.0


def _bin_centers(bins):
    return bins[:-1] + np.diff(bins)/2


_std_age_bins = np.arange(0.0, 10.0, 0.25)
_std_age_bin_centers = _bin_centers(_std_age_bins)
_std_frac_bins = np.arange(0.0, 1.05, 0.05)
_std_frac_bin_centers = _bin_centers(_std_frac_bins)
_std_theta_bins = np.concatenate([np.arange(380, 1080, 50), np.arange(1080, 1680, 100), np.arange(1680, 3680, 200)])
_std_theta_bin_centers = _bin_centers(_std_theta_bins)


def ace_lut_driver(ace_dir, rw_supp_table_file, geos_path, lut_save_dir, overwrite_age_file=True, nprocs=0):
    """Generate all the ACE-derived look-up tables needed for ginput

    Parameters
    ----------
    ace_dir : str
        Directory to the ACE-FTS data. Expects the ACEs files therein to be named "ACE*{specie}.nc", e.g.
        "ACEFTS_L2_v3p6_N2O.nc".

    rw_supp_table_file : str
        Path to text file containing Table S3.1 from Washenfelder et al. 2003 (doi:10.1029/2003GL017969). This is
        usually "washenfelder03_table_s3.txt" in ginput/data.

    geos_path : str
        Path to the GEOS-FPIT met directory containing the "Nv" subdirectory with eta-level GEOS met files.

    lut_save_dir : str
        Path to save the newly generated LUT files to

    overwrite_age_file : bool
        Whether or not to recreate and overwrite the ACE-FTS "AGE" file if it exists.

    nprocs : int
        Number of processors to use when generating the ACE AGE file.

    Returns
    -------
    None
        Writes 3 files (ace_fn2o_lut.nc, n2o_ch4_acefts.nc, and ch4_hf_slopes.nc) to `lut_save_dir` and an ACE "AGE"
        file to `ace_dir` if either that file does not exist or `overwrite_age_file` is `True`.

    """
    # Make a dictionary of the available ACE files by species
    ace_files = {re.search(r'[A-Z0-9]+(?=\.nc)', os.path.basename(f)).group(): f for f in glob(os.path.join(ace_dir, 'ACE*.nc'))}

    if 'AGE' not in ace_files or overwrite_age_file:
        age_file = re.sub(r'P.nc$', 'AGE.nc', ace_files['P'])
        logger.info('Creating ACE-FTS AGE file. Will save to {}. Using {} processors'.format(age_file, nprocs))
        generate_ace_age_file(ace_in_file=ace_files['P'], age_file_out=age_file, geos_path=geos_path,
                              use_geos_theta_for_age=False, nprocs=nprocs)
        ace_files['AGE'] = age_file

    fn2o_save_file = os.path.join(lut_save_dir, 'ace_fn2o_lut.nc')
    logger.info('Creating F(N2O) look up table file. Will save to {}.'.format(fn2o_save_file))
    make_fn2o_lookup_table(ace_age_file=ace_files['AGE'], ace_n2o_file=ace_files['N2O'], lut_save_file=fn2o_save_file)

    fch4_save_file = os.path.join(lut_save_dir, 'n2o_ch4_acefts.nc')
    logger.info('Creating F(CH4):F(N2O) look up table file. Will save to {}.'.format(fch4_save_file))
    make_fch4_fn2o_lookup_table(ace_age_file=ace_files['AGE'], ace_n2o_file=ace_files['N2O'], ace_ch4_file=ace_files['CH4'],
                                lut_save_file=fch4_save_file)

    hf_slope_save_file = os.path.join(lut_save_dir, 'ch4_hf_slopes.nc')
    logger.info('Creating CH4:HF slope look up table file. Will save to {}'.format(hf_slope_save_file))
    make_hf_ch4_slopes(ace_ch4_file=ace_files['CH4'], ace_hf_file=ace_files['HF'], ace_age_file=ace_files['AGE'],
                       washenfelder_supp_table_file=rw_supp_table_file, lut_save_file=hf_slope_save_file,
                       clip_values=True)


def make_fn2o_lookup_table(ace_age_file, ace_n2o_file, lut_save_file):
    logger.info('Loading ACE data')
    ace_alt, ace_fn2o, ace_theta, extra_info = calc_fraction_remaining_from_acefts(ace_n2o_file, 'N2O', bc_approach='fit')

    ace_dates = extra_info['dates']
    ace_doy = np.array([int(mod_utils.clams_day_of_year(date)) for date in ace_dates.flat]).reshape(ace_dates.shape)

    with ncdf.Dataset(ace_n2o_file, 'r') as nch:
        ace_lat = read_ace_var(nch, 'latitude', None)

    with ncdf.Dataset(ace_age_file, 'r') as nch:
        ace_age = nch.variables['age'][:].filled(np.nan)

    logger.info('Binning F(N2O) vs. age and theta')
    # Similar to the F(CH4) vs. F(N2O) approach, we ignore fill values, want only positive F(N2O) values, want to
    # avoid the polar vortex and want to stay within TCCON relevant altitudes.
    #
    # is_vortex expects everything to be the same size, so we need to tile lat and doy
    ace_lat = np.tile(ace_lat[:, np.newaxis], [1, ace_age.shape[1]])
    ace_doy = np.tile(ace_doy[:, np.newaxis], [1, ace_age.shape[1]])

    xx = ~np.isnan(ace_fn2o) & ~np.isnan(ace_age) & (ace_fn2o >= 0) & \
         ~mod_utils.is_vortex(ace_lat, ace_doy, ace_age) & (ace_alt[np.newaxis, :] < _tccon_top_alt)

    age_bins = _std_age_bins
    age_bin_centers = _bin_centers(age_bins)
    theta_bins = _std_theta_bins
    theta_bin_centers = _bin_centers(theta_bins)

    # We're not going to worry about outliers initially
    fn2o_means, fn2o_counts, fn2o_overall, theta_overall = _bin_z_vs_xy(ace_age, ace_theta, ace_fn2o, xx, age_bins, theta_bins)

    fn2o_means[fn2o_means > 1] = 1
    fn2o_overall[fn2o_overall > 1] = 1

    _save_fn2o_lut(lut_save_file, fn2o_means=fn2o_means, fn2o_counts=fn2o_counts, fn2o_overall=fn2o_overall,
                   theta_overall=theta_overall, age_bin_centers=age_bin_centers, age_bin_edges=age_bins,
                   theta_bin_centers=theta_bin_centers, theta_bin_edges=theta_bins,
                   ace_n2o_file=ace_n2o_file, ace_age_file=ace_age_file)


def make_fch4_fn2o_lookup_table(ace_age_file, ace_n2o_file, ace_ch4_file, lut_save_file):
    logger.info('Reading ACE data')
    ace_alt, ace_fn2o, ace_theta, extra_info = calc_fraction_remaining_from_acefts(ace_n2o_file, 'N2O', bc_approach='fit')
    ace_dates = extra_info['dates']
    ace_doy = np.array([int(mod_utils.clams_day_of_year(date)) for date in ace_dates.flat]).reshape(ace_dates.shape)

    _, ace_fch4, _, _ = calc_fraction_remaining_from_acefts(ace_ch4_file, 'CH4', bc_approach='fit')

    with ncdf.Dataset(ace_age_file, 'r') as nch:
        ace_age = nch.variables['age'][:].filled(np.nan)

    with ncdf.Dataset(ace_ch4_file, 'r') as nch:
        ace_ch4_raw = nch.variables['CH4'][:].filled(np.nan)
        ace_lat = nch.variables['latitude'][:].filled(np.nan)

    logger.info('Binning F(CH4) vs. F(N2O)')
    # We're looking for reliable stratospheric relationships. Therefore we limit to data where the concentration is
    # positive and definitely not tropospheric (CH4 < 2e-6 i.e. < 2000 ppb), not in the polar vortex (abs(lat) < 50)
    # and not in the mesosphere or upper stratosphere (alt < 40).
    ace_lat = np.tile(ace_lat[:, np.newaxis], [1, ace_age.shape[1]])
    ace_doy = np.tile(ace_doy[:, np.newaxis], [1, ace_age.shape[1]])
    xx = ~np.isnan(ace_fch4) & ~np.isnan(ace_fn2o) & (ace_fch4 >= 0) & (ace_fn2o >= 0) & (ace_ch4_raw < 2e-6) & \
         ~mod_utils.is_vortex(ace_lat, ace_doy, ace_age) & (ace_alt[np.newaxis, :] < _tccon_top_alt)

    # Define bins for F(N2O) and theta
    fn2o_bins = _std_frac_bins
    fn2o_bin_centers = _std_frac_bin_centers

    theta_bins = _std_theta_bins
    theta_bin_centers = _std_theta_bin_centers

    # Find outliers to avoid overly noisy relationships
    oo = _find_ch4_outliers(ace_fn2o, ace_fch4, fn2o_bins, xx)

    # Do the actual binning
    fch4_means, fch4_counts, fch4_overall, theta_overall = _bin_z_vs_xy(ace_fn2o, ace_theta, ace_fch4, xx & oo,
                                                                        fn2o_bins, theta_bins)

    logger.info('Saving F(CH4):F(N2O) file')
    _save_fch4_lut(lut_save_file, fch4_means=fch4_means, fch4_counts=fch4_counts, fch4_overall=fch4_overall,
                   theta_overall=theta_overall, fn2o_bin_centers=fn2o_bin_centers, fn2o_bin_edges=fn2o_bins,
                   theta_bin_centers=theta_bin_centers, theta_bin_edges=theta_bins,
                   ace_ch4_file=ace_ch4_file, ace_n2o_file=ace_n2o_file, ace_age_file=ace_age_file)


def make_hf_ch4_slopes(ace_ch4_file, ace_hf_file, ace_age_file, washenfelder_supp_table_file, lut_save_file,
                       clip_values=True):

    logger.info('Loading ACE data')
    with ncdf.Dataset(ace_ch4_file, 'r') as nch_ch4, ncdf.Dataset(ace_hf_file, 'r') as nch_hf, ncdf.Dataset(ace_age_file, 'r') as nch_age:
        ace_ch4_qual = nch_ch4.variables['quality_flag'][:].filled(9)
        ace_hf_qual = nch_hf.variables['quality_flag'][:].filled(9)

        ace_ch4 = read_ace_var(nch_ch4, 'CH4', ace_ch4_qual)
        ace_ch4_err = read_ace_var(nch_ch4, 'CH4_error', ace_ch4_qual)
        ace_hf = read_ace_var(nch_hf, 'HF', ace_hf_qual)
        ace_hf_err = read_ace_var(nch_hf, 'HF_error', ace_hf_qual)
        ace_ages = read_ace_var(nch_age, 'age', None)

        ace_lat = read_ace_var(nch_ch4, 'latitude', None)
        ace_lat = np.tile(ace_lat.reshape(-1, 1), [1, ace_ch4.shape[1]])
        ace_alt = read_ace_var(nch_ch4, 'altitude', None)
        ace_alt = np.tile(ace_alt, [ace_ch4.shape[0], 1])
        ace_theta = read_ace_theta(nch_ch4, ace_ch4_qual)

        ace_year = nch_ch4.variables['year'][:].filled(np.nan)
        ace_dates = read_ace_date(nch_ch4)
        ace_doy = np.array([mod_utils.day_of_year(d) + 1 for d in ace_dates])
        ace_doy = ace_doy.reshape(-1, 1)

    logger.info('Calculating CH4 vs. HF slopes')
    # We do the same filtering as for the F(N2O):F(CH4) relationship, plus we require that the CH4 and HF error is < 5%.
    # Also have to filter for excessively high HF concentrations, or it will mess up the tropics. Generally HF doesn't
    # exceed 3 ppb and the ultra high values were all > 200 ppb, so 10 should leave all reasonable data and exclude
    # unreasonable data (this occurred in tropics 2005)
    #
    # Quality filtering and fill values were handled upon reading in the data.
    xx = ~np.isnan(ace_ch4) & ~np.isnan(ace_ch4_err) & ~np.isnan(ace_hf) & ~np.isnan(ace_hf_err) & (ace_alt < _tccon_top_alt)

    if clip_values:
        xx &= (ace_ch4 <= 2e-6) & (ace_hf < 10e-9)

    # Read in the early slopes from Washenfelder et al. 2003 (doi: 10.1029/2003GL017969), table S3.
    washenfelder_df = pd.read_csv(washenfelder_supp_table_file, header=2, sep=r'\s+')
    washenfelder_slopes = washenfelder_df.set_index('Date').b.astype(float)
    washenfelder_slopes.index = pd.DatetimeIndex(washenfelder_slopes.index)
    washenfelder_slopes.name = 'slope'

    # For each bin, get the slopes from the ACE-FTS data. Fit the data (including the Washenfelder slopes) to an
    # exponential.
    lat_bin_functions = OrderedDict([('tropics', mod_utils.is_tropics),
                                     ('midlat', mod_utils.is_midlat),
                                     ('vortex', mod_utils.is_vortex)])
    lat_bin_slopes = pd.DataFrame(index=_get_ace_date_range(ace_dates), columns=lat_bin_functions.keys())
    lat_bin_counts = lat_bin_slopes.copy()

    fit_params = np.zeros([len(lat_bin_functions), 4], dtype=float)

    for ibin, (bin_name, bin_fxn) in enumerate(lat_bin_functions.items()):
        for year_start_date in lat_bin_slopes.index:
            year = year_start_date.year
            logger.debug('Generating {} slopes for {}'.format(bin_name, year))
            lat_bin_slopes.loc[year_start_date, bin_name], lat_bin_counts.loc[year_start_date, bin_name] \
                = _bin_and_fit_hf_vs_ch4(ace_lat=ace_lat, ace_year=ace_year, ace_dates=ace_dates, ace_doy=ace_doy,
                                         ace_ages=ace_ages, ace_theta=ace_theta, ace_ch4=ace_ch4, ace_hf=ace_hf, xx=xx,
                                         year=year, bin_fxn=bin_fxn)

        full_series = pd.concat([washenfelder_slopes, lat_bin_slopes.loc[:, bin_name]])
        full_series = full_series.sort_index()
        full_series_year = full_series.index.year
        full_series = full_series.to_numpy().astype(float)

        # This was found to be a good initial guess for all latitude bins by trial and error. 3000.0 gets about the
        # right magnitude at year 0, 0.18 get about the right decay, using the last slope as the offset generally works
        # better than using the maximum (sometimes curve_fit has trouble moving this offset enough) and naturally we
        # want t0 to be the first year. Trying to allow curve_fit to choose its own initial conditions ends badly.
        p0 = [-3000.0, -0.18, full_series[-1], full_series_year[0]]
        fit = curve_fit(mod_utils.hf_ch4_slope_fit, full_series_year, full_series, p0=p0)
        fit_params[ibin, :] = fit[0]

    # for the netCDF file, we'll include both the fits and the actual slopes for the ACE-FTS era. The default is to
    # use the fits for all years.
    first_year = washenfelder_slopes.index.min().year
    last_year = lat_bin_slopes.index.max().year
    full_dtindex = pd.date_range(start=dt.datetime(first_year, 1, 1), end=dt.datetime(last_year, 1, 1), freq='YS')

    logger.info('Saving CH4 vs. HF slopes')
    _save_hf_ch4_lut(lut_save_file, ace_ch4_file=ace_ch4_file, ace_hf_file=ace_hf_file, ace_age_file=ace_age_file,
                     lat_bin_edges=lat_bin_functions, full_date_index=full_dtindex,
                     ace_slopes=lat_bin_slopes, ace_counts=lat_bin_counts, slope_fit_params=fit_params)

    return lat_bin_slopes, lat_bin_counts


def _find_ch4_outliers(ace_fn2o, ace_fch4, fn2o_bins, good_data):

    # Identify outliers in each F(N2O) bin. We're not binning by theta yet; we just want to find outliers in the dataset
    # as a whole right now. I chose m=5 since 5 sigma is the standard threshold in physics, even through we're using
    # mean absolute deviation and not standard deviation, since the former is more robust.
    #
    # oo will be true for non-outlier values
    oo = np.zeros_like(ace_fch4, dtype=np.bool_)
    bin_inds = np.digitize(ace_fn2o, fn2o_bins) - 1
    for b_ind in np.unique(bin_inds):
        bb = good_data & (bin_inds == b_ind)
        ch4_sub = ace_fch4[bb]
        oo[bb] = ~mod_utils.isoutlier(ch4_sub, m=5)

    return oo


def _bin_z_vs_xy(x, y, z, good_data, x_bins, y_bins):
    n_x_bins = x_bins.size - 1
    n_y_bins = y_bins.size - 1
    x_means = np.full([n_x_bins, n_y_bins], np.nan)
    x_counts = np.zeros_like(x_means, dtype=int)

    for i, (tlow, thigh) in enumerate(zip(y_bins[:-1], y_bins[1:])):
        aa = good_data & (y >= tlow) & (y < thigh)
        x_sub = x[aa]
        z_sub = z[aa]
        bin_inds = np.digitize(x_sub, x_bins) - 1

        for j in range(n_x_bins):
            jj = bin_inds == j
            x_means[j, i] = np.nanmean(z_sub[jj])
            x_counts[j, i] = np.sum(jj)

    z_overall = np.full([n_x_bins], np.nan)
    y_overall = np.full([n_x_bins], np.nan)
    bin_inds = np.digitize(x, x_bins) - 1

    for j in range(n_x_bins):
        jj = (bin_inds == j) & good_data
        z_overall[j] = np.nanmean(z[jj])
        y_overall[j] = np.nanmean(y[jj])

    return x_means, x_counts, z_overall, y_overall


def _bin_and_fit_hf_vs_ch4(ace_lat, ace_year, ace_dates, ace_doy, ace_ages, ace_theta, ace_ch4, ace_hf, xx, year,
                           bin_fxn):
    yy = bin_fxn(ace_lat, ace_doy, ace_ages) & (np.isclose(ace_year, year)).reshape(-1, 1)

    profile_ch4_sbc = get_bc_from_ace(ace_dates=ace_dates, gas_conc=ace_ch4, theta=ace_theta, latitude=ace_lat[:, 0],
                                      bc_approach='fit')

    x = ace_hf[xx & yy] * 1e9
    y = (ace_ch4 - profile_ch4_sbc.reshape(-1, 1))
    y = y[xx & yy] * 1e9

    # We use the robust fitting method with Tukey's biweighting as in Saad et al. 2014. That paper used `robustfit` from
    # Matlab, RLM from statsmodels is the Python equivalent. We do not include an intercept because we are using the
    # CH4 stratospheric boundary condition as the intercept, and we've already subtracted that.
    slope = RLM(y, x, M=TukeyBiweight()).fit().params.item()

    return slope, np.sum(xx & yy)


def _get_year_avg_ch4_sbc(year, ch4):
    tt = (ch4.conc_seasonal.index >= dt.datetime(year, 1, 1)) & (ch4.conc_seasonal.index < dt.datetime(year + 1, 1, 1))
    return ch4.conc_seasonal.dmf_mean[tt].mean()


def _save_fn2o_lut(nc_filename, fn2o_means, fn2o_counts, fn2o_overall, theta_overall,
                   age_bin_centers, age_bin_edges, theta_bin_centers, theta_bin_edges,
                   ace_n2o_file, ace_age_file):

    with ncdf.Dataset(nc_filename, 'w') as nch:
        age_dim = ioutils.make_ncdim_helper(nch, 'age', age_bin_centers,
                                            description='Age of stratospheric air from the CLaMS model',
                                            units='years')
        ioutils.make_ncdim_helper(nch, 'age_bins', age_bin_edges,
                                  description='Edges of age bins used when binning the F(N2O) data',
                                  units='years')
        theta_dim = ioutils.make_ncdim_helper(nch, 'theta', theta_bin_centers,
                                              description='Potential temperature',
                                              units='K')
        ioutils.make_ncdim_helper(nch, 'theta_bins', theta_bin_edges,
                                  description='Edges of potential temperature bins used when binning the F(N2O) data',
                                  units='K')

        ioutils.make_ncvar_helper(nch, 'fn2o', fn2o_means, (age_dim, theta_dim),
                                  description='Mean value of F(N2O) (fraction of N2O remaining relative to the '
                                              'stratospheric boundary condition) in the age/theta bin',
                                  units='unitless')
        ioutils.make_ncvar_helper(nch, 'fn2o_counts', fn2o_counts, (age_dim, theta_dim),
                                  description='Number of F(N2O) observations in each age/theta bin',
                                  units='number')
        ioutils.make_ncvar_helper(nch, 'fn2o_overall', fn2o_overall, (age_dim,),
                                  description='Mean value of F(N2O) for a given age bin, not separated by theta',
                                  units='unitless')
        ioutils.make_ncvar_helper(nch, 'theta_overall', theta_overall, (age_dim,),
                                  description='Mean value of potential temperature in a given age bin',
                                  units='K')
        ioutils.add_dependent_file_hash(nch, 'ace_n2o_file_sha1', ace_n2o_file)
        ioutils.add_dependent_file_hash(nch, 'ace_age_file_sha1', ace_age_file)
        ioutils.add_creation_info(nch, 'make_fn2o_lookup_table')


def _save_fch4_lut(nc_filename, fch4_means, fch4_counts, fch4_overall, theta_overall,
                   fn2o_bin_centers, fn2o_bin_edges, theta_bin_centers, theta_bin_edges,
                   ace_ch4_file, ace_n2o_file, ace_age_file):
    with ncdf.Dataset(nc_filename, 'w') as nch:
        fn2o_dim = ioutils.make_ncdim_helper(nch, 'fn2o', fn2o_bin_centers,
                                             description='Fraction of N2O remaining relative to the stratospheric boundary condition.',
                                             units='unitless')
        ioutils.make_ncdim_helper(nch, 'fn2o_bins', fn2o_bin_edges,
                                  description='Edges of F(N2O) bins used when binning the F(CH4) data',
                                  units='unitless')
        theta_dim = ioutils.make_ncdim_helper(nch, 'theta', theta_bin_centers,
                                              description='Potential temperature',
                                              units='K')
        ioutils.make_ncdim_helper(nch, 'theta_bins', theta_bin_edges,
                                  description='Edges of potential temperature bins used when binning the F(CH4) data',
                                  units='K')

        ioutils.make_ncvar_helper(nch, 'fch4', fch4_means, (fn2o_dim, theta_dim),
                                  description='Mean value of F(CH4) (fraction of CH4 remaining relative to the '
                                              'stratospheric boundary condition) in the F(N2O)/theta bin',
                                  units='unitless')
        ioutils.make_ncvar_helper(nch, 'fch4_counts', fch4_counts, (fn2o_dim, theta_dim),
                                  description='Number of CH4 observations in each F(N2O)/theta bin',
                                  units='number')
        ioutils.make_ncvar_helper(nch, 'fch4_overall', fch4_overall, (fn2o_dim,),
                                  description='Mean value of F(CH4) for a given F(N2O) bin, not separated by theta',
                                  units='unitless')
        ioutils.make_ncvar_helper(nch, 'theta_overall', theta_overall, (fn2o_dim,),
                                  description='Mean value of potential temperature in a given F(N2O) bin',
                                  units='K')
        ioutils.add_creation_info(nch, 'make_fch4_fn2o_lookup_table')
        ioutils.add_dependent_file_hash(nch, 'ace_ch4_file_sha1', ace_ch4_file)
        ioutils.add_dependent_file_hash(nch, 'ace_n2o_file_sha1', ace_n2o_file)
        ioutils.add_dependent_file_hash(nch, 'ace_age_file_sha1', ace_age_file)


def _save_hf_ch4_lut(nc_filename, ace_ch4_file, ace_hf_file, ace_age_file, lat_bin_edges,
                     full_date_index, ace_slopes, ace_counts, slope_fit_params):
    with ncdf.Dataset(nc_filename, 'w') as nch:
        ioutils.add_creation_info(nch, creation_note='make_hf_ch4_slopes')
        ioutils.add_dependent_file_hash(nch, 'ace_ch4_file_sha1', ace_ch4_file)
        ioutils.add_dependent_file_hash(nch, 'ace_hf_file_sha1', ace_hf_file)
        ioutils.add_dependent_file_hash(nch, 'ace_age_file_sha1', ace_age_file)

        nch.ace_ch4_file = ace_ch4_file
        nch.ace_hf_file = ace_hf_file

        ch4_hf_slopes = pd.DataFrame(index=full_date_index, columns=ace_slopes.columns)
        ch4_hf_count = pd.DataFrame(index=full_date_index, columns=ace_counts.columns).fillna(0)
        ch4_hf_source = pd.DataFrame(index=full_date_index, columns=ace_slopes.columns).fillna(1)
        # transpose so that the bins dim is second like the above dataframes
        bin_names_char_array = ncdf.stringtochar(np.array(list(lat_bin_edges.keys()))).T
        slope_fit_params = slope_fit_params.T

        # For each latitude bin, make a pandas series for the full date index from the fit first, then replace available
        # years with the actual fit
        for ibin, bin_name in enumerate(lat_bin_edges.keys()):
            ch4_hf_slopes.loc[:, bin_name] = mod_utils.hf_ch4_slope_fit(ch4_hf_slopes.index.year, *slope_fit_params[:, ibin])
            ch4_hf_slopes.loc[ace_slopes.index, bin_name] = ace_slopes.loc[:, bin_name]
            ch4_hf_count.loc[ace_counts.index, bin_name] = ace_counts.loc[:, bin_name]
            ch4_hf_source.loc[ace_slopes.index, bin_name] = 0

        # Setup dimensions.
        nbins = len(lat_bin_edges)
        nchars = bin_names_char_array.shape[0]
        nparams = slope_fit_params.shape[0]
        year_dim = ioutils.make_ncdim_helper(nch, 'year', full_date_index.year.to_numpy())
        bins_dim = nch.createDimension('latitude_bins', size=nbins)
        char_dim = nch.createDimension('bin_name_length', size=nchars)
        fit_params_dim = nch.createDimension('exp_fit_params_length', size=nparams)

        # Save the variables
        ioutils.make_ncvar_helper(nch, 'ch4_hf_slopes', ch4_hf_slopes.to_numpy().astype(float), [year_dim, bins_dim],
                                  units='ppb CH4/ppb HF',
                                  description='Slope of stratospheric CH4 concentrations vs HF concentrations from '
                                              'ACE-FTS data for specific years and latitude bins')
        ioutils.make_ncvar_helper(nch, 'ace_counts', ch4_hf_count.to_numpy().astype(float), [year_dim, bins_dim],
                                  units='#',
                                  description='Number of ACE-FTS observations used to calculate this slope')
        ioutils.make_ncvar_helper(nch, 'ch4_hf_slopes_source', ch4_hf_source.to_numpy().astype(int), [year_dim, bins_dim],
                                  units='N/A',
                                  description='Flag indicating how the slope was computed. 0 means taken directly from '
                                              'fits of ACE-FTS CH4 vs. HF; 1 means sampled from the exponential fit '
                                              'of ACE-FTS + Washenfelder 03 slopes')
        ioutils.make_ncvar_helper(nch, 'slope_fit_params', slope_fit_params, [fit_params_dim, bins_dim],
                                  description='Fitting parameters a, b, c, and t0 for a function of form '
                                              '"a * exp(b*(t - t0)) + c" used to fit the temporal trend of CH4 vs. HF '
                                              'slopes, where t is the year (e.g. 2004)')
        ioutils.make_ncvar_helper(nch, 'bin_names', bin_names_char_array, [char_dim, bins_dim],
                                  description='The names used for the latitude bins')


def plot_hf_ch4_slopes(lut_file, years=np.arange(1970, 2019), ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    colors = ['r', 'g', 'b']
    linestyles = ['-', '--', ':']
    with xr.open_dataset(lut_file) as ds:
        lat_bins = ds['latitude_bins'][:]
        params = ds['slope_fit_params'][:]
        bin_names = ds['bin_names'][:]
        for i, col, style in zip(lat_bins, colors, linestyles):
            label = bin_names.sel(latitude_bins=i).data
            try:
                label = ''.join(label)
            except TypeError:
                label = b''.join(label).decode('utf8')

            slopes = mod_utils.hf_ch4_slope_fit(years, *params.sel(latitude_bins=i).data)
            ax.plot(years, slopes, color=col, linestyle=style, label=label)

    ax.grid()
    ax.set_ylabel('CH4:HF slope from fit')
    ax.legend()

    return fig, ax


def get_bc_from_ace(ace_dates, gas_conc, theta, latitude, bc_approach='fit'):
    """
    Calcaulate the stratospheric boundary condition for each ACE profile from ACE data itself

    :param ace_dates: a 1D array of datetimes with one entry per ACE profile
    :type ace_dates: array-like

    :param gas_conc: the array of ACE concentrations for the gas of interest. May be any units, the boundary condition
     values will be returned in the same units.
    :type gas_conc: array-like

    :param theta: the array of ACE potential temperatures.
    :type theta: array-like

    :param latitude: the array of ACE profile latitudes (1D)
    :type latitude: array-like

    :param bc_approach: how to calculate the boundary condition. "fit" (default) fits all ACE data in the tropics
     between 360 and 390 K to generate a quadratic fit over time. "per-profile" calculates the boundary condition for
     each profile using its own data between 360 and 390 K.
    :type bc_approach: str

    :return: the array of boundary condition concentrations, one per profile.
    """
    bottom_pt = 360
    top_pt = 390

    def _get_bc_per_profile(dates, concentration, pt, lat):
        bc_conc = np.full([concentration.shape[0]], np.nan)
        for i, (conc_row, pt_row) in enumerate(zip(concentration, pt)):
            zz = (pt_row > bottom_pt) & (pt_row < top_pt)

            if zz.sum() > 0:
                bc_conc[i] = np.nanmean(conc_row[zz])

        return bc_conc

    def _get_bc_by_fit(dates, concentration, pt, lat):
        datenums = np.full_like(concentration, np.nan)
        datenums[:] = np.array([np.datetime64(d) for d in dates]).reshape(-1, 1)
        in_pt = (pt > 360) & (pt < 390)
        in_tropics = mod_utils.is_tropics(lat, None, None)
        in_both = in_pt & in_tropics.reshape(-1, 1)
        bc_concs = concentration[in_both]
        bc_datenums = datenums[in_both]
        not_outliers = ~mod_utils.isoutlier(bc_concs, m=5)
        not_nans = ~np.isnan(bc_datenums) & ~np.isnan(bc_concs)
        xx = not_outliers & not_nans
        fit = np.polynomial.polynomial.Polynomial.fit(bc_datenums[xx], bc_concs[xx], deg=2)
        return fit(datenums[:, 0])

    if bc_approach == 'per-profile':
        bc_fxn = _get_bc_per_profile
    elif bc_approach == 'fit':
        bc_fxn = _get_bc_by_fit
    else:
        raise ValueError('bc_approach must be "per-profile" or "fit"')

    return bc_fxn(ace_dates, gas_conc, theta, latitude)


def calc_fraction_remaining_from_acefts(nc_file, gas_name, bc_approach='per-profile'):
    """
    Calculate the fraction remaining of a gas in the stratosphere from an ACE-FTS netCDF file

    :param nc_file: the path to the ACE-FTS netCDF file
    :type nc_file: str

    :param gas_name: the variable name in the netCDF file that holds the gas concentrations
    :type gas_name: str

    :param gas_record: a subclass instance of :class:`TropicsTraceGasRecord` that provides the stratospheric boundary
     condition for the given gas.
    :type gas_record: :class:`tccon_priors.TropicsTraceGasRecord`

    :param tropopause_approach: how to find the tropopause. Options are:

        * 'wmo' - uses the WMO definition, looking for lapse rate < 2 K/km
        * 'theta' - finds the altitude at which potential temperature is 380 K

    :type tropopause_approach: str

    :return: the vector of altitudes that the ACE-FTS profiles are defined on and the array of fraction of gas
     remaining. The latter will be set to NaN in the troposphere.
    :rtype: :class:`numpy.ndarray` x2
    """

    with ncdf.Dataset(nc_file, 'r') as nch:
        alt = read_ace_var(nch, 'altitude', qflags=None)
        latitude = read_ace_var(nch, 'latitude', qflags=None)
        ace_dates = read_ace_date(nch)
        qflags = read_ace_var(nch, 'quality_flag', qflags=None)

        gas_conc = read_ace_var(nch, gas_name, qflags=qflags)
        # ACE data appears to use -999 as fill value in the concentrations
        gas_conc[gas_conc < -900.0] = np.nan

        theta = read_ace_theta(nch, qflags)

    bc_concentrations = get_bc_from_ace(ace_dates, gas_conc, theta, latitude, bc_approach=bc_approach)
    fgas = gas_conc / bc_concentrations.reshape(-1, 1)
    fgas[theta < 380] = np.nan

    return alt, fgas, theta, {'dates': ace_dates, 'bc_concs': bc_concentrations, 'gas_concs': gas_conc}


def _get_ace_date_range(ace_dates, freq='YS'):
    min_date = np.min(ace_dates)
    if min_date < dt.datetime(min_date.year, 4, 1):
        start_year = min_date.year
    else:
        start_year = min_date.year + 1

    max_date = np.max(ace_dates)
    if max_date >= dt.datetime(max_date.year, 10, 1):
        end_year = max_date.year
    else:
        end_year = max_date.year - 1

    return pd.date_range(start=dt.datetime(start_year, 1, 1), end=dt.datetime(end_year, 1, 1), freq=freq)


def add_clams_age_to_file(ace_nc_file, save_nc_file=None):
    """
    Calculate CLaMS age for profiles in an ACE-FTS files. Save either to the given ACE file or a new file.

    Note: this function does not calculate equivalent latitude for the ACE profiles before computing the age, it simply
    uses the geographic latitude of the ACE profiles.

    :param ace_nc_file: the ACE-FTS netCDF file to compute ages for.
    :type ace_nc_file: str

    :param save_nc_file: optional, if given a path to a file, the ages will be saved to this file (which is overwritten)
     along with the altitude dimension. If not given (or ``None``), the ages are added to the file specified by
     ``ace_nc_file``.
    :type save_nc_file: str

    :return: None
    """
    with ncdf.Dataset(ace_nc_file, 'r') as nch:
        ace_lat = read_ace_var(nch, 'latitude', None)
        ace_alt = read_ace_var(nch, 'altitude', None)
        ace_dates = read_ace_date(nch)
        ace_doy = np.array([mod_utils.day_of_year(d) + 1 for d in ace_dates])
        ace_theta = read_ace_theta(nch, None)

    ace_ages = np.full_like(ace_theta, np.nan)
    pbar = mod_utils.ProgressBar(ace_theta.shape[0], prefix='Calculating ages for profile', style='counter')
    for iprof in range(ace_theta.shape[0]):
        pbar.print_bar(iprof)
        prof_theta = ace_theta[iprof]
        prof_lat = np.full_like(prof_theta, ace_lat[iprof])
        prof_doy = ace_doy[iprof]
        ace_ages[iprof, :] = tccon_priors.get_clams_age(prof_theta, prof_lat, prof_doy)

    pbar.finish()

    if save_nc_file is None:
        logger.info('Added CLaMS age to input file: {}'.format(ace_nc_file))
        with ncdf.Dataset(ace_nc_file, 'a') as nch:
            ioutils.make_ncvar_helper(nch, 'age', ace_ages, ['index', 'altitude'], units='years',
                                      description='Age from the CLaMS model along the ACE profiles')
    else:
        logger.info('Saving CLaMS age to new file: {}'.format(save_nc_file))
        with ncdf.Dataset(save_nc_file, 'w') as nch:
            index_dim = nch.createDimension('index', ace_ages.shape[0])
            alt_dim = ioutils.make_ncdim_helper(nch, 'altitude', ace_alt, long_name='altitude',
                                                description='Fixed altitude grid', units='km')
            ioutils.make_ncvar_helper(nch, 'age', ace_ages, [index_dim, alt_dim], units='years', long_name='clams_age',
                                      description='Age from the CLaMS model along the ACE profiles')


def generate_ace_age_file(ace_in_file, age_file_out, geos_path, use_geos_theta_for_age=False, geos_product='fpit', allow_missing_geos=False, nprocs=0):
    """Create a file that mimics the structure of an ACE-FTS v3 file but stores age of air and equivalent latitude

    Parameters
    ----------
    ace_in_file: str
        Path to any of the original ACE files. It doesn't need to be any one in particular; all should have the lat,
        lon, pressure, temperature, quality flag, and date information needed. However, if this is for a particular
        gas, you may as well pass that gas's file, in case the quality flags differ.

    age_file_out: str
        Path to write the resulting file to (the file itself, not a directory).

    geos_path: str
        Path to the directory containing the GEOS met data. This directory must contain an "Nv" subdirectory with the
        3D met files. 

    use_geos_theta_for_age: bool
        Set to ``True`` to use potential temperature calculated from the GEOS files, rather than ACE, when determining
        the age of air.

    geos_product: str
        A string, usually "fpit", "fp", or "it" that describes which GEOS product to read. This is necessary to determine
        the file names.

    allow_missing_geos: bool
        Set to ``True`` to skip profiles for which the necessary GEOS files are not available. Otherwise, any missing GEOS
        file is an error.

    nprocs: int
        How many processors to use. <= 1 will run in serial, >= 2 will use multiprocessing to run in parallel.
    """
    # This will need to group ACE profiles by which GEOS files they fall between, read the PV from those files,
    # interpolate to the ACE profile lat/lon/time, generate the EL interpolators, calculate the EL profiles, then
    # finally lookup age from CLAMS.
    logger.info('Generating ACE age file with NATIVE-level GEOS files')
    ace_data = dict()
    with ncdf.Dataset(ace_in_file) as nh:
        ace_data['dates'] = read_ace_date(nh)
        ace_qflags = read_ace_var(nh, 'quality_flag', None)
        # This wraps lons outside [-180,180] and lats outside [-90,90] into
        # those ranges
        ace_data['lon'], ace_data['lat'] = read_ace_latlon(nh, clip=True)
        ace_data['theta'] = read_ace_theta(nh, qflags=ace_qflags)

    date_groups = _bin_ace_to_geos_times(ace_data['dates'])

    # Now we need to loop over the date groups and calculate the eq. lat. and age for each group (since each group
    # is the ace profiles in a 3 hour window bracketed by 2 GEOS files). There's usually only a few profiles per group,
    # so it makes more sense to parallelize over groups than profiles within groups.
    #
    # Even though it may be memory intensive, we're going to build the inputs for each call to _calc_el_age_for_ace
    # first then iterate so that we can use the same inputs for serial or parallel mode. We'll build as iterators to
    # limit the memory use as much as possible.
    group_inputs = dict()
    group_inputs['ace_dates'] = (ace_data['dates'][xx] for xx in date_groups.values())
    group_inputs['ace_lons'] = (ace_data['lon'][xx] for xx in date_groups.values())
    group_inputs['ace_lats'] = (ace_data['lat'][xx] for xx in date_groups.values())
    group_inputs['ace_theta'] = (ace_data['theta'][xx, :] for xx in date_groups.values())
    group_inputs['geos_dates'] = ((d, d+dt.timedelta(hours=3)) for d in date_groups.keys())
    group_inputs['geos_path'] = repeat(geos_path)
    group_inputs['use_geos_theta'] = repeat(use_geos_theta_for_age)
    group_inputs['geos_product'] = repeat(geos_product)
    group_inputs['allow_missing'] = repeat(allow_missing_geos)

    input_order = ('ace_dates', 'ace_lons', 'ace_lats', 'ace_theta', 'geos_dates', 'geos_path', 'use_geos_theta', 'geos_product', 'allow_missing')
    inputs_interator = zip(*[group_inputs[k] for k in input_order])

    if nprocs <= 1:
        results = []
        for inputs in inputs_interator:
            results.append(_calc_el_age_for_ace(*inputs))
    else:
        with Pool(processes=nprocs) as pool:
            results = pool.starmap(_calc_el_age_for_ace, inputs_interator)

    # _calc_el_age_for_age returns a dictionary of nprofs-by-nlevels arrays, where nprofs is the number of profiles in
    # the group it was given. results is then a list of these dictionaries, so we need to go through those dictionaries
    # and put their subset of profiles in the right rows in the bigger array
    output_keys = ('EL', 'age')
    ace_outputs = {k: np.full_like(ace_data['theta'], np.nan) for k in output_keys}
    for i, inds in enumerate(date_groups.values()):
        if results[i] is None and allow_missing_geos:
            continue
        elif results[i] is None and not allow_missing_geos:
            raise TypeError('Got a None value when missing GEOS files were forbidden')
        
        for k in output_keys:
            ace_outputs[k][inds] = results[i][k]

    save_ace_age_file(age_file_out, ('longitude', 'latitude', 'orbit', 'year', 'month', 'day', 'hour', 'quality_flag', 'temperature', 'pressure'),
                      ace_in_file, ace_outputs['EL'], ace_outputs['age'])
    logger.info('Done generating ACE age file with NATIVE-level GEOS files')


def _calc_el_age_for_ace(ace_dates, ace_lon, ace_lat, ace_theta, geos_dates, geos_path, use_geos_theta_for_age=False, geos_product='fpit', allow_missing=False):
    # Let's get to the point where we load the GEOS data
    date_str = ', '.join(d.strftime('%Y-%m-%d %H:%M') for d in ace_dates)
    logger.info('Calculating eqlat and age for profiles on {}'.format(date_str))
    # Read in the GEOS data, calculating quantities as necessary #
    geos_vars = {'EPV': 1e6, 'T': 1.0, 'DELP': 1.0}

    try:
        geos_files = [os.path.join(geos_path, 'Nv', mod_utils._format_geosfp_name(geos_product, 'met', 'eta', d)) for d in geos_dates]
        missing_files = ', '.join(f for f in geos_files if not os.path.exists(f))
        if missing_files:
            if allow_missing:
                date_range = f'{min(ace_dates)} to {max(ace_dates)}'
                logger.warn(f'Missing at least one GEOS file for {date_range}: {missing_files}')
                return None
            else:
                raise IOError(f'Missing GEOS files: {missing_files}')
        geos_data_on_std_times = []
        geos_pres = []
        for i, f in enumerate(geos_files):
            this_data = _interp_geos_vars_to_ace_lat_lon(f, geos_vars, ace_lon, ace_lat)
            geos_pres.append(np.fliplr(mod_utils.convert_geos_eta_coord(this_data.pop('DELP'))))
            
            # Handle flipping vertically after computing pressure in case the convert function expects the
            # delp vector space-to-surface. Flipping lr instead of ud because the flip has to go along the
            # second dimension
            for k, v in this_data.items():
                this_data[k] = np.fliplr(v)
                
            geos_data_on_std_times.append(this_data)

        # Generate the eq. lat. intepolators and find the eq. lat. profiles on the GEOS levels #
        eqlat_interpolators = mm.equivalent_latitude_functions_from_native_geos_files(geos_files, geos_dates, muted=True)
        for i, geos_data in enumerate(geos_data_on_std_times):
            gdate = geos_dates[i]
            geos_data['PT'] = mod_utils.calculate_potential_temperature(geos_pres[i], geos_data['T'])
            # Calculate the equivalent latitude profiles.
            eqlat = np.full_like(geos_data['T'], np.nan)
            for j in range(ace_lon.size):
                eqlat[j, :] = mod_utils.get_eqlat_profile(eqlat_interpolators[gdate], geos_data['EPV'][j, :], geos_data['PT'][j, :])
            geos_data['EL'] = eqlat

        # Put the profiles onto the ACE times and levels #
        geos_data_at_ace_times = _interp_geos_vars_to_ace_times(ace_dates, geos_dates, geos_data_on_std_times)
        geos_data_on_ace_levels = dict()
        for varname, geos_data in geos_data_at_ace_times.items():
            ace_profiles = np.full_like(ace_theta, np.nan)
            for i, _ in enumerate(geos_data):
                # Limit to GEOS levels where potential temperature is monotonically increasing. 
                decr_theta = np.diff(geos_data_at_ace_times['PT'][i,:]) <= 0
                if np.any(decr_theta):
                    pt_incr = np.flatnonzero(decr_theta)[-1] + 1
                    logger.debug('Limiting GEOS data to level {} and up to ensure monotonically increasing potential temperature'.format(pt_incr+1))
                else:
                    pt_incr = 0
                    logger.debug('GEOS potential temperature monotonically increasing for all levels')
                # Check that the GEOS data still covers the range of ACE potential temperatures. It should be okay if it doesn't
                # (ACE should just get NaNs there) this is just to see how often it happens
                min_geos_pt = np.nanmin(geos_data_at_ace_times['PT'][i, pt_incr:])
                min_ace_pt = np.nanmin(ace_theta[i, :])
                if min_geos_pt > min_ace_pt:
                    logger.info('Note that GEOS PT starts increasing monotonically above the bottom ACE level (profile time: {date}, ACE bottom: {ace:.2f}, GEOS bottom: {geos:.2f})'
                                .format(date=ace_dates[i], ace=min_ace_pt, geos=min_geos_pt))
                
                # This is one case where I think we do not want to extrapolate because the ACE data spans much greater
                # vertical extent than the GEOS data, so it can't be reliable
                ace_profiles[i, :] = np.interp(ace_theta[i, :], geos_data_at_ace_times['PT'][i, pt_incr:], geos_data[i, pt_incr:],
                                               left=np.nan, right=np.nan)
            geos_data_on_ace_levels[varname] = ace_profiles

        # Use the equivalent latitude and potential temperature profiles to look up CLAMS ages #
        ace_age = np.full_like(ace_theta, np.nan)
        for i in range(ace_lon.size):
            if use_geos_theta_for_age:
                theta_prof = geos_data_on_ace_levels['PT'][i, :]
            else:
                theta_prof = ace_theta[i, :]

            eqlat_prof = geos_data_on_ace_levels['EL'][i, :]
            ace_doy = mod_utils.clams_day_of_year(ace_dates[i])

            ace_age[i, :] = tccon_priors.get_clams_age(theta_prof, eqlat_prof, ace_doy)

        geos_data_on_ace_levels['age'] = ace_age
        return geos_data_on_ace_levels
    except Exception as err:
        msg = f'Problem occurred in ACE profile group for dates {date_str} (GEOS file date {geos_dates[0]}): {err}'
        raise err.__class__(msg)


def _bin_ace_to_geos_times(ace_dates):
    """
    Group ACE profiles by which GEOS FP files surround them

    :param ace_dates: an array of ACE profile datetimes
    :type ace_dates: array of datetimes

    :return: a dictionary where the keys will be Pandas timestamps representing the time of the GEOS file preceding the
     profiles in that group and the values are the indices of the profiles in that group as a numpy array.
    :rtype: dict
    """
    def to_3hr(d):
        return dt.datetime(d.year, d.month, d.day, to_3(d.hour))

    def to_3(v):
        return int(3*np.floor(v/3))

    first_time = ace_dates.min()
    last_time = ace_dates.max()

    first_time = to_3hr(first_time)
    last_time = to_3hr(last_time) + dt.timedelta(hours=3)

    # Create the bins as floats so we can use numpy.digitize
    ace_datenums = np.array([sat_utils.datetime2datenum(d) for d in ace_dates])
    bin_edges = pd.date_range(first_time, last_time, freq='3h')
    bin_edge_datenums = np.array([sat_utils.datetime2datenum(d) for d in bin_edges])

    inds = np.digitize(ace_datenums, bin_edge_datenums) - 1
    date_groups = OrderedDict()
    for i in np.unique(inds):
        date_groups[bin_edges[i]] = np.flatnonzero(inds == i)

    return date_groups


def _interp_geos_vars_to_ace_lat_lon(geos_file_path, geos_vars, ace_lon, ace_lat):
    def interp_prof_helper(data, glat, glon, alat, alon, interp_inds):
        profs_out = np.full([alat.size, data.shape[0]], np.nan)
        for i, data_layer in enumerate(data):
            # Reshape to nprofs-by-nlevels to be consistent with ACE
            profs_out[:, i] = mm.lat_lon_interp(data_layer, glat, glon, alat, alon, interp_inds)

        return profs_out

    geos_data = dict()
    geos_interp_inds = []
    with ncdf.Dataset(geos_file_path) as nh:
        geos_lon_half_res = 0.5 * float(nh.LongitudeResolution)
        geos_lat_half_res = 0.5 * float(nh.LatitudeResolution)

        for varname, scale in geos_vars.items():
            # convert to 3D - remove singleton time dimension
            geos_data[varname] = nh.variables[varname][0].filled(np.nan) * scale

        for lon, lat in zip(ace_lon, ace_lat):
            geos_interp_inds.append(mm.querry_indices(nh, lat, lon, geos_lat_half_res, geos_lon_half_res))

        geos_lat = nh.variables['lat'][:].filled(np.nan)
        geos_lon = nh.variables['lon'][:].filled(np.nan)

        for varname, array in geos_data.items():
            geos_data[varname] = interp_prof_helper(array, geos_lat, geos_lon, ace_lat, ace_lon, geos_interp_inds)

    return geos_data


def _interp_geos_vars_to_ace_times(ace_datetimes, geos_datetimes, geos_vars):
    interped_vars = dict()
    for ace_dt in ace_datetimes:
        weight = sat_utils.time_weight_from_datetime(ace_dt, geos_datetimes[0], geos_datetimes[1])
        for k in geos_vars[0].keys():
            interped_vars[k] = weight * geos_vars[0][k] + (1 - weight) * geos_vars[1][k]

    return interped_vars


def save_ace_age_file(ace_out_file, vars_to_copy, orig_ace_file, eqlat, age):
    def get_var_attrs(nh, varname):
        return {k: nh.variables[varname].getncattr(k) for k in nh.variables[varname].ncattrs()}

    with ncdf.Dataset(orig_ace_file, 'r') as nhread, ncdf.Dataset(ace_out_file, 'w') as nhwrite:
        # Copy all the dimensions
        for dimname in nhread.dimensions.keys():
            try:
                attributes = get_var_attrs(nhread, dimname)
                ioutils.make_ncdim_helper(nhwrite, dimname, nhread.variables[dimname][:], **attributes)
            except KeyError:
                # Sometimes we get a dimension with no corresponding variable
                nhwrite.createDimension(dimname, nhread.dimensions[dimname].size)

        # Copy all the requested variables
        for varname in vars_to_copy:
            attributes = get_var_attrs(nhread, varname)
            dims = nhread.variables[varname].dimensions
            ioutils.make_ncvar_helper(nhwrite, varname, nhread.variables[varname][:], dims, **attributes)

        # Add the new variables
        dims = ('index', 'altitude')
        ioutils.make_ncvar_helper(nhwrite, 'eqlat', eqlat, dims, long_name='equivalent_latitude', units='degrees_north',
                                  description='Equivalent latitude computed from potential vorticity')
        ioutils.make_ncvar_helper(nhwrite, 'age', age, dims, long_name='clams_age', units='years',
                                  description='Time in years since the air entered the stratosphere')

        ioutils.add_creation_info(nhwrite, creation_note='ace_fts_analysis.generate_ace_age_file')
        nhwrite.setncattr('template_ace_file', orig_ace_file)
        ioutils.add_dependent_file_hash(nhwrite, 'template_ace_sha1', orig_ace_file)


def strat_co(ch4, T, P, age, co_0, oh=2.5e5, gamma=3):
    R = 8.314 * 100 ** 3 / 100 / 6.626e23

    def ndens_air(T, P):
        return P / (R * T)

    def kCO_OH(nair):
        return 1.57e-13 + 3.54e-33 * nair

    def kCH4_OH(T):
        return 2.8e-14 * T ** 0.667 * np.exp(-1575 / T)

    ndens = ndens_air(T, P)
    age = age * 365.25 * 24 * 3600
    ss = gamma * kCH4_OH(T) * ch4 / kCO_OH(ndens)
    bc = co_0 * np.exp(-kCO_OH(ndens) * oh * age)
    return ss + bc


def make_cmam_excess_co_lut(save_file, cmam_file):

    # First we need to rearrange the CMAM data so that year and month are organized separately.
    with xr.open_dataset(cmam_file) as ds:
        month_slices = [ds['vmrco'][i::12] for i in range(12)]
        for i, month in enumerate(month_slices):
            doy = mod_utils.day_of_year(pd.Timestamp(2005, i+1, 15))
            mdim = OrderedDict([('doy', [doy])])
            month = month.expand_dims(mdim, axis=1)
            month.coords['time'] = [t.item().year for t in month.coords['time']]
            month_slices[i] = month

        # Duplicate the first and last months on the opposite ends to make periodic interpolation possible
        first_month = month_slices[0].copy()
        first_month.coords['doy'] = first_month.coords['doy'] + mod_utils.days_per_year

        last_month = month_slices[-1].copy()
        last_month.coords['doy'] = last_month.coords['doy'] - mod_utils.days_per_year

        month_slices.append(first_month)
        month_slices.insert(0, last_month)

        co_monthly = xr.concat(month_slices, dim='doy')

    # Limit it to after 2000 - there is a trend in CO but it looks like it more or less levels off after 2000.
    co_attrs = co_monthly.attrs
    co_monthly = co_monthly.sel(time=slice(2000, None))

    # Then we just need to average in time and longitude to get monthly pressure-latitude slices.
    co_monthly = co_monthly.mean(dim='time').mean(dim='lon')
    co_monthly.attrs = co_attrs.copy()
    # will convert to ppb later, more convenient to leave in mole fraction for now because we need to calculate number
    # density too.

    # Next we need to calculate the number density. Since we only have pressure from the model, we'll get temperature
    # from the US standard atmosphere.
    co_monthly.coords['plev'] /= 100  # convert Pa to hPa
    co_monthly.coords['plev'].attrs['units'] = 'hPa'

    p = co_monthly.coords['plev']
    t, z = mod_utils.get_ussa_for_pres(p)
    t = xr.DataArray(t, coords=p.coords)
    z = xr.DataArray(z, coords=p.coords)

    t.attrs['units'] = 'K'
    t.attrs['standard_name'] = 'absolute_temperature'
    t.attrs['long_name'] = 'temperature'

    z.attrs['units'] = 'km'
    z.attrs['standard_name'] = 'altitude_above_sea_level'
    z.attrs['long_name'] = 'altitude'

    nair = mod_utils.number_density_air(p, t)
    nair.attrs['units'] = 'molec. cm^{-3}'
    nair.attrs['standard_name'] = 'dry_number_density'
    nair.attrs['long_name'] = 'number density of air'

    co_monthly_nd = co_monthly * nair
    co_monthly_nd.attrs['units'] = 'molec. cm^{-3}'
    co_monthly_nd.attrs['standard_name'] = 'co_number_density'
    co_monthly_nd.attrs['long_name'] = 'CO number density'

    co_monthly *= 1e9
    co_monthly.attrs['units'] = 'ppb'

    full_ds = xr.Dataset({'co': co_monthly, 'co_nd': co_monthly_nd, 'temperature': t, 'altitude': z, 'nair': nair})
    full_ds.attrs['cmam_file'] = cmam_file if not os.path.islink(cmam_file) else os.readlink(cmam_file)
    full_ds.attrs['cmam_sha1'] = ioutils.make_dependent_file_hash(cmam_file)
    full_ds.attrs['history'] = ioutils.make_creation_info('ace_fts_analysis.make_cmam_excess_co_lut')
    full_ds.to_netcdf(save_file)


def make_excess_co_lut(save_file, ace_co_file, ace_ch4_file, ace_age_file, lat_type='eq', min_req_pts=10, smoothing_window_width=5):
    R = 8.314 * 100 ** 3 / 100 / 6.626e23
    flag_bits = {'clip': 2 ** 0, 'toofew': 2 ** 1, 'filled': 2 ** 2, 'ussa_any': 2**3, 'ussa_half': 2**4,
                 'lowval': 2**5}
    flag_units = 'Bit meanings (least significant first): 1 = value <0 clipped, 2 = fewer than {} points, ' \
                 '3 = NaN filled, 4 = At least one point used USSA P/T, 5 = >50% points used USSA P/T, ' \
                 '6 = value below rolling mean - 1 sigma filled'.format(min_req_pts)

    def ndens_air(T, P):
        return P / (R * T)

    def kCO_OH(nair):
        return 1.57e-13 + 3.54e-33 * nair

    def kCH4_OH(T):
        return 2.8e-14 * T ** 0.667 * np.exp(-1575 / T)

    def strat_co(ch4, T, P, age, co_0, oh=2.5e5, gamma=3):
        ndens = ndens_air(T, P)
        age = age * 365.25 * 24 * 3600
        ss = gamma * kCH4_OH(T) * ch4 / kCO_OH(ndens)
        bc = co_0 * np.exp(-kCO_OH(ndens) * oh * age)
        return ss + bc

    def fill_nans(ds, flags, primary_keys, dim_interp_order):
        # We want to interpolate/extrapolate along theta first. I tried originally doing periodic interpolation along
        # the day-of-year axis, but there's not enough data in the northen hemisphere polar summer to get the decrease
        # right. However, there are some columns that have 0 or 1 non-NaN values which the interpolator then chokes
        # on, so we need to find those columns and replace the NaNs with a fill value temporarily to avoid trying
        # to handle them here.
        was_nans = np.zeros(ds[primary_keys[0]].shape, dtype=np.bool_)
        for k in primary_keys:
            was_nans |= np.isnan(ds[k].data)

        all_nans = dict()
        for dim, do_extrap in dim_interp_order.items():
            logger.debug('Filling NaNs along'.format(dim))

            fill_val = 'extrapolate' if do_extrap else np.nan

            for key, variable in ds.items():
                var_not_nans = ~np.isnan(variable.data)

                # DataArray's sum function doesn't work with keepdims in v0.12.1. This is fixed in v0.12.2 which doesn't
                # seem to yet be available via conda
                dim_ind = variable.dims.index(dim)
                this_most_nans = var_not_nans.sum(axis=dim_ind, keepdims=True) < 2
                this_most_nans = np.broadcast_to(this_most_nans, variable.shape)
                # only replace NaNs in lines with 0 or 1 non-NaN values. Leave non-NaN values in those columns alone,
                # and leave NaNs elsewhere alone.
                this_most_nans = this_most_nans & ~var_not_nans
                variable.data[this_most_nans] = -999
                all_nans[key] = this_most_nans

            ds = ds.interpolate_na(dim=dim, method='linear', fill_value=fill_val)

            for key, this_most_nans in all_nans.items():
                ds[key].data[this_most_nans] = np.nan

        nans_filled = np.zeros_like(was_nans)
        for k in primary_keys:
            nans_filled |= (was_nans & ~np.isnan(ds[k].data))

        flags[nans_filled] = np.bitwise_or(flags[nans_filled], flag_bits['filled'])

        return ds, flags

    def bin_co(df_dict, all_bins, all_coords, bin_names, primary_key):
        bin_cols = tuple(df_dict.keys())
        coord_cols = []
        total_bins = 1
        for i, (coord, bin) in enumerate(zip(all_coords, all_bins)):
            total_bins *= (bin.size - 1)
            inds = np.digitize(coord.flatten(), bin) - 1
            colname = 'ind{}'.format(i)
            df_dict[colname] = inds
            coord_cols.append(colname)

        df = pd.DataFrame(df_dict)
        all_bin_centers = [0.5 * (b[:-1] + b[1:]) for b in all_bins]

        proto_array = np.full([np.size(b) - 1 for b in all_bins],
                              np.nan if np.issubdtype(excess_co.dtype, np.floating) else 0)
        proto_array = xr.DataArray(proto_array, coords=all_bin_centers, dims=bin_names)
        means = xr.Dataset({k: proto_array.copy() for k in bin_cols})
        stds = xr.Dataset({k: proto_array.copy() for k in bin_cols})

        flags = np.zeros(proto_array.shape, dtype=int)
        counts = np.zeros(proto_array.shape, dtype=int)

        pbar = mod_utils.ProgressBar(total_bins, prefix='Binning', style='counter')
        iind = 0
        for inds, grp in df.groupby(coord_cols):
            pbar.print_bar(iind)
            iind += 1

            if any(i < 0 for i in inds):
                # negative index indicates that the value is below the minimum bin so we skip it
                continue
            try:
                subdata = grp[primary_key]
                counts[inds] = (~subdata.isna()).sum()
                if subdata.size < min_req_pts:
                    flags[inds] = np.bitwise_or(flags[inds], flag_bits['toofew'])
                    continue

                for k in bin_cols:
                    subdata = grp[k]
                    means[k][inds] = np.nanmean(subdata)
                    stds[k][inds] = np.nanstd(subdata)

            except IndexError:
                # inds will be too large if any of the values is outside the right edge of the last bin, so just skip
                # those since we don't want to bin them. We caught any outside the left edge of the bin checking if any
                # are < 0
                continue

        pbar.finish()

        return means, stds, flags, counts, all_bin_centers

    def replace_low_values(darray, dim, flags, remove_neg=True, extra_mask=None):
        darray = darray.copy()
        periodic = dim == 'doy'
        # Concatenate extra slices to allow periodic interpolation if operating along DOY
        if periodic:
            ex_width = int(np.ceil(smoothing_window_width/2))
            dim_max = darray.coords[dim].max()
            darray_start = darray.isel(**{dim: slice(None, ex_width)})
            darray_start.coords[dim] = darray_start.coords[dim] + dim_max
            darray_end = darray.isel(**{dim: slice(-ex_width, None)})
            darray_end.coords[dim] = darray_end.coords[dim] - dim_max
            darray = xr.concat([darray_end, darray, darray_start], dim=dim)

        if remove_neg:
            darray.data[darray.data < 0] = np.nan
        kwargs = {dim: smoothing_window_width, 'center': True}
        mean = darray.rolling(**kwargs).mean()
        std = darray.rolling(**kwargs).std()
        xx = darray < (mean - std)
        if extra_mask is not None:
            xx = xx | extra_mask
        darray.data[xx.data] = np.nan

        darray = darray.interpolate_na(dim=dim)
        if periodic:
            s = slice(ex_width, -ex_width)
            darray = darray.isel(**{dim: s})
            xx = xx.isel(**{dim: s})
        flags[xx.data] = np.bitwise_or(flags[xx.data], flag_bits['lowval'])
        return darray, flags

    #################
    # Load ACE data #
    #################

    with ncdf.Dataset(ace_age_file) as ds:
        ace_dates = read_ace_date(ds)
        ace_alt = read_ace_var(ds, 'altitude', None)
        qflags = read_ace_var(ds, 'quality_flag', None)

        if lat_type == 'geog':
            ace_lat = read_ace_var(ds, 'latitude', None)
            ace_lat = np.broadcast_to(ace_lat.reshape(-1, 1), qflags.shape)
        elif lat_type == 'eq':
            ace_lat = read_ace_var(ds, 'eqlat', None)
            # Equivalent latitude may not be defined on all levels. Fill in any internal levels with linear
            # interpolation then extends the profile assuming that the equivalent latitude is constant above the top
            # level.
            ace_lat = xr.DataArray(ace_lat, coords=[np.arange(ace_lat.shape[0]), ace_alt], dims=['index', 'alt'])
            ace_lat = ace_lat.interpolate_na(dim='alt', method='linear').interpolate_na(dim='alt', method='nearest', fill_value='extrapolate')
            ace_lat = ace_lat.data
        else:
            raise ValueError('lat must be "geog" or "eq"')

        ace_t = read_ace_var(ds, 'temperature', qflags)
        ace_p = read_ace_var(ds, 'pressure', qflags) * 1013.25  # atm -> hPa
        ace_pt = read_ace_theta(ds, qflags=qflags)
        ace_age = read_ace_var(ds, 'age', qflags)

    with ncdf.Dataset(ace_ch4_file) as ds:
        qflags = read_ace_var(ds, 'quality_flag', None)
        ace_ch4 = read_ace_var(ds, 'CH4', qflags)

    with ncdf.Dataset(ace_co_file) as ds:
        qflags = read_ace_var(ds, 'quality_flag', None)
        all_ace_co = read_ace_var(ds, 'CO', qflags)

    ace_doys = np.array([mod_utils.day_of_year(d) for d in ace_dates])
    ace_doys = np.broadcast_to(ace_doys.reshape(-1, 1), all_ace_co.shape)

    # Create versions of T and P filled in by the US standard atmosphere for the mesospheric CO profile table.
    xx_ussa = np.isnan(ace_t) | np.isnan(ace_p)
    ace_alt_full = np.broadcast_to(ace_alt.reshape(1, -1), ace_t.shape)
    ace_t_ussa = ace_t.copy()
    ace_p_ussa = ace_p.copy()
    ace_t_ussa[xx_ussa], ace_p_ussa[xx_ussa] = mod_utils.get_ussa_for_alts(ace_alt_full[xx_ussa])

    co_calc = strat_co(ace_ch4, ace_t, ace_p, ace_age, 50.0e-9)
    excess_co = (all_ace_co - co_calc)*1e9  # calculate excess and convert to ppb
    excess_notnans = ~np.isnan(excess_co)
    excess_co = excess_co[excess_notnans]

    air_nd = mod_utils.number_density_air(ace_p_ussa, ace_t_ussa)
    ace_co_nd = all_ace_co * air_nd
    not_all_nans = np.any(~np.isnan(ace_co_nd), axis=0)
    air_nd = air_nd[:, not_all_nans]
    ace_co_nd = ace_co_nd[:, not_all_nans]
    ace_p_ussa = ace_p_ussa[:, not_all_nans]
    ace_t_ussa = ace_t_ussa[:, not_all_nans]
    xx_ussa = xx_ussa[:, not_all_nans]
    ace_alt_full = ace_alt_full[:, not_all_nans]

    meso_notnans = ~np.isnan(ace_co_nd)
    ace_co_nd = ace_co_nd[meso_notnans]
    ace_p_ussa = ace_p_ussa[meso_notnans]
    ace_t_ussa = ace_t_ussa[meso_notnans]
    air_nd = air_nd[meso_notnans]
    xx_ussa = xx_ussa[meso_notnans]

    lat_bins = np.arange(-90, 91, 10)
    doy_bins = np.arange(0, 370, 5)
    doy_bins[-1] = 366  # account for leap years
    theta_bins = np.arange(350, 4150, 100)
    alt_bins = np.arange(ace_alt[not_all_nans].min(), ace_alt[not_all_nans].max()+5, 5.)

    excess_bins = [lat_bins, doy_bins, theta_bins]
    excess_coords = [ace_lat[excess_notnans], ace_doys[excess_notnans], ace_pt[excess_notnans]]
    excess_bin_names = ['lat', 'doy', 'theta']
    excess_bin_units = ['degrees_north', 'day of year (0-based)', 'K']

    meso_bins = [lat_bins, doy_bins, alt_bins]
    meso_coords = [ace_lat[:, not_all_nans][meso_notnans], ace_doys[:, not_all_nans][meso_notnans], ace_alt_full[meso_notnans]]
    meso_bin_names = ['lat', 'doy', 'altitude']
    meso_bin_units = ['degrees_north', 'doy of year (0-based)', 'km']

    ####################################################
    # Use pandas groupby() to bin the data efficiently #
    ####################################################

    excess_means, excess_stds, excess_flags, excess_counts, excess_bin_centers = bin_co(
        {'excess_co': excess_co}, all_bins=excess_bins, all_coords=excess_coords, bin_names=excess_bin_names,
        primary_key='excess_co'
    )
    excess_means['excess_co'].attrs['unit'] = 'ppb'
    excess_means['excess_co'].attrs['description'] = 'Excess CO in ACE-FTS data beyond that predicted by a kinetic ' \
                                                     'model of the CO chemistry in the stratosphere.'
    excess_stds['excess_co'].attrs['unit'] = 'ppb'
    excess_stds['excess_co'].attrs['description'] = 'Standard deviation of the excess CO in this bin.'

    meso_means, meso_stds, meso_flags, meso_counts, meso_bin_centers = bin_co(
        {'co_nd': ace_co_nd, 'air_nd': air_nd, 'pressure': ace_p_ussa, 'temperature': ace_t_ussa, 'is_ussa': xx_ussa},
        all_bins=meso_bins, all_coords=meso_coords, bin_names=meso_bin_names, primary_key='co_nd'
    )

    inds = meso_means['is_ussa'].data > 0
    meso_flags[inds] = np.bitwise_or(meso_flags[inds], flag_bits['ussa_any'])
    inds = meso_means['is_ussa'].data > 0.5
    meso_flags[inds] = np.bitwise_or(meso_flags[inds], flag_bits['ussa_half'])

    meso_means = meso_means.drop('is_ussa')
    meso_stds = meso_stds.drop('is_ussa')

    meso_means['co_nd'].attrs['unit'] = 'molec cm^{-3}'
    meso_means['co_nd'].attrs['description'] = 'ACE CO in number density. ACE T and P extended using the US standard ' \
                                               'atmosphere as needed.'
    meso_means['air_nd'].attrs['unit'] = 'molec cm^{-3}'
    meso_means['air_nd'].attrs['description'] = 'Number density of air calculated using the ideal gas law with ACE T ' \
                                                'and P extended using the US standard atmosphere as needed'
    meso_means['pressure'].attrs['unit'] = 'hPa'
    meso_means['pressure'].attrs['description'] = 'ACE pressure extended using the US standard atmosphere as needed'
    meso_means['temperature'].attrs['unit'] = 'K'
    meso_means['temperature'].attrs['description'] = 'ACE temperatures extended using the US standard atmosphere as needed'

    meso_stds['co_nd'].attrs['unit'] = 'molec cm^{-3}'
    meso_stds['co_nd'].attrs['description'] = 'Standard deviation of ACE CO in number density in the bins.'
    meso_stds['air_nd'].attrs['unit'] = 'molec cm^{-3}'
    meso_stds['air_nd'].attrs['description'] = 'Standard deviation of number density of air in the bins'
    meso_stds['pressure'].attrs['unit'] = 'hPa'
    meso_stds['pressure'].attrs['description'] = 'Standard deviation of ACE pressure in the bins.'
    meso_stds['temperature'].attrs['unit'] = 'K'
    meso_stds['temperature'].attrs['description'] = 'Standard deviation of ACE temperatures in the bins'

    all_bin_edges = excess_bins + meso_bins[-1:]
    all_bin_centers = excess_bin_centers + meso_bin_centers[-1:]
    all_bin_names = excess_bin_names + meso_bin_names[-1:]
    all_bin_units = excess_bin_units + meso_bin_units[-1:]

    #######################################################################################
    # Fill in NaNs and clip to positive values. Track these actions in the flags variable #
    #######################################################################################
    inds = excess_means['excess_co'].data < 0
    excess_flags[inds] = np.bitwise_or(excess_flags[inds], flag_bits['clip'])
    excess_means['excess_co'] = excess_means['excess_co'].clip(0)

    excess_interp_order = OrderedDict([('theta', True), ('lat', True), ('doy', False)])
    excess_means, excess_flags = fill_nans(excess_means, excess_flags, ['excess_co'], excess_interp_order)
    meso_interp_order = OrderedDict([('doy', False), ('lat', True), ('altitude', True)])
    meso_means, meso_flags = fill_nans(meso_means, meso_flags, ['co_nd', 'air_nd', 'pressure', 'temperature'],
                                       meso_interp_order)

    excess_means['excess_co'], excess_flags = replace_low_values(excess_means['excess_co'], 'doy', excess_flags)
    meso_means['co_nd'], meso_flags = replace_low_values(meso_means['co_nd'], 'doy', meso_flags)

    # Add the count variables
    excess_means['excess_co_counts'] = xr.DataArray(excess_counts, coords=excess_means['excess_co'].coords)
    excess_means['excess_co_counts'].attrs['unit'] = '#'
    excess_means['excess_co_counts'].attrs['description'] = 'Number of ACE points contributing to the bin'
    meso_means['co_nd_counts'] = xr.DataArray(meso_counts, coords=meso_means['co_nd'].coords)
    meso_means['co_nd_counts'].attrs['unit'] = '#'
    meso_means['co_nd_counts'].attrs['description'] = 'Number of ACE points contributing to the bin'

    _save_excess_co_lut(save_file=save_file,
                        theta_binned_means=excess_means, theta_binned_stds=excess_stds, theta_binned_flags=excess_flags,
                        alt_binned_means=meso_means, alt_binned_stds=meso_stds, alt_binned_flags=meso_flags,
                        bin_edges=all_bin_edges, bin_centers=all_bin_centers, bin_names=all_bin_names,
                        bin_units=all_bin_units, theta_binned_dims=excess_bin_names, alt_binned_dims=meso_bin_names,
                        flag_description=flag_units, lat_type=lat_type,
                        ace_co_file=ace_co_file, ace_ch4_file=ace_ch4_file, ace_age_file=ace_age_file)


def _save_excess_co_lut(save_file, theta_binned_means, theta_binned_stds, theta_binned_flags,
                        alt_binned_means, alt_binned_stds, alt_binned_flags,
                        bin_edges, bin_centers, bin_names, bin_units,
                        flag_description, theta_binned_dims, alt_binned_dims, lat_type,
                        ace_co_file, ace_ch4_file, ace_age_file):

    with ncdf.Dataset(save_file, 'w') as nh:

        dims = []
        for name, centers, edges, units in zip(bin_names, bin_centers, bin_edges, bin_units):
            this_dim = ioutils.make_ncdim_helper(nh, name, centers, unit=units)
            dims.append(this_dim)
            ioutils.make_ncdim_helper(nh, name + '_bin_edges', edges, unit=units)

        for k in theta_binned_means.keys():
            data = theta_binned_means[k]
            ioutils.make_ncvar_helper(nh, k, data.data, dims=theta_binned_dims,
                                      flag_field='theta_binned_flags', **data.attrs)
            if k in theta_binned_stds:
                std = theta_binned_stds[k]
                ioutils.make_ncvar_helper(nh, k+'_std', std.data, dims=theta_binned_dims,
                                          flag_field='theta_binned_flags', **std.attrs)
        ioutils.make_ncvar_helper(nh, 'theta_binned_flags', theta_binned_flags, dims=theta_binned_dims,
                                  unit='Bit array flag', description=flag_description)

        for k in alt_binned_means.keys():
            data = alt_binned_means[k]
            ioutils.make_ncvar_helper(nh, k, data.data, dims=alt_binned_dims,
                                      flag_field='alt_binned_flags', **data.attrs)
            if k in alt_binned_stds:
                std = alt_binned_stds[k]
                ioutils.make_ncvar_helper(nh, k+'_std', std.data, dims=alt_binned_dims,
                                          flag_field='alt_binned_flags', **std.attrs)
        ioutils.make_ncvar_helper(nh, 'alt_binned_flags', alt_binned_flags, dims=alt_binned_dims,
                                  unit='Bit array flag', description=flag_description)

        ioutils.add_dependent_file_hash(nh, 'ace_co_file', ace_co_file)
        ioutils.add_dependent_file_hash(nh, 'ace_ch4_file', ace_ch4_file)
        ioutils.add_dependent_file_hash(nh, 'ace_age_file', ace_age_file)
        ioutils.add_creation_info(nh, 'ace_fts_analysis.create_excess_co_lut')
        nh.setncattr('latitude_type', 'Binned by {} latitude'.format(lat_type))
