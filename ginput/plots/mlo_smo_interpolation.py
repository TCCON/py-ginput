from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
import pandas as pd
import re

from .plot_utils import ColorMapper


def plot_interp_cutoff_rmse(rmse_df: pd.DataFrame, gas_label: str, units: str, relative: bool = False, ax=None) -> pd.DataFrame:
    """Plot the RMSE of interpolated MLO or SMO timeseries versus truth.

    Parameters
    ----------
    rmse_df
        A dataframe returned by :func:`make_rmse_df`.

    gas_label
        The name of the gas being plotted, can use Latex-style strings to subscript numbers.

    units
        The units of the deltas, e.g. "ppm" or "ppb"

    relative
        Set to ``True`` to plot the RMSE as multiples of the smallest RMSE for that site.
        The text listing the smallest RMSE will always be in the units of the deltas.

    ax
        Axes to plot into.

    Returns
    -------
    rmses
        A dataframe with the RMSE values; the cutoff is the index and the columns are the site
        keys.
    """
    ax = ax or plt.gca()
    min_text = []
    for site, site_rmse in rmse_df.items():
        min_cutoff = site_rmse.argmin()
        min_rmse = site_rmse.min()
        min_text.append(f'{site.upper()} min at {min_cutoff} = {min_rmse:.2f} {units}')
        if relative:
            site_rmse = site_rmse / min_rmse
        ax.plot(site_rmse, label=site.upper(), marker='o')

    min_text = '\n'.join(min_text)
    ax.text(0.02, 0.98, min_text, va='top', transform=ax.transAxes)
    ax.set_xlabel('max_months_simple_interp')
    if relative:
        ax.set_ylabel(f'{gas_label} RMSE relative to min')    
    else:
        ax.set_ylabel(f'{gas_label} RMSE ({units})')
    ax.grid(True)


def make_rmse_df(deltas: dict) -> pd.DataFrame:
    """Make a dataframe of the RMSE by site and linear interpolation cutoff.

    Parameters
    ----------
    deltas
        A dictionary of dictionaries. The top keys are the sites, and the second level
        keys are the max number of months to use linear interpolation. The values of the
        inner dictionary are dataframes with the differences between the true and interpolated
        timeseries. The column names must be "YYYYMM_GGgap" where the YYYYMM is the year
        and month the gap in the test data starts, and GG is the length of the gap in months.
        The index must be the date of the data.

    Returns
    -------
    rmse_df
        The dataframe with RMSE values. The columns will be the site keys and the index the number
        of months linear interpolation cutoff.
    """
    all_rmses = []
    for site, site_deltas in deltas.items():
        x = []
        y = []
        for cutoff, cutoff_df in site_deltas.items():
            x.append(cutoff)
            y.append(_compute_interp_cutoff_rmse(cutoff_df))
        all_rmses.append(pd.Series(y, index=x, name=site))

    return pd.concat(all_rmses, axis='columns')


def _compute_interp_cutoff_rmse(delta_df):
    se = 0.0
    n = 0.0
    for colname, colvals in delta_df.items():
        n_gap = _get_colname_gap(colname)
        se += (colvals**2).sum()
        n += n_gap
    return np.sqrt(se / n)


def rmse_df_to_nc(rmse_df: pd.DataFrame, gas: str, units: str, nc_file: os.PathLike):
    """Write an RMSE dataframe from :func:`make_rmse_df` to a netCDF file.

    Parameters
    ----------
    rmse_df
        The datafrom from :func:`make_rmse_df`.

    gas
        The name of the gas the RMSE values are for (used in attributes).

    units
        The units of the RMSE values (e.g., "ppm" or "ppb")

    nc_file
        Path to the netCDF file to write, will be overwritten if it exists.
    """
    with Dataset(nc_file, 'w') as ds:
        dim_name = 'cutoff'
        ds.createDimension(dim_name, rmse_df.shape[0])
        var = ds.createVariable('cutoff', 'i4', (dim_name,))
        var.description = 'Maximum number of months long a gap may be to be filled with linear interpolation'
        var.units = 'months'
        var[:] = rmse_df.index.to_numpy()

        for colname, colvals in rmse_df.items():
            var = ds.createVariable(colname, 'f8', (dim_name,))
            var.description = f'{gas} RMSE between interpolated and true timeseries for site {colname.upper()}'
            var.units = units
            var[:] = colvals.to_numpy()


def rmse_df_from_nc(nc_file: os.PathLike) -> pd.DataFrame:
    """Load a dataframe that :func:`make_rmse_df` would create from a netCDF file.
    """
    with Dataset(nc_file) as ds:
        index = None
        data = dict()
        for varname, var in ds.variables.items():
            values = var[:]
            if varname == 'cutoff':
                index = values
            else:
                data[varname] = values
        return pd.DataFrame(data, index=index)


def _get_colname_gap(colname):
    return int(re.search(r'_(\d+)gap', colname).group(1))


def plot_delta_vs_month(data: dict, site: str, gas_label: str, units: str):
    """Plot a grid of the difference between the true and interpolated MLO or SMO data.

    The top row will be the timeseries with the true DMFs and the mean and range of the
    interpolated DMFs for different cutoffs between linear and trend+seasonal cycle interpolation.
    The remaining rows will show a grid where each panel is the difference between truth and
    interpolation (for months with non-zero difference) for a particular gap width and cutoff
    length. The differences are colored by starting year.

    Parameters
    ----------
    data
        A three level dictionary. The top must have keys "deltas", "base", and "test". The second level
        must be the site key, usually "mlo" or "smo". For "base", the second level dictionary must have
        the true timeseries as date-indexed :class:`pandas.Series`. "deltas" and "test" have the cutoff
        lengths as third level keys, and each third level value is a dataframe with column names of the
        form "YYYYMM_GGgap" with "YYYYMM" specifying the start year & month of the gap in the test data
        and "GG" the length of the gap in months. The index must be the data dates. For "deltas", the 
        dataframe values must be the difference between the interpolated and true timeseries, while "test"
        has the interpolated timeseries.

    site
        The site key in the second level of ``data``.

    gas_label
        The name of the gas being plotted, can use Latex-style strings to subscript numbers.

    units
        The units of the DMFs and differences, e.g. "ppm" or "ppb".
    """
    site_deltas = data['deltas'][site]

    # Assume that the years are the same for each cutoff, since we're testing on the same data
    cutoffs = sorted(site_deltas.keys())
    c0 = cutoffs[0]
    has_delta = (site_deltas[c0] != 0).any(axis='columns')
    min_year = has_delta.index[has_delta].year.min()
    max_year = has_delta.index[has_delta].year.max()
    year_cmapper = ColorMapper(vmin=min_year, vmax=max_year, cmap='rainbow')

    # Also assume that the gaps are the same
    unique_gaps = np.unique([_get_colname_gap(c) for c in site_deltas[c0].columns])

    # We'll handle standardizing the y-limits manually to avoid issues with the colorbars
    # sharing y-axes with the data plots
    ymin = min(df.to_numpy().min() for df in site_deltas.values()) - 0.5
    ymax = max(df.to_numpy().max() for df in site_deltas.values()) + 0.5

    nx = unique_gaps.size
    ny = len(cutoffs) + 1
    wr = [20]*nx + [1]
    fig = plt.figure(figsize=(4*nx + 4/20, 4*ny))
    gs = GridSpec(nrows=ny, ncols=nx+1, width_ratios=wr, wspace=0.4, hspace=0.3)

    ts_ax = fig.add_subplot(gs[0, :-1])
    ts_cax = fig.add_subplot(gs[0, -1])
    plot_site_timeseries(data['base'][site], data['test'][site], gas_label, units, ax=ts_ax, cax=ts_cax)

    first_row = True
    for icut, cut in enumerate(cutoffs):
        delta_df = site_deltas[cut]
        df_gaps = np.asarray([_get_colname_gap(c) for c in site_deltas[c0].columns])
        for igap, gap in enumerate(unique_gaps):
            gap_df = delta_df.loc[:, df_gaps == gap]
            ax = fig.add_subplot(gs[icut+1, igap])
            _plot_delta_vs_month_one_cutoff_one_gap(gap_df, year_cmapper, ax=ax)
            ax.set_ylim(ymin, ymax)
            if first_row:
                ax.set_title(f'Gap = {gap} months')
            if igap == 0:
                ax.set_ylabel(f'$\\Delta$ {gas_label} ({units}), cutoff = {cut}')

        cax = fig.add_subplot(gs[icut+1, -1])
        plt.colorbar(year_cmapper, cax=cax, label='Starting year')
        first_row = False

def _plot_delta_vs_month_one_cutoff_one_gap(delta_df, year_cmapper, ax=None):
    ax = ax or plt.gca()
    all_months = set()
    for _, col in delta_df.items():
        col = col[col != 0]
        start_year = col.index.year.min()
        months = _wrap_months(col.index.month)
        all_months.update(months)
        ax.plot(months, col.to_numpy(), color=year_cmapper(start_year), marker='.', markersize=3)

    def xtick_fmt(x, pos):
        m = ((int(x) - 1) % 12) + 1
        d = pd.Timestamp(2000, m , 1)
        return d.strftime('%b')
    ax.set_xticks(sorted(all_months))
    ax.xaxis.set_major_formatter(xtick_fmt)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.grid(True)

def _wrap_months(months):
    ii = np.flatnonzero(np.diff(months) < 0)
    if ii.size == 0:
        return months.to_numpy()
    else:
        months = months.to_numpy()
        for i in ii:
            months[i+1:] += 12
        return months


def plot_site_timeseries(site_base_dmfs, site_test_dmfs, gas_label, units, ax=None, cax=None):
    if ax is None or cax is None:
        _, axs = plt.subplots(1, 2, figsize=(16,4), gridspec_kw={'width_ratios': [20,1]})
        ax = ax or axs[0]
        cax = cax or axs[1]

    ax.plot(site_base_dmfs, color='black', lw=2, marker='.', label='Truth')
    cutoffs = sorted(site_test_dmfs.keys())
    cmapper = ColorMapper.from_discrete_values(cutoffs, cmap='coolwarm')

    for cutoff, cutoff_df in site_test_dmfs.items():
        color = cmapper(cutoff)
        _plot_one_cutoff_df_mean_and_range(ax, cutoff_df, color)

    plt.colorbar(cmapper, cax=cax, label='Cutoff (months)', ticks=cutoffs)
    ax.grid(True)
    ax.set_ylabel(f'{gas_label} ({units})')


def _plot_one_cutoff_df_mean_and_range(ax, cutoff_df, color):
    mean_dmf = cutoff_df.mean(axis='columns')
    min_dmf = cutoff_df.min(axis='columns')
    max_dmf = cutoff_df.max(axis='columns')

    ax.plot(mean_dmf, color=color, ls='--', lw=0.75)
    ax.plot(min_dmf, color=color, ls='-.', lw=0.75)
    ax.plot(max_dmf, color=color, ls='-.', lw=0.75)
