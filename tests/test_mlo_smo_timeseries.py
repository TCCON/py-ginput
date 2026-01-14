from ginput.priors import tccon_priors
from ginput.common_utils.ggg_logging import setup_logger
from ginput.common_utils.readers import read_priors_conc_from_netcdf
from ginput.common_utils.test_utils import compare_dataframes
import ginput.plots.mlo_smo_interpolation as msiplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import pytest


def test_mlo_smo_default_filling(
    subtests,
    noaa_gas,
    mlo_smo_default_expected_dir,
    mlo_smo_default_out_dir,
    mlo_smo_default_end_date,
):
    rec_class = tccon_priors.gas_records[noaa_gas]
    rec = rec_class(
        last_date=mlo_smo_default_end_date,
        save_strat=False,
        recalculate_strat_lut=False,
        recalc_if_custom_dates=False,
    )
    seas_out_file = mlo_smo_default_out_dir / f'{noaa_gas}_seasonal.nc'
    trend_out_file = mlo_smo_default_out_dir / f'{noaa_gas}_trend.nc'
    rec.conc_df_to_nc(seas_out_file, trend=False)
    rec.conc_df_to_nc(trend_out_file, trend=True)

    with subtests.test(dataframe='seasonal'):
        expected_df = read_priors_conc_from_netcdf(
            mlo_smo_default_expected_dir / f'{noaa_gas}_seasonal.nc'
        )
        compare_dataframes(expected_df, rec.conc_seasonal)

    with subtests.test(dataframe='trend'):
        expected_df = read_priors_conc_from_netcdf(
            mlo_smo_default_expected_dir / f'{noaa_gas}_trend.nc'
        )
        compare_dataframes(expected_df, rec.conc_trend)


def test_mlo_smo_default_filling_backwards_compat(
    subtests,
    noaa_gas,
    mlo_smo_default_back_compat_expected_dir,
    mlo_smo_default_back_compat_out_dir,
    mlo_smo_default_end_date,
):
    rec_class = tccon_priors.gas_records[noaa_gas]
    rec = rec_class(
        last_date=mlo_smo_default_end_date,
        save_strat=False,
        recalculate_strat_lut=False,
        recalc_if_custom_dates=False,
        use_pre1p6_interpolation=True
    )
    seas_out_file = mlo_smo_default_back_compat_out_dir / f'{noaa_gas}_seasonal.nc'
    trend_out_file = mlo_smo_default_back_compat_out_dir / f'{noaa_gas}_trend.nc'
    rec.conc_df_to_nc(seas_out_file, trend=False)
    rec.conc_df_to_nc(trend_out_file, trend=True)

    with subtests.test(dataframe='seasonal'):
        expected_df = read_priors_conc_from_netcdf(
            mlo_smo_default_back_compat_expected_dir / f'{noaa_gas}_seasonal.nc'
        )
        compare_dataframes(expected_df, rec.conc_seasonal)

    with subtests.test(dataframe='trend'):
        expected_df = read_priors_conc_from_netcdf(
            mlo_smo_default_back_compat_expected_dir / f'{noaa_gas}_trend.nc'
        )
        compare_dataframes(expected_df, rec.conc_trend)


def test_real_smo_gap_filling(
    subtests,
    smo_real_gap_input_dir,
    smo_real_gap_expected_dir,
    smo_real_gap_out_dir,
    mlo_smo_default_end_date,
):
    seas_filename = 'co2_seasonal.nc'
    trend_filename = 'co2_trend.nc'

    rec = tccon_priors.CO2TropicsRecord(
        mlo_file=smo_real_gap_input_dir / 'co2_mlo_monthly.txt',
        smo_file=smo_real_gap_input_dir / 'co2_smo_monthly.txt',
        recalc_if_custom_dates=False,
        recalculate_strat_lut=False,
        save_strat=False,
        last_date=mlo_smo_default_end_date,
    )

    rec.conc_df_to_nc(smo_real_gap_out_dir / seas_filename, trend=False)
    rec.conc_df_to_nc(smo_real_gap_out_dir / trend_filename, trend=True)

    with subtests.test(dataframe='seasonal'):
        expected_df = read_priors_conc_from_netcdf(
            smo_real_gap_expected_dir / seas_filename
        )
        compare_dataframes(expected_df, rec.conc_seasonal)

    with subtests.test(dataframe='trend'):
        expected_df = read_priors_conc_from_netcdf(
            smo_real_gap_expected_dir / trend_filename
        )
        compare_dataframes(expected_df, rec.conc_trend)


def test_mlo_and_smo_gap_filling(
    mlo_gap_test_file,
    smo_gap_test_file,
    gap_test_expected,
    gap_test_output,
    noaa_gas,
    noaa_gap_month,
    mlo_smo_default_end_date,
    smo_gap_only
):
    # Since noaa_gas, noaa_gap_month, and smo_gap_only are parameterized, this will run once
    # for each combination of the inputs
    if smo_gap_only:
        mlo_file = mlo_gap_test_file.get_test_file(noaa_gas, None)
    else:
        mlo_file = mlo_gap_test_file.get_test_file(noaa_gas, noaa_gap_month)
    smo_file = smo_gap_test_file.get_test_file(noaa_gas, noaa_gap_month)
    out_file = gap_test_output.get_file(noaa_gas, noaa_gap_month, smo_only=smo_gap_only, version='pre1.5')
    rec_class = tccon_priors.gas_records[noaa_gas]
    rec = rec_class(
        last_date=mlo_smo_default_end_date,
        mlo_file=mlo_file,
        smo_file=smo_file,
        save_strat=False,
        recalculate_strat_lut=False,
        recalc_if_custom_dates=False,
    )

    # Always write the result to the output directory first so that we can copy it
    # as the expected value if it is correct.
    rec.conc_df_to_nc(out_file, trend=False)

    expected_df = read_priors_conc_from_netcdf(
        gap_test_expected.get_file(noaa_gas, noaa_gap_month, smo_only=smo_gap_only, version='pre1.5')
    )
    compare_dataframes(expected_df, rec.conc_seasonal)


@pytest.mark.slow
def test_mlo_smo_interp_cutoff(
    mlo_gap_test_file,
    smo_gap_test_file,
    interp_cutoff_noaa_file_root,
    interp_cutoff_noaa_expected_dir,
    noaa_gas,
    test_plots_dir,
):
    setup_logger(level=1)
    start_dates = pd.date_range('2000-01-01', '2017-12-01', freq='MS')
    # start_dates = pd.date_range('2010-01-01', '2010-12-01', freq='MS')
    first_diff_date = start_dates.min()
    # Add enough days to be sure we get to or past the first of the month after
    # a leap year
    last_diff_date = start_dates.max() + pd.Timedelta(days=366)

    print('Generating test input files')
    file_pairs = _generate_interp_cutoff_test_files(
        mlo_gap_test_file,
        smo_gap_test_file,
        start_dates,
        gas=noaa_gas,
        root_out_dir=interp_cutoff_noaa_file_root
    )

    print('Running interpolations')
    base_dmfs, test_dmfs = _make_interp_cutoff_data(
        mlo_base_file=mlo_gap_test_file.get_base_file(noaa_gas),
        smo_base_file=smo_gap_test_file.get_base_file(noaa_gas),
        noaa_gas=noaa_gas,
        first_diff_date=first_diff_date,
        last_diff_date=last_diff_date,
        file_pairs=file_pairs
    )

    print('Calculating differences')
    test_deltas = dict()
    for site, site_dmfs in test_dmfs.items():
        test_deltas[site] = dict()
        for cutoff, cutoff_df in site_dmfs.items():
            tmp_df = cutoff_df.apply(lambda col: col - base_dmfs[site])
            test_deltas[site][cutoff] = tmp_df

    test_results = {'deltas': test_deltas, 'test': test_dmfs, 'base': base_dmfs}
    with open(interp_cutoff_noaa_file_root / f'{noaa_gas}_results.pkl', 'wb') as f:
        pickle.dump(test_results, f)

    expected_file = interp_cutoff_noaa_expected_dir / f'{noaa_gas}_rmse.nc'
    expected_rmse_df = msiplt.rmse_df_from_nc(expected_file)
    # Plot the results and save the RMSE values first - this ensures the plots are always made.
    units = tccon_priors.gas_records[noaa_gas]._gas_unit
    rmse_df = msiplt.make_rmse_df(test_results['deltas'])
    msiplt.rmse_df_to_nc(rmse_df, noaa_gas, units, interp_cutoff_noaa_file_root / f'{noaa_gas}_rmse.nc')

    plot_dir = test_plots_dir / 'mlo-smo-interp-cutoff'
    plot_dir.mkdir(exist_ok=True)
    _, axs = plt.subplots(2, 2, figsize=(12,4), gridspec_kw={'hspace': 0.4})
    msiplt.plot_interp_cutoff_rmse(expected_rmse_df, gas_label=noaa_gas.upper(), units=units, ax=axs[0,0])
    msiplt.plot_interp_cutoff_rmse(expected_rmse_df, gas_label=noaa_gas.upper(), units=units, relative=True, ax=axs[0,1])
    msiplt.plot_interp_cutoff_rmse(rmse_df, gas_label=noaa_gas.upper(), units=units, ax=axs[1,0])
    msiplt.plot_interp_cutoff_rmse(rmse_df, gas_label=noaa_gas.upper(), units=units, relative=True, ax=axs[1,1])
    for ax in axs[0]:
        ax.set_title('Expected')
    for ax in axs[1]:
        ax.set_title('Current')
    plt.savefig(plot_dir / f'{noaa_gas}_interpolation_rmse.pdf', bbox_inches='tight')
    plt.close()

    for site in test_dmfs.keys():
        msiplt.plot_delta_vs_month(data=test_results, site=site, gas_label=noaa_gas.upper(), units=units)
        plt.savefig(plot_dir / f'{site}_{noaa_gas}_interpolation_timeseries.pdf', bbox_inches='tight')
        plt.close()

    # Now we do the comparison. 
    compare_dataframes(expected_df=expected_rmse_df, curr_df=rmse_df)


def _make_interp_cutoff_data(
    mlo_base_file,
    smo_base_file,
    noaa_gas,
    first_diff_date,
    last_diff_date,
    file_pairs
):
    """Load the base DMFs from MLO and SMO without any filling,
    and all the results of filling in different gaps, reindexed to
    only the months with real data.
    """
    rec_class = tccon_priors.gas_records[noaa_gas]

    base_dmfs = dict()
    base_mlo_dmf = rec_class.read_insitu_gas(
        full_file_path=mlo_base_file,
    )
    tt = (base_mlo_dmf.index >= first_diff_date) & (base_mlo_dmf.index <= last_diff_date)
    base_dmfs['mlo'] = base_mlo_dmf[noaa_gas]


    base_smo_dmf = rec_class.read_insitu_gas(
        full_file_path=smo_base_file,
    )
    tt = (base_smo_dmf.index >= first_diff_date) & (base_smo_dmf.index <= last_diff_date)
    base_dmfs['smo'] = base_smo_dmf[noaa_gas]

    # Get the date ranges of the sites' raw data before we truncate to the test period.
    # We want the reader function below to keep the full set of data while it reads so
    # that the interpolation has enough data on either end to work with
    first_dates = {k: v.index.min() for k, v in base_dmfs.items()}
    last_dates = {k: v.index.max() for k, v in base_dmfs.items()}

    # Now we want to cut the base DMF timeseries down to the period of interest
    # so that indexing the test series by the base indices ensures that the test
    # series only contain values for months which have real data and are in the test
    # period. The first criterion in needed to deal with missing data in the original
    # files.
    for site, dmfs in base_dmfs.items():
        tt = (dmfs.index >= first_diff_date) & (dmfs.index <= last_diff_date)
        base_dmfs[site] = dmfs[tt]

    test_dmfs = {
        'mlo': dict(),
        'smo': dict()
    }
    for cutoff in np.arange(0, 7):
        for site in ['mlo', 'smo']:
            test_dmfs[site][cutoff] = []
            for key, pair in file_pairs.items():
                this_test_dmf = rec_class.read_and_fill_insitu_gas(
                    full_file_path=pair[site],
                    first_date=first_dates[site],
                    last_date=last_dates[site],
                    truncate_date=None,
                    max_months_simple_interp=cutoff,
                )
                this_test_dmf = this_test_dmf.dmf_mean
                this_test_dmf.name = key
                test_dmfs[site][cutoff].append(this_test_dmf)
            tmp = pd.concat(test_dmfs[site][cutoff], axis=1)
            test_dmfs[site][cutoff] = tmp.loc[base_dmfs[site].index, :]


    return base_dmfs, test_dmfs


def _generate_interp_cutoff_test_files(mlo_gap_test_file, smo_gap_test_file, start_dates, gas: str, root_out_dir: Path) -> list:
    file_pairs = dict()
    gap_widths = np.arange(1, 7)
    for start_date in start_dates:
        dest_dir = root_out_dir / gas / f'gap{start_date:%Y%m}'
        dest_dir.mkdir(exist_ok=True, parents=True)
        for gap in gap_widths:
            mlo_file = mlo_gap_test_file.get_test_file(gas=gas, n_months_missing=gap, start_date=start_date, dest_dir=dest_dir)
            smo_file = smo_gap_test_file.get_test_file(gas=gas, n_months_missing=gap, start_date=start_date, dest_dir=dest_dir)
            key = f'{start_date:%Y%m}_{gap:02d}gap'
            file_pairs[key] = {'mlo': mlo_file, 'smo': smo_file}
    return file_pairs
