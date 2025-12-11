from ginput.priors import tccon_priors
from ginput.common_utils.test_utils import compare_dataframes
from ginput.common_utils.readers import read_priors_conc_from_netcdf


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
