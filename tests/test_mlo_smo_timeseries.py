from ginput.priors import tccon_priors
from ginput.common_utils.test_utils import compare_dataframes
from ginput.common_utils.readers import read_priors_conc_from_netcdf


def test_mlo_smo_default_interp_extrap(subtests, mlo_smo_default_expected_dir, mlo_smo_default_end_date):
    record_classes = {
        'co2': tccon_priors.CO2TropicsRecord,
        'n2o': tccon_priors.N2OTropicsRecord,
        'ch4': tccon_priors.CH4TropicsRecord,
    }

    for gas, klass in record_classes.items():
        rec = klass(last_date=mlo_smo_default_end_date)
        with subtests.test(gas=gas, dataframe='seasonal'):
            expected_df = read_priors_conc_from_netcdf(
                mlo_smo_default_expected_dir / f'{gas}_seasonal.nc'
            )
            compare_dataframes(expected_df, rec.conc_seasonal)

        with subtests.test(gas=gas, dataframe='trend'):
            expected_df = read_priors_conc_from_netcdf(
                mlo_smo_default_expected_dir / f'{gas}_trend.nc'
            )
            compare_dataframes(expected_df, rec.conc_trend)
