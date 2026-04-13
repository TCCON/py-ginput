from datetime import datetime
from ginput.priors import tccon_priors
from ginput.common_utils.test_utils import compare_dataframes
import numpy as np
import pytest

@pytest.mark.slow
def test_truncation(subtests, noaa_gas, mlo_trunc_test_file, smo_trunc_test_file, mlo_smo_default_end_date):
    """This test verifies that the ``truncate_date`` argument of the MLO/SMO gas classes works correctly.

    The ``truncate_date`` argument must force those classes to behave exactly as if
    they received input MLO and SMO timeseries that end on that month. This is required
    to allow reproducibility of old priors when using more recent NOAA data.
    """
    rec_class = tccon_priors.gas_records[noaa_gas]
    mlo_full_file = mlo_trunc_test_file.get_test_file(noaa_gas, short=False)
    mlo_short_file = mlo_trunc_test_file.get_test_file(noaa_gas, short=True)
    smo_full_file = smo_trunc_test_file.get_test_file(noaa_gas, short=False)
    smo_short_file = smo_trunc_test_file.get_test_file(noaa_gas, short=True)

    rec_short = rec_class(
        last_date=mlo_smo_default_end_date,
        save_strat=False,
        recalculate_strat_lut=True,
        mlo_file=mlo_short_file,
        smo_file=smo_short_file,
    )

    rec_trunc = rec_class(
        last_date=mlo_smo_default_end_date,
        truncate_date=datetime(2017,12,1),
        save_strat=False,
        recalculate_strat_lut=True,
        mlo_file=mlo_full_file,
        smo_file=smo_full_file,
    )

    with subtests.test(table='seasonal timeseries'):
        compare_dataframes(rec_short.conc_seasonal, rec_trunc.conc_seasonal)
    with subtests.test(table='trend timeseries'):
        compare_dataframes(rec_short.conc_trend, rec_trunc.conc_trend)

    assert rec_short.conc_strat.keys() == rec_trunc.conc_strat.keys()

    for key, short_arr in rec_short.conc_strat.items():
        trunc_arr = rec_trunc.conc_strat[key]
        with subtests.test(table=f'strat {key}'):
            assert np.allclose(
                short_arr.data,
                trunc_arr.data,
                equal_nan=True
            )

    # Check the converse, that creating a record without truncating does
    # not equal the shorter input result
    rec_full = rec_class(
        last_date=mlo_smo_default_end_date,
        save_strat=False,
        recalculate_strat_lut=True,
        mlo_file=mlo_full_file,
        smo_file=smo_full_file,
    )

    with subtests.test(table='seasonal timeseries converse'):
        compare_dataframes(rec_short.conc_seasonal, rec_full.conc_seasonal, negate=True)
    with subtests.test(table='trend timeseries converse'):
        compare_dataframes(rec_short.conc_trend, rec_full.conc_trend, negate=True)

    assert rec_short.conc_strat.keys() == rec_full.conc_strat.keys()

    for key, short_arr in rec_short.conc_strat.items():
        trunc_arr = rec_full.conc_strat[key]
        with subtests.test(table=f'strat {key} converse'):
            assert not np.allclose(
                short_arr.data,
                trunc_arr.data,
                equal_nan=True
            )
