import numpy as np

def compare_dataframes(expected_df, curr_df, negate=False):
    assert expected_df.columns.to_list() == curr_df.columns.to_list(
    ), 'Column names do not match expected'
    assert expected_df.index.to_list() == curr_df.index.to_list(
    ), 'Index values do not match expected'
    bad_columns = []
    for colname, colvals in expected_df.items():
        if not np.allclose(colvals.to_numpy(), curr_df[colname].to_numpy(), equal_nan=True):
            bad_columns.append(colname)

    if negate:
        assert len(bad_columns) > 0, 'Dataframes are identical when they should not be'
    else:
        # For some reason, autoformatting kept trying to break this incorrectly
        # when it was a single-quoted string.
        msg = f'''Some columns do not match. Expected:
{expected_df[bad_columns]}
Current:
{curr_df[bad_columns]}'''
        assert len(bad_columns) == 0, msg


def simple_xco2_calc(co2_profiles, pressure_levels, surface_pressure):
    surface_pressure[surface_pressure < -999] = np.nan
    pressure_levels[pressure_levels < -999] = np.nan
    co2_profiles[co2_profiles < -999] = np.nan

    if np.nanmean(pressure_levels[:,:,0]) > np.nanmean(pressure_levels[:,:,-1]):
        raise ValueError('pressure levels must be ordered space-to-surface')

    dp = np.full(pressure_levels.shape, np.nan)
    pressure_edges = 0.5 * (pressure_levels[:,:,:-1] + pressure_levels[:,:,1:])
    dp[:,:,1:-1] = pressure_edges[:,:,1:] - pressure_edges[:,:,:-1]
    dp[:,:,0] = pressure_levels[:,:,0]
    dp[:,:,-1] = surface_pressure - pressure_edges[:,:,-1]

    pres_wt = dp / np.nansum(dp, axis=2, keepdims=True)
    tmp = co2_profiles * pres_wt
    all_nan = np.all(np.isnan(tmp), axis=2)
    xco2 = np.nansum(tmp, axis=2)
    xco2[all_nan] = np.nan
    return xco2
