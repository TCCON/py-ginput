import numpy as np
import pickle

from ginput.priors import fo2_prep


def test_read_pre2025_file(fo2_pre2025_pkl, fo2_pre2025_csv):
    _check_scripps_files(pkl_file=fo2_pre2025_pkl, csv_file=fo2_pre2025_csv)


def test_read_v2025_file(fo2_v2025_pkl, fo2_v2025_csv):
    _check_scripps_files(pkl_file=fo2_v2025_pkl, csv_file=fo2_v2025_csv)


def _check_scripps_files(pkl_file, csv_file):
    curr_df = fo2_prep._read_o2n2_file(csv_file)
    with open(pkl_file, 'rb') as f:
        expected_df = pickle.load(f)
    _compare_dataframes(expected_df, curr_df)


def _compare_dataframes(expected_df, curr_df):
    assert expected_df.columns.to_list() == curr_df.columns.to_list(
    ), 'Column names do not match expected'
    assert expected_df.index.to_list() == curr_df.index.to_list(
    ), 'Index values do not match expected'
    bad_columns = []
    for colname, colvals in expected_df.items():
        if not np.allclose(colvals.to_numpy(), curr_df[colname].to_numpy(), equal_nan=True):
            bad_columns.append(colname)

    # For some reason, autoformatting kept trying to break this incorrectly
    # when it was a single-quoted string.
    msg = f'''Some columns do not match. Expected:
{expected_df[bad_columns]}
Current:
{curr_df[bad_columns]}'''
    assert len(bad_columns) == 0, msg
