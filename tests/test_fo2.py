import pickle

from ginput.priors import fo2_prep
from ginput.common_utils.test_utils import compare_dataframes


def test_read_pre2025_file(fo2_pre2025_pkl, fo2_pre2025_csv):
    """This test confirms that the reader for the Scripps O2 file format
    before 2025 still reads one such file in correctly.

    The Scripps file format changed slightly in 2025; this test ensures
    we retain backwards compatibility with the old format.
    """
    _check_scripps_files(pkl_file=fo2_pre2025_pkl, csv_file=fo2_pre2025_csv)


def test_read_v2025_file(fo2_v2025_pkl, fo2_v2025_csv):
    """This test confirms that the reader for the Scripps O2 file format
    starting in 2025 still reads one such file in correctly.

    The Scripps file format changed slightly in 2025; this test ensures
    that the reader still works for this version.
    """
    _check_scripps_files(pkl_file=fo2_v2025_pkl, csv_file=fo2_v2025_csv)


def _check_scripps_files(pkl_file, csv_file):
    curr_df = fo2_prep._read_o2n2_file(csv_file)
    with open(pkl_file, 'rb') as f:
        expected_df = pickle.load(f)
    compare_dataframes(expected_df, curr_df)
