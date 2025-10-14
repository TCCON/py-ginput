import numpy as np
import pickle
import unittest

from . import test_utils
from ..priors import fo2_prep


class TestReadScrippsFiles(unittest.TestCase):
    def test_read_pre2025_file(self):
        self.check_scripps_files(pkl_file=test_utils.fo2_pre2025_pkl, csv_file=test_utils.fo2_pre2025_csv)

    def test_read_v2025_file(self):
        self.check_scripps_files(pkl_file=test_utils.fo2_v2025_pkl, csv_file=test_utils.fo2_v2025_csv)


    def check_scripps_files(self, pkl_file, csv_file):
        curr_df = fo2_prep._read_o2n2_file(csv_file)
        with open(pkl_file, 'rb') as f:
            expected_df = pickle.load(f)
        self._compare_dataframes(expected_df, curr_df)


    def _compare_dataframes(self, expected_df, curr_df):
        self.assertEqual(expected_df.columns.to_list(), curr_df.columns.to_list(), msg='Column names do not match expected')
        self.assertEqual(expected_df.index.to_list(), curr_df.index.to_list(), msg='Index values do not match expected')
        bad_columns = []
        for colname, colvals in expected_df.items():
            if not np.allclose(colvals.to_numpy(), curr_df[colname].to_numpy(), equal_nan=True):
                bad_columns.append(colname)

        self.assertTrue(
            len(bad_columns) == 0,
            msg=f'Some columns do not match. Expected:\n{expected_df[bad_columns]}\nCurrent:\n{curr_df[bad_columns]}'
        )

if __name__ == '__main__':
    unittest.main()
