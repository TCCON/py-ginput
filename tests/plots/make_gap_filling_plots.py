from dateutil.relativedelta import relativedelta
from ginput.common_utils import readers
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import re

MY_DIR = Path(__file__).parent
TEST_DIR = MY_DIR.parent

def main():
    for gas in ['co2', 'ch4', 'n2o']:
        plot_gap_test_results(gas)
        plot_file = MY_DIR / f'test_mlo_and_smo_gap_filling-{gas}.pdf'
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        print(f'Saved {plot_file}')


def plot_gap_test_results(gas, date_range=[pd.Timestamp(2011,1,1), pd.Timestamp(2015,1,1)]):
    expected_dir = TEST_DIR / 'test_input_data/noaa-interp-extrap/expected/gap-tests/'
    current_dir = TEST_DIR / 'test_output_data/noaa-interp-extrap/expected/gap-tests/'

    both_expected_dmfs = load_gap_test_results(expected_dir / 'mlo-and-smo', gas)
    both_current_dmfs = load_gap_test_results(current_dir / 'mlo-and-smo', gas)
    smo_only_expected_dmfs = load_gap_test_results(expected_dir / 'smo-only', gas)
    smo_only_current_dmfs = load_gap_test_results(current_dir / 'smo-only', gas)

    nmonths = sorted(both_expected_dmfs.columns)
    nrow = len(nmonths)
    _, axs = plt.subplots(nrow, 1, figsize=(12, 3*nrow), sharex=True)

    for n, ax in zip(nmonths, axs):
        _plot_one_gap_result(
            both_expected_dmfs[n], both_current_dmfs[n],
            smo_only_expected_dmfs[n], smo_only_current_dmfs[n],
            date_range, gas, ax
        )
        ax.axvline(pd.Timestamp(2012,1,1), ls='-.', color='gray')
        ax.axvline(pd.Timestamp(2012,1,1) + relativedelta(months=n), ls='-.', color='gray')
        ax.set_title(f'{n} month gap')


def load_gap_test_results(source_dir, gas):
    files = Path(source_dir).glob(f'{gas}_seasonal*.nc')
    dmfs = dict()
    for file in files:
        df = readers.read_priors_conc_from_netcdf(file)
        col = file.stem.split('_')[-1]
        n = int(re.search(r'\d+', col).group())
        dmfs[n] = df.dmf_mean

    return pd.concat(dmfs, axis=1)


def _plot_one_gap_result(both_expected, both_current, smo_expected, smo_current, date_range, gas, ax):
    d1, d2 = date_range
    ax.plot(smo_expected[(smo_expected.index >= d1) & (smo_expected.index < d2)], color='tab:red', marker='x', ls='--', label='SMO only, expected')
    ax.plot(smo_current[(smo_current.index >= d1) & (smo_current.index < d2)], color='tab:blue', marker='+', ls='--', label='SMO only, current')
    ax.plot(both_expected[(both_expected.index >= d1) & (both_expected.index < d2)], color='tab:red', marker='x', ls=':', label='MLO & SMO, expected')
    ax.plot(both_current[(both_current.index >= d1) & (both_current.index < d2)], color='tab:blue', marker='+', ls=':', label='MLO & SMO, current')
    ax.legend(ncol=2, loc='upper left')

    unit = 'ppm' if gas == 'co2' else 'ppb'
    ax.set_ylabel(f'{gas.upper()} ({unit})')
    ax.grid(True)


if __name__ == '__main__':
    main()
