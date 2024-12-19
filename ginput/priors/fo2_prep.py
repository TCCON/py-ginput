from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
import pandas as pd
from pathlib import Path
from ..common_utils import versioning, readers, mod_constants
from ..download import get_fo2_data
from ..common_utils.ggg_logging import logger

from typing import Union, Optional

__version__ = '1.0.0'
PROGRAM = f'fo2_prep v.{__version__}'

DEFAULT_FO2_FILE = Path(mod_constants.data_dir) / 'o2_mean_dmf.dat'
DEFAULT_X_O2_REF = 0.209341
FO2_FILE_HEADER = [
    '# Timeseries of dry mole fraction of O2 and the required inputs to calculate it\n',
    '#\n'
    '#\n',
    '# Column definitions:\n',
    '#  - "year": the year which the other values are averaged over\n',
    '#  - "fo2": the dry mole fraction of O2\n',
    '#  - "o2_n2": the delta O2/N2 values in per meg averaged from multiple Scripps sites\n',
    '#  - "do2_n2": the "o2_n2" values with the base year subtracted off\n',
    '#  - "co2": the NOAA marine surface annual global mean CO2 dry mole fraction in ppm\n',
    '#  - "dco2": the "co2" values with the base year substracted off\n'
    '#\n'
]


def parse_args(parser: Optional[ArgumentParser]):
    description = 'Create or update the O2 mole fraction file'
    if parser is None:
        parser = ArgumentParser(description=description)
        am_i_main = True
    else:
        parser.description = description
        am_i_main = False

    parser.add_argument('fo2_dest_file', default=str(DEFAULT_FO2_FILE), nargs='?',
                        help='O2 mole fraction file to create or update. Default is %(default)s.')
    parser.add_argument('--max-num-backups', type=int, default=5,
                        help=' Maximum number of backups of the O2 mole fraction file to keep. Default is %(default)d.')

    if am_i_main:
        return vars(parser.parse_args())
    else:
        parser.set_defaults(driver_fxn=fo2_update_driver)



def fo2_update_driver(fo2_dest_file: Union[str, Path] = DEFAULT_FO2_FILE, max_num_backups: int = 5, time_since_mod: Optional[timedelta] = None):
    """Checks for new versions of the input files needed for f(O2) and updates the f(O2) table file if needed

    Parameters
    ----------
    fo2_dest_file
        Which file containing the calculated f(O2) data to write or update.

    max_num_backups
        Maximum number of backups of the f(O2) file to keep.

    time_since_mod
        If given a timedelta, then this function will return without trying to update if the f(O2) file
        has a modification time more recent than (now - time_since_mod).

    See also
    --------
    - :func:``create_or_update_fo2_file`` if you want to update an f(O2) data file without downloading
      new input data.
    """
    fo2_dest_file = Path(fo2_dest_file)
    if time_since_mod is not None and fo2_dest_file.exists():
        if _check_time_since_modification(fo2_dest_file, time_since_mod):
            logger.info('Will check if fO2 file needs updated')
        else:
            logger.info('Skipping fO2 file update (modified recently enough)')
            return

    dl_dir, _ = get_fo2_data.download_fo2_inputs(only_if_new=True)
    create_or_update_fo2_file(dl_dir, fo2_dest_file, max_num_backups=max_num_backups)


def _check_time_since_modification(fo2_dest_file: Path, time_since_mod: timedelta) -> bool:
    mtime = fo2_dest_file.stat().st_mtime
    mtime = datetime.fromtimestamp(mtime, tz=timezone.utc)
    logger.info(f'fO2 file last updated on {mtime:%Y-%m-%d %H:%M}')
    now = datetime.now(timezone.utc)
    return (now - mtime) > time_since_mod


def create_or_update_fo2_file(fo2_input_data_dir: Union[str, Path], fo2_dest_file: Union[str, Path], max_num_backups: int = 5):
    """Update the f(O2) data file or create a new copy.

    Parameters
    ----------
    fo2_input_data_dir
        Path to a directory containing the input files (co2_annmean_gl.txt, monthly_o2_alt.csv, monthly_o2_cgo.csv, monthly_o2_ljo.csv).

    fo2_dest_file
        Path to the f(O2) file to write or update. If this points to an existing file, only years after then end of the
        existing file will be added. The existing file will be backed up.

    max_num_backups
        The number of backup copies of ``fo2_dest_file`` to keep; if the current number of backups is greater than or equal to
        this number, the oldest one(s) will be removed. Set this to ``None`` to keep all backups.
    """

    # Although our Scripps reader uses the "CO2 filled" column, which contains O2/N2 values 
    # to the end of the current year filled in by a fit, those data should not get included
    # because the NOAA data will almost certainly be released after the Scripps values are
    # updated with real measurements.
    source_files = _fo2_files_from_dir(fo2_input_data_dir)
    new_fo2_df = fo2_from_scripps_o2n2_and_noaa_co2(**source_files).dropna()
    new_fo2_df.index.name = 'year'

    fo2_dest_file = Path(fo2_dest_file)

    if not fo2_dest_file.exists():
        # Creating the file for the first time, use the default header
        logger.info(f'f(O2) file {fo2_dest_file} does not exist, creating initial file')
        prev_file = FO2_FILE_HEADER
        data_descr = f'{new_fo2_df.index.min()} to {new_fo2_df.index.max()}'
        fo2_df = new_fo2_df

    else:
        # File already existed, create a backup and point the header history
        # to that file.
        logger.info(f'f(O2) file {fo2_dest_file} exists, checking if update required')
        fo2_df = readers.read_tabular_file_with_header(fo2_dest_file).set_index('year')
        tt = new_fo2_df.index > fo2_df.index.max()
        if tt.sum() == 0:
            # No new data, nothing to do except to touch the file to ensure future checks
            # based on its modification time recognize that we tried to update it.
            fo2_dest_file.touch()
            logger.info(f'No new f(O2) data (last year in current file = {fo2_df.index.max()}, in new data = {new_fo2_df.index.max()}), not updating the file')
            return

        new_years = new_fo2_df.index[tt]
        new_years_str = ', '.join(str(y) for y in new_years)
        data_descr = f'{new_years.min()} to {new_years.max()}' if len(new_years) > 1 else f'{new_years[0]}'
        logger.info(f'Adding data for {new_years_str} to {fo2_dest_file}')
        fo2_df = pd.concat([fo2_df, new_fo2_df.loc[tt,:]])
        backup_method = versioning.RollingBackupByDate(date_fmt='%Y%m%dT%H%M')
        prev_file = backup_method.make_rolling_backup(fo2_dest_file, max_num_backups=max_num_backups)
        logger.info(f'Backed up current f(O2) file to {prev_file}')

    new_header = versioning.update_versioned_file_header(
        prev_file=prev_file,
        new_data_descr=data_descr,
        program_descr=PROGRAM,
        source_files=source_files,
        insert_line_index=2,  # want this after the first blank line in the header
    )

    with open(fo2_dest_file, 'w') as f:
        f.writelines(new_header)
        fo2_df.reset_index().to_string(f, index=False)
    logger.info(f'Wrote updated {fo2_dest_file}')


def fo2_from_scripps_noaa_dir(fo2_data_dir: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Calculate f(O2) from the data contained in a directory.

    Parameters
    ----------
    fo2_data_dir
        Path to a directory containing the input files (co2_annmean_gl.txt, monthly_o2_alt.csv, monthly_o2_cgo.csv, monthly_o2_ljo.csv).

    kwargs
        Additional keyword arguments for :func:`fo2_from_scripps_o2n2_and_noaa_co2`. The input file arguments
        (``co2gl_file``, ``alt_o2n2_file``, ``cgo_o2n2_file``, and  ``ljo_o2n2_file``) will be provided.
    """
    fo2_data_files = _fo2_files_from_dir(fo2_data_dir)
    return fo2_from_scripps_o2n2_and_noaa_co2(
        co2gl_file=fo2_data_files['co2gl_file'],
        alt_o2n2_file=fo2_data_files['alt_o2n2_file'],
        cgo_o2n2_file=fo2_data_files['cgo_o2n2_file'],
        ljo_o2n2_file=fo2_data_files['ljo_o2n2_file'],
        **kwargs
    )


def _fo2_files_from_dir(fo2_data_dir):
    """Returns a dictionary mapping keywords for :func:`fo2_from_scripps_o2n2_and_noaa_co2` to the corresponding files under ``fo2_data_dir``.

    Note: does not check if the files exist.
    """
    fo2_data_dir = Path(fo2_data_dir)
    return {
        'co2gl_file': fo2_data_dir / 'co2_annmean_gl.txt',
        'alt_o2n2_file': fo2_data_dir / 'monthly_o2_alt.csv',
        'cgo_o2n2_file': fo2_data_dir / 'monthly_o2_cgo.csv',
        'ljo_o2n2_file': fo2_data_dir / 'monthly_o2_ljo.csv',
    }


def fo2_from_scripps_o2n2_and_noaa_co2(co2gl_file: Union[str, Path], alt_o2n2_file: Union[str, Path], cgo_o2n2_file: Union[str, Path],
                                       ljo_o2n2_file: Union[str, Path], base_year: int = 2015, x_o2_ref: float = DEFAULT_X_O2_REF) -> pd.DataFrame:
    """Compute the dry mole fraction of O2 in the atmosphere using NOAA global annual mean CO2 and Scripps O2/N2 ratio data

    Parameters
    ----------
    co2gl_file
        Path to the NOAA global annual mean CO2 file.

    alt_o2n2_file
        Path to the Scripps O2/N2 ratio file for Alert, NWT, Canada.

    cgo_o2n2_file
        Path to the Scripps O2/N2 ratio file for Cape Grim, Australia.

    ljo_o2n2_file
        Path to the Scripps O2/N2 ratio file for La Jolla Pier, California, USA.

    base_year
        The year for which ``x_o2_ref`` is defined.

    x_o2_ref
        The dry O2 mole fraction in the atmosphere during ``base_year``.

    Returns
    -------
    DataFrame
        A dataframe containing the dry mole fraction of O2 (as the column "fo2") along with the inputs needed to
        calculate it. Note that this will contain NAs for years with some data (e.g. Scripps but not NOAA) so
        be sure to call ``.dropna()`` on it if you only want complete years.
    """
    co2gl = _read_co2gl_file(co2gl_file)['mean']
    d_co2gl = co2gl - co2gl[base_year]

    # The "CO2 filled column" is the actual O2/N2 measurements (yes, confusing that it is called "CO2", I think it's supposed
    # to be like "C(O2)" not "carbon dioxide") but with missing values filled in by a fit. Using that simplifies the calculation
    # because we don't need to deal with fill values, and should not introduce a significant error, especially since the NOAA
    # data will usually be the latency-limited one
    o2_n2 = _read_global_mean_o2n2(alt_o2n2_file=alt_o2n2_file, cgo_o2n2_file=cgo_o2n2_file, ljo_o2n2_file=ljo_o2n2_file,
                                   yearly_avg=True, keep_datetime_index=False, column='CO2 filled')
    d_o2_n2 = o2_n2 - o2_n2[base_year]

    d_xo2 = _delta_xo2_explicit_xco2(d_o2_n2, d_co2=d_co2gl, x_co2=co2gl)
    fo2_df = d_xo2 + x_o2_ref
    return pd.DataFrame({'fo2': fo2_df, 'o2_n2': o2_n2, 'd_o2_n2': d_o2_n2, 'co2': co2gl, 'd_co2': d_co2gl})


def _delta_xo2_explicit_xco2(d_o2_n2, d_co2, x_co2, x_o2_ref=DEFAULT_X_O2_REF):
    """Calculate the change in the O2 mole fraction relative to a reference value.
    See Appendix E2 of Laughner et al. 2024 (https://doi.org/10.5194/essd-16-2197-2024)
    for the derivation.

    Parameters
    ----------
    d_o2_n2
        The difference in O2/N2 ratios versus the base year ``x_o2_ref`` is for, in units of
        per meg.

    d_co2
        The difference the CO2 dry mole fraction versus the base year ``x_o2_ref`` is for, in
        units of ppm.

    x_co2
        CO2 dry mole fraction for the year for which the O2 mole fraction is being calculated,
        in units of ppm.

    x_o2_ref
        The reference O2 mole fraction for the base year, in units of mol/mol.

    Returns
    -------
    delta_xo2
        The change in O2 mole fraction from the base year, in units of mol/mol.
    """
    return (1 - x_o2_ref) * x_o2_ref * d_o2_n2*1e-6 - x_o2_ref * 1e-6 * d_co2 / (1 - 1e-6*x_co2)


def _read_co2gl_file(co2_file, datetime_index=False):
    """Read the NOAA global mean CO2 file.
    """
    with open(co2_file) as f:
        while True:
            line = f.readline()
            if line.startswith('# year'):
                break

        columns = line[1:].split()
        df = pd.read_csv(f, sep=r'\s+')
        df.columns = columns
        if datetime_index:
            df.index = pd.to_datetime({'year': df['year'], 'month': 7, 'day': 1})
        else:
            df.index = df['year']
        return df


def _read_global_mean_o2n2(alt_o2n2_file, cgo_o2n2_file, ljo_o2n2_file, yearly_avg=False, keep_datetime_index=False, column='CO2'):
    """Read the three Scripps O2/N2 files and average them to produce a global estimate O2/N2 ratio.

    The use of Alert and La Jolla to represent the northern hemisphere and Cape Grim the southern
    hemisphere was recommended by Brit Stephens.
    """
    alt_o2n2 = _read_o2n2_file(alt_o2n2_file)[column]
    cgo_o2n2 = _read_o2n2_file(cgo_o2n2_file)[column]
    ljo_o2n2 = _read_o2n2_file(ljo_o2n2_file)[column]
    global_mean = (alt_o2n2 + ljo_o2n2)/4 + cgo_o2n2/2
    global_mean.name = 'd(o2/n2)'
    if yearly_avg and keep_datetime_index:
        global_mean = global_mean.groupby(lambda i: i.year).mean()
        global_mean.index = pd.to_datetime({'year': global_mean.index, 'month': 7, 'day': 1})
        return global_mean
    if yearly_avg:
        return global_mean.groupby(lambda i: i.year).mean()
    else:
        return global_mean


def _read_o2n2_file(o2n2_file):
    """Read one of the Scripps O2/N2 files.
    """
    with open(o2n2_file) as f:
        line = f.readline()
        while line.startswith('"'):
            line = f.readline()

        # The header *should* be the first line that doesn't start with a quote mark
        columns = [x.strip() for x in line.split(',')]
        # The next line continues the header for some columns
        line = f.readline()
        columns = [f'{c} {x.strip()}'.strip() for c, x in zip(columns, line.split(','))]

        df = pd.read_csv(f, header=None, na_values='-99.99')
        df.columns = columns

    # Make a proper datetime index
    df.index = pd.to_datetime({'year': df['Yr'], 'month': df['Mn'], 'day': 1})
    return df
