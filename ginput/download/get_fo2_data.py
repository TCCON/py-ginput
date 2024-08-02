from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import requests

from ..common_utils import mod_constants as const
from ..common_utils import mod_utils
from ..common_utils.ggg_logging import logger

from typing import Optional, Union

DEFAULT_OUT_DIR = Path(const.data_dir) / 'fo2_data'

def parse_args(parser: Optional[ArgumentParser] = None):
    description = 'Download files necessary to compute the O2 mole fraction'
    if parser is None:
        parser = ArgumentParser(description=description)
        am_i_main = True
    else:
        parser.description = description
        am_i_main = False

    parser.add_argument('out_dir', nargs='?', default=DEFAULT_OUT_DIR,
                        help='Directory in which to create subdirectories or download the files if --no-make-subdir is given. '
                             'By default, will download into the ginput data directory (%(default)s).')
    parser.add_argument('--no-make-subdir', dest='make_subdir', action='store_false', 
                        help='By default, the files will be downloaded into a subdirectory of OUT_DIR, named by download date & time. '
                             'That subdirectory will be created if needed, but the parent must exist. Use this flag to download directly into '
                             'OUT_DIR.')
    parser.add_argument('--only-if-new', action='store_true', help='Give this flag to only save the files if the contents changed since the last download within OUT_DIR.')

    if am_i_main:
        clargs = vars(parser.parse_args())
        return clargs
    else:
        parser.set_defaults(driver_fxn=download_fo2_inputs)


def download_fo2_inputs(out_dir: Union[str, Path] = DEFAULT_OUT_DIR, make_subdir: bool = True, only_if_new: bool = False) -> (Path, bool):
    """Download the required inputs (NOAA global mean CO2 and Scripps O2/N2 data) to calculate f(O2).

    Scripps data are available at https://scrippso2.ucsd.edu/data.html.
    NOAA data are available at https://gml.noaa.gov/ccgg/trends/gl_data.html.

    Parameters
    ----------
    out_dir
        Directory to download the files to, behavior also controlled by ``make_subdir``. This directory
        must exist.

    make_subdir
        If ``True``, then this function creates subdirectory inside ``out_dir`` named "fo2_inputs_%Y%m%dT%H%M%S",
        where the part beginning with "%Y" is the current timestamp, and the files are saved to that subdirectory.
        If ``False``, then the files are saved directly in ``out_dir``.

    only_if_new
        If ``True``, then the files will only be saved (and the subdirectory created given ``make_subdir = True``) if
        the files downloaded are different than the previously downloaded.

    Returns
    -------
    Path
        Path to the directory where the files can be found. Given ``only_if_new = True``, this will point to a previous
        download directory if the files have not changed since the last download.

    bool
        ``True`` if new files were downloaded, ``False`` otherwise.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise IOError(f'Target download directory, {out_dir}, does not exist or is not a directory')
    
    urls = {
        'monthly_o2_alt.csv': 'https://scrippso2.ucsd.edu/assets/data/o2_data/monthly/monthly_o2_alt.csv',
        'monthly_o2_ljo.csv': 'https://scrippso2.ucsd.edu/assets/data/o2_data/monthly/monthly_o2_ljo.csv',
        'monthly_o2_cgo.csv': 'https://scrippso2.ucsd.edu/assets/data/o2_data/monthly/monthly_o2_cgo.csv',
        'co2_annmean_gl.txt': 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_gl.txt',
    }

    logger.info(f'Retrieving data from {len(urls)} URLs...')
    file_content = {k: _retrieve_url(v) for k, v in urls.items()}
    logger.info(f'{len(file_content)} URLs retrieved.')
    if only_if_new:
        prev_dir = _check_if_files_changed(out_dir, make_subdir, file_content)
        if prev_dir is not None:
            logger.info('MD5 sums match existing files, not saving new files.')
            return (prev_dir, False)
    
    if make_subdir:
        out_dir = out_dir / f'fo2_inputs_{datetime.now():%Y%m%dT%H%M%S}'
        out_dir.mkdir()
        logger.info(f'Created directory {out_dir}')

    for filename, content in file_content.items():
        out_file = out_dir / filename
        with open(out_file, 'wb') as f:
            f.write(content)
        logger.info(f'Wrote {out_file}')

    return (out_dir, True)
    

def _retrieve_url(url: str) -> bytes:
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f'Failed to download {url}, status code = {r.status_code}')
    else:
        return r.content
    

def _check_if_files_changed(out_dir: Path, subdirs: bool, file_content: dict) -> Optional[Path]:
    if subdirs:
        possible_subdirs = sorted([p for p in out_dir.glob('fo2_inputs_*') if p.is_dir()])
        if len(possible_subdirs) == 0:
            # No existing directories, so need to download
            return None
        else:
            # Otherwise, get the last one lexigraphically, which will be the most
            # recent one because they're named by year-month-day-hour-minute-second
            prev_dir = possible_subdirs[-1]
    else:
        prev_dir = out_dir

    for filename, content in file_content.items():
        prev_file = prev_dir / filename
        if not prev_file.exists():
            return None
        
        content_hash = mod_utils.compute_bytes_checksum(content)
        file_hash = mod_utils.compute_file_checksum(prev_file)
        if content_hash != file_hash:
            return None

    # If all files are present and have the same hash, then no need to download
    return prev_dir


if __name__ == '__main__':
    clargs = parse_args()
    download_fo2_inputs(**clargs)
