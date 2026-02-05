from datetime import datetime
from dateutil.relativedelta import relativedelta
from hashlib import md5, sha1
import os
from pathlib import Path
import pytest
import re
import requests
import tarfile as tf


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow/long-running (deselect with `-m 'not slow'`)"
    )
    config.addinivalue_line(
        "markers", "glacial: mark test as extremely long running (deselect with `-m 'not glacial'`)"
    )


LARGE_FILES_DOI='10.22002/4rgh7-zss31'
_mydir = Path(__file__).parent.resolve()
input_data_dir = _mydir / 'test_input_data'
output_data_dir = _mydir / 'test_output_data'
fo2_dir = input_data_dir / 'fo2'
gap_test_dir = input_data_dir / 'noaa-interp-extrap' / 'monthly-inputs' / 'gap-tests'
gap_test_expected_dir = input_data_dir / 'noaa-interp-extrap' / 'expected' / 'gap-tests'
gap_test_out_dir = output_data_dir / 'noaa-interp-extrap' / 'expected' / 'gap-tests'
_large_file_dir = input_data_dir / 'large-files'
_large_file_out_dir = output_data_dir / 'large-files'

@pytest.fixture(scope='session')
def check_geos_hashes():
    geos_sha_file = input_data_dir / 'geosfp-it' / 'fpit_hashes.sha1'
    if not _check_hash_list(geos_sha_file, input_data_dir):
        raise ValueError('One or more GEOS FP-IT files were missing or had the wrong checksum')



@pytest.fixture(scope='session')
def lat_lon_file():
    return input_data_dir / 'GEOS5124-lat-lon.nc4'


@pytest.fixture(scope='session')
def std_vmr_file():
    return input_data_dir / 'summer_35N.vmr'


@pytest.fixture(scope='session')
def fo2_pre2025_csv():
    return fo2_dir / 'monthly_o2_ljo.pre2025.csv'


@pytest.fixture(scope='session')
def geos_dir(large_files_dir):
    return large_files_dir / 'geos'


@pytest.fixture(scope='session')
def geos_3d_met_files_by_datetime(large_files_dir):
    """Returns the available GEOS FP-IT and IT files as dictionary indexed by their UTC datetime
    """
    geos_dir = large_files_dir / 'geos' / 'met' / 'Nv'
    geos_files = dict()
    date_patterns = [
        (r'\d{8}_\d{4}', '%Y%m%d_%H%M'), # GEOS FP-IT
        (r'\d{4}-\d{2}-\d{2}T\d{4}', '%Y-%m-%dT%H%M') # GEOS IT
    ]
    for file in geos_dir.glob('GEOS*.nc4'):
        for re_pat, date_pat in date_patterns:
            m = re.search(re_pat, file.stem)
            if m is not None:
                date = datetime.strptime(m.group(), date_pat)
                geos_files[date] = file
                continue

    return geos_files


@pytest.fixture(scope='session')
def mod_input_dir():
    return input_data_dir / 'mod_files'


@pytest.fixture(scope='session')
def vmr_input_dir():
    return input_data_dir / 'vmr_files'


@pytest.fixture(scope='session')
def map_input_dir():
    return input_data_dir / 'map_files'


@pytest.fixture(scope='session')
def mod_output_dir():
    return output_data_dir / 'mod_files'


@pytest.fixture(scope='session')
def vmr_output_dir():
    return output_data_dir / 'vmr_files'


@pytest.fixture(scope='session')
def map_output_dir():
    return output_data_dir / 'map_files'


@pytest.fixture(scope='session')
def map_dry_output_dir():
    return output_data_dir / 'map_files_dry'


@pytest.fixture(scope='session')
def test_plots_dir():
    return _mydir / 'plots'


@pytest.fixture(scope='session')
def fo2_v2025_csv():
    return fo2_dir / 'monthly_o2_ljo.v2025.csv'


@pytest.fixture(scope='session')
def fo2_pre2025_expected():
    return fo2_dir / 'monthly_o2_ljo.pre2025.expected.csv'


@pytest.fixture(scope='session')
def fo2_v2025_expected():
    return fo2_dir / 'monthly_o2_ljo.v2025.expected.csv'


@pytest.fixture(scope='session')
def mlo_smo_default_expected_dir():
    return input_data_dir / 'noaa-interp-extrap' / 'expected' / 'default-data' / 'post1.6'

@pytest.fixture(scope='session')
def mlo_smo_default_back_compat_expected_dir():
    return input_data_dir / 'noaa-interp-extrap' / 'expected' / 'default-data' / 'pre1.6'

@pytest.fixture(scope='session')
def mlo_smo_default_out_dir():
    """Returns the directory where the MLO/SMO priors record class tests
    should place the current MLO/SMO mean timeseries netCDF files
    """
    out_dir = output_data_dir / 'noaa-interp-extrap' / 'expected' / 'default-data' / 'post1.6'
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(out_dir)
    return out_dir

@pytest.fixture(scope='session')
def mlo_smo_default_back_compat_out_dir():
    """Returns the directory where the MLO/SMO priors record class tests
    should place the current MLO/SMO mean timeseries netCDF files
    """
    out_dir = output_data_dir / 'noaa-interp-extrap' / 'expected' / 'default-data' / 'pre1.6'
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(out_dir)
    return out_dir

@pytest.fixture(scope='session')
def interp_cutoff_noaa_expected_dir():
    root_dir = input_data_dir / 'noaa-interp-extrap' / 'expected' / 'interp-cutoff'
    return root_dir

@pytest.fixture(scope='session')
def interp_cutoff_noaa_file_root():
    root_dir = output_data_dir / 'noaa-interp-extrap' / 'interp-cutoff-inputs'
    root_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(root_dir)
    return root_dir

@pytest.fixture(scope='session')
def mlo_smo_default_end_date():
    """Return the default end date for the MLO/SMO priors record class timeseries tests.
    Using this for the ``last_date`` keyword ensures the tests are repeatable.
    """
    return datetime(2028, 1, 1)

@pytest.fixture(scope='session')
def smo_real_gap_input_dir():
    """Return the directory with the input monthly average MLO & SMO CO2 files
    that include the large gaps in SMO data in 2024 and 2025.
    """
    return input_data_dir / 'noaa-interp-extrap' / 'monthly-inputs' / 'real-smo-gap'

@pytest.fixture(scope='session')
def smo_real_gap_expected_dir():
    """Return the directory that contains the expected MLO+SMO mean CO2 timeseries
    for the tests using the real MLO & SMO data with the 2024 & 2025 gaps in SMO.
    """
    return input_data_dir / 'noaa-interp-extrap' / 'expected' / 'real-smo-gap'

@pytest.fixture(scope='session')
def smo_real_gap_out_dir():
    """Return the directory that the tests using the real MLO & SMO data with the
    2024 & 2025 gaps in SMO data should write their current mean MLO/SMO timeseries
    netCDF files to.
    """
    out_dir = output_data_dir / 'noaa-interp-extrap' / 'expected' / 'real-smo-gap'
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(out_dir)
    return out_dir

@pytest.fixture(scope='session')
def smo_real_gap_o2_file():
    """Returns the path to an O2 file that goes up to 2024, sufficient for testing
    the files after the SMO data gap ending in Feb 2025.
    """
    return input_data_dir / 'noaa-interp-extrap' / 'monthly-inputs' / 'real-smo-gap' / 'o2_dmf_yearly_2024.txt'

@pytest.fixture(scope='session')
def mlo_gap_test_file():
    """Return an instance which will create and return paths to modified MLO
    input files for different gases with different gaps in the data.

    See :class:`GapTestFile`
    """
    return GapTestFile(site='ML')

@pytest.fixture(scope='session')
def smo_gap_test_file():
    """Return an instance which will create and return paths to modified SMO
    input files for different gases with different gaps in the data.

    See :class:`GapTestFile`
    """
    return GapTestFile(site='SMO')


@pytest.fixture(scope='session')
def mlo_trunc_test_file():
    return TruncationTestFile(site='ML')

@pytest.fixture(scope='session')
def smo_trunc_test_file():
    return TruncationTestFile(site='SMO')

@pytest.fixture(scope='session')
def gap_test_expected():
    """Return an instance which can provide the path a netCDF file containing
    the expected mean MLO/SMO timeseries for a given gas, test gap, etc.
    See :class:`GapTestResultFile`.
    """
    return GapTestResultFile(gap_test_expected_dir)

@pytest.fixture(scope='session')
def gap_test_output():
    """Return an instance which can provide the correct path to write a
    netCDF file containing the mean MLO/SMO timeseries for a given gas,
    test gap, etc. See :class:`GapTestResultFile`.
    """
    out_dir = gap_test_out_dir
    return GapTestResultFile(out_dir, mkdir=True)

@pytest.fixture(scope='session', params=['co2', 'n2o', 'ch4'])
def noaa_gas(request):
    """Use this fixture to parameterize a test over all gases that require
    input from NOAA sites.
    """
    return request.param

@pytest.fixture(scope='session', params=[1, 3, 6, 9, 12, 18, 24])
def noaa_gap_month(request):
    """Use this fixture to parameterize a test over the numbers of months
    for which we test the ability of the MLO/SMO interpolation code to fill in.
    """
    return request.param

@pytest.fixture(scope='session', params=[True, False], ids=['smo-only', 'mlo-and-smo'])
def smo_gap_only(request):
    """Use this fixture to parameterize a test over whether the MLO/SMO interpolation
    code is testing having only one site with a gap (SMO) or both (MLO and SMO).
    """
    return request.param

@pytest.fixture(scope='session')
def test_date():
    """Return the default test date for creating .mod and .vmr files in the end-to-end test."""
    return datetime(2018, 1, 1)


@pytest.fixture(scope='session')
def test_site():
    """Return the default test site for creating .mod and .vmr files in the end-to-end test."""
    return 'oc'


@pytest.fixture(scope='session')
def large_files_dir(pytestconfig):
    _download_large_files(pytestconfig)
    return _large_file_dir

@pytest.fixture(scope='session')
def oco_file_dir(large_files_dir):
    return large_files_dir / 'oco'

@pytest.fixture(scope='session')
def gosat_empty_file_dir(large_files_dir):
    return large_files_dir / 'gosat' / 'empty' 

@pytest.fixture(scope='session')
def gosat_nonempty_file_dir(large_files_dir):
    return large_files_dir / 'gosat' / 'nonempty' 

@pytest.fixture(scope='session')
def oco_file_out_dir():
    out_dir = _large_file_out_dir / 'oco'
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(out_dir)
    return out_dir

@pytest.fixture(scope='session')
def gosat_empty_file_out_dir():
    out_dir = _large_file_out_dir / 'gosat' / 'empty'
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(out_dir)
    return out_dir

@pytest.fixture(scope='session')
def gosat_nonempty_file_out_dir():
    out_dir = _large_file_out_dir / 'gosat' / 'nonempty'
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(out_dir)
    return out_dir

# ---------------- #
# Helper functions #
# ---------------- #

def _check_hash_list(hash_list_filename, input_data_dir):
    hash_dict = _read_hash_list(hash_list_filename, input_data_dir)
    for filename, hash_hex in hash_dict.items():
        new_hash_hex = _hash_file(filename)
        if hash_hex != new_hash_hex:
            return False

    return True


def _read_hash_list(hash_filename, input_data_dir):
    hash_dict = dict()
    with open(hash_filename, 'r') as fobj:
        for line in fobj:
            hash_obj, filename = line.split()
            filename = os.path.join(input_data_dir, filename)
            hash_dict[filename] = hash_obj

    return hash_dict


def _hash_file(filename, hash_class=sha1):
    block_size = 2**16
    hash_obj = hash_class()
    with open(filename, 'rb') as fobj:
        buf = fobj.read(block_size)
        while len(buf) > 0:
            hash_obj.update(buf)
            buf = fobj.read(block_size)

    return hash_obj.hexdigest()


def _ensure_gitignored(dir: Path):
    gitignore = dir / '.gitignore'
    if not gitignore.exists():
        with open(gitignore, 'w') as f:
            f.write('*')


class TruncationTestFile:
    def __init__(self, site: str):
        self.site = site

    def get_test_file(self, gas: str, short: bool) -> Path:
        if short:
            subdir = 'truncation'
            pat = '{site}_monthly_obs_short_{gas}.txt'
        else:
            subdir = 'gap-tests'
            pat = '{site}_monthly_obs_{gas}.txt'

        return input_data_dir / 'noaa-interp-extrap' / 'monthly-inputs' / subdir / pat.format(site=self.site, gas=gas)

class GapTestFile:
    """Creates and provides the path to an MLO or SMO monthly input file for timeseries interpolation tests.

    Calling `get_test_file` on this instance will create the input file for whichever site
    the instance is for with the desired number of months set to NaN to simulate missing data.
    """
    def __init__(self, site):
        self.site = site

    def get_base_file(self, gas: str):
        return gap_test_dir / f'{self.site}_monthly_obs_{gas}.txt'

    def get_test_file(self, gas: str, n_months_missing: int, start_date: datetime = datetime(2012,1,1), dest_dir: Path = gap_test_dir / 'modified-files') -> Path:
        """Create a monthly NOAA input file with some months set to NaN and return the path to that file.

        :param gas: which gas (co2, n2o, or ch4) to create the file for.

        :param n_months_missing: how many months, starting with ``start_date``, to set to NaN to simulate
         missing data. Give ``None`` to use the base file with no missing data.

        :param start_date: first month to set as missing. Must be a date on the first of the month.

        :return: the path to the monthly mean NOAA file.
        """
        base_file = self.get_base_file(gas)
        if n_months_missing is None:
            return base_file

        dest_file = dest_dir / f'{self.site}_month_obs_{gas}_gap_{n_months_missing}.txt'
        self._make_test_noaa_file(base_file, dest_file, n_months_missing, start_date=start_date)
        return dest_file

    @staticmethod
    def _make_test_noaa_file(base_file, dest_file, n_months_missing, start_date):
        end_date = start_date + relativedelta(months=n_months_missing)
        with open(base_file, 'r') as base, open(dest_file, 'w') as dest:
            for line in base:
                if line.strip().startswith('#'):
                    dest.write(line)
                else:
                    site, year, month, value = line.strip().split()
                    year = int(year)
                    month = int(month)
                    date = datetime(year, month, 1)
                    if start_date <= date < end_date:
                        dest.write(f'{site} {year} {month:2d}  NaN\n')
                    else:
                        dest.write(line)


class GapTestResultFile:
    """Provides the path to a netCDF file containing a mean MLO/SMO timeseries result.

    Used for both writing the current result and getting the path for the expected result.

    :param source_dir: directory under which the netCDF files will reside.

    :param mkdir: if ``True``, the correct subdirectory under ``source_dir`` (and any parents)
     will be created if needed, and populated with a ``.gitignore`` file to set to ignore all
     files in that directory. Useful for output directories for current results."""
    def __init__(self, source_dir: os.PathLike, mkdir: bool = False):
        self.source_dir = Path(source_dir)
        self.mkdir = mkdir

    def get_file(self, gas: str, n_months_missing: int, smo_only: bool, version: str, trend: bool = False) -> Path:
        """Get the path to the file containing the MLO/SMO mean timeseries results.

        :param gas: which gas the file contains, e.g. "co2", "n2o", or "ch4".

        :param n_months_missing: how many months were set to missing data for this test.

        :param smo_only: ``True`` if this was for a test with only SMO having missing data,
         ``False`` if both MLO and SMO had missing data.

        :param version: a string to identify which version of the interpolation algorithm
         was used. Used for tests of backwards compatibility.

        :param trend: whether the file contains the deseasonalized trend timeseries (``True``)
         or the seasonal timeseries (``False``).

        :return: path to the netCDF file.
        """
        subdir = 'smo-only' if smo_only else 'mlo-and-smo'
        tgt_dir = self.source_dir / subdir

        if self.mkdir:
            tgt_dir.mkdir(exist_ok=True, parents=True)
            _ensure_gitignored(tgt_dir)

        df_type = 'trend' if trend else 'seasonal'
        if n_months_missing is None:
            return tgt_dir / f'{gas}_{df_type}_{version}.nc'
        else:
            return tgt_dir / f'{gas}_{df_type}_{n_months_missing}months.nc'


def _download_large_files(pytestconfig):
    skip_check = os.getenv('GINPUT_TEST_SKIP_LARGE_FILE_CHECK', '0')
    if skip_check == '1':
        return

    clobber = os.getenv('GINPUT_TEST_CLOBBER_LARGE_FILES', '0') == '1'
    # In order to show that we're downloading and checksumming big files,
    # we disable the stdout capturing for the duration of the download
    # function.
    capmanager = pytestconfig.pluginmanager.getplugin("capturemanager")
    with capmanager.global_and_fixture_disabled():
        print('\n--> Checking if large input files need downloaded')
        _download_from_large_file_record('ginput-large-files.md5')
        to_extract = _check_large_files(clobber=clobber)
        if len(to_extract) == 0:
            return

        print(f'--> {len(to_extract)} large files need downloaded... this will take a few minutes')
        _download_from_large_file_record('ginput-large-files.tgz')
        tarball_file = _large_file_dir / 'ginput-large-files.tgz'
        with tf.open(tarball_file) as tgz:
            for item in to_extract:
                member = tgz.getmember(item)
                tgz.extract(member, path=_large_file_dir, set_attrs=False)

        tarball_file.unlink()
        print(f'--> Removing tarball downloaded from CaltechData ({tarball_file})')


def _check_large_files(md5_file = _large_file_dir / 'ginput-large-files.md5', clobber=False):
    """Get the current .md5 file from the CaltechData repo and see if any input files need downloaded"""
    to_extract = []
    with open(md5_file) as f:
        for line in f:
            expected_checksum, relpath = line.strip().split()
            print(f'--> Checking if large file {relpath} needs downloading')
            abspath = _large_file_dir / relpath
            if not abspath.exists():
                print(f'      * {relpath} must be downloaded (does not exist locally)')
                must_extract = True
            else:
                curr_checksum = _hash_file(abspath, hash_class=md5)
                file_differs = curr_checksum != expected_checksum
                must_extract = clobber and file_differs
                if must_extract:
                    print(f'      * {relpath} must be downloaded (remote file has a different checksum)')
                elif file_differs:
                    print(f'WARNING: {relpath} has a different checksum than expected. ')
            if must_extract:
                to_extract.append(relpath)
    return to_extract


def _download_from_large_file_record(filename: str, max_redirects=10):
    record_id = LARGE_FILES_DOI.split('/')[-1]
    metadata_url = f'https://data.caltech.edu/api/records/{record_id}/files'
    with requests.get(metadata_url) as r:
        r.raise_for_status()
        metadata = r.json()

    url = f'https://data.caltech.edu/api/records/{record_id}/files/{filename}/content'
    record_url = url
    for _ in range(max_redirects):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            if 'Location' in r.headers:
                url = r.headers['Location']
            else:
                progress = ContentProgress(r.headers.get('content-length', None))
                output_file = _large_file_dir / filename
                print(f'--> Downloading {output_file}')
                with open(output_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:  # do not write empty, keep-alive chunks
                            progress.update(len(chunk))
                            f.write(chunk)
                print(f'--> Verifying checksum of {output_file}')
                _verify_caltechdata_checksum(metadata, output_file)
                print('--> Download complete')
                return
    raise RuntimeError(f'Exceeded {max_redirects} redirects trying to access {record_url}')


def _verify_caltechdata_checksum(metadata, output_file):
    expected_md5 = [entry['checksum'].split(':')[1] for entry in metadata.get('entries', []) if entry['key'] == output_file.name]
    if len(expected_md5) != 1:
        raise RuntimeError(f'Found {len(expected_md5)} entries for {output_file.name} in the CaltechData metadata, expected 1')
    expected_md5 = expected_md5[0]
    actual_md5 = _hash_file(output_file, hash_class=md5)
    if expected_md5 != actual_md5:
        raise RuntimeError(f'MD5 checksum for {output_file} did not match: expected {expected_md5} from CaltechData, current value checksum is {actual_md5}')


class ContentProgress:
    def __init__(self, content_length, size_step=50):
        # Not all requests will provide a content length, so if
        # it was None, we'll have to just print the size downloaded.
        if content_length is not None:
            content_length = int(content_length)
            if content_length > 10 * 1024**3:
                self.size_divisor = 1024**3
                self.size_unit = 'GB'
            if content_length > 10 * 1024**2:
                self.size_divisor = 1024**2
                self.size_unit = 'MB'
            elif content_length > 10 * 1024:
                self.size_divisor = 1024
                self.size_unit = 'kB'
            else:
                self.size_divisor = 1
                self.size_unit = 'B'
            self.content_length = int(content_length) / self.size_divisor
        else:
            self.content_length = None
        self.step = size_step
        self.next_size = 0
        self.curr_size = 0

    def update(self, chunk_len: int):
        self.curr_size += chunk_len / self.size_divisor
        if self.curr_size > self.next_size:
            if self.content_length is None:
                print(f'{self.curr_size} {self.size_unit} downloaded')
            else:
                frac = self.curr_size / self.content_length
                print(f'      * {self.curr_size:.1f} of {self.content_length:.1f} {self.size_unit} downloaded ({frac:.1%})')
            self.next_size = (self.curr_size // self.step + 1) * self.step
