from datetime import datetime
from dateutil.relativedelta import relativedelta
from hashlib import sha1
import os
from pathlib import Path
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow/long-running (deselect with `-m 'not slow'`)"
    )


_mydir = Path(__file__).parent.resolve()
input_data_dir = _mydir / 'test_input_data'
output_data_dir = _mydir / 'test_output_data'
fo2_dir = input_data_dir / 'fo2'
gap_test_dir = input_data_dir / 'noaa-interp-extrap' / 'monthly-inputs' / 'gap-tests'
gap_test_expected_dir = input_data_dir / 'noaa-interp-extrap' / 'expected' / 'gap-tests'
gap_test_out_dir = output_data_dir / 'noaa-interp-extrap' / 'expected' / 'gap-tests'

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
def geos_fpit_dir():
    return input_data_dir / 'geosfp-it'


@pytest.fixture(scope='session')
def mod_input_dir():
    return input_data_dir / 'mod_files' / 'fpit'


@pytest.fixture(scope='session')
def test_site_mod_input_dir(mod_input_dir, test_site):
    return mod_input_dir / test_site / 'vertical'


@pytest.fixture(scope='session')
def vmr_input_dir():
    return input_data_dir / 'vmr_files' / 'fpit'


@pytest.fixture(scope='session')
def map_input_dir():
    return input_data_dir / 'map_files' / 'fpit'


@pytest.fixture(scope='session')
def mod_output_dir():
    return output_data_dir / 'mod_files' / 'fpit'


@pytest.fixture(scope='session')
def vmr_output_dir():
    return output_data_dir / 'vmr_files' / 'fpit'


@pytest.fixture(scope='session')
def map_output_dir():
    return output_data_dir / 'map_files' / 'fpit'


@pytest.fixture(scope='session')
def map_dry_output_dir():
    return output_data_dir / 'map_files_dry' / 'fpit'


@pytest.fixture(scope='session')
def test_plots_dir():
    return _mydir / 'plots'


@pytest.fixture(scope='session')
def fo2_v2025_csv():
    return fo2_dir / 'monthly_o2_ljo.v2025.csv'


@pytest.fixture(scope='session')
def fo2_pre2025_pkl():
    return fo2_dir / 'monthly_o2_ljo.pre2025.pkl'


@pytest.fixture(scope='session')
def fo2_v2025_pkl():
    return fo2_dir / 'monthly_o2_ljo.v2025.pkl'


@pytest.fixture(scope='session')
def mlo_smo_default_expected_dir():
    return input_data_dir / 'noaa-interp-extrap' / 'expected' / 'default-data'

@pytest.fixture(scope='session')
def mlo_smo_default_out_dir():
    """Returns the directory where the MLO/SMO priors record class tests
    should place the current MLO/SMO mean timeseries netCDF files
    """
    out_dir = output_data_dir / 'noaa-interp-extrap' / 'expected' / 'default-data'
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignored(out_dir)
    return out_dir

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


def _hash_file(filename):
    block_size = 2**16
    hash_obj = sha1()
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


class GapTestFile:
    """Creates and provides the path to an MLO or SMO monthly input file for timeseries interpolation tests.

    Calling `get_test_file` on this instance will create the input file for whichever site
    the instance is for with the desired number of months set to NaN to simulate missing data.
    """
    def __init__(self, site):
        self.site = site

    def get_test_file(self, gas: str, n_months_missing: int, start_date: datetime = datetime(2010,1,1)) -> Path:
        """Create a monthly NOAA input file with some months set to NaN and return the path to that file.

        :param gas: which gas (co2, n2o, or ch4) to create the file for.

        :param n_months_missing: how many months, starting with ``start_date``, to set to NaN to simulate
         missing data. Give ``None`` to use the base file with no missing data.

        :param start_date: first month to set as missing. Must be a date on the first of the month.

        :return: the path to the monthly mean NOAA file.
        """
        base_file = gap_test_dir / f'{self.site}_monthly_obs_{gas}.txt'
        if n_months_missing is None:
            return base_file

        dest_file = gap_test_dir / 'modified-files' / f'{self.site}_month_obs_{gas}_gap_{n_months_missing}.txt'
        self._make_test_noaa_file(base_file, dest_file, n_months_missing, start_date=start_date)
        return dest_file

    @staticmethod
    def _make_test_noaa_file(base_file, dest_file, n_months_missing, start_date=datetime(2010,1,1)):
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
