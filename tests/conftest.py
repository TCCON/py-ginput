from datetime import datetime
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
def mlo_smo_default_end_date():
    return datetime(2028, 1, 1)

@pytest.fixture(scope='session')
def test_date():
    return datetime(2018, 1, 1)


@pytest.fixture(scope='session')
def test_site():
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
