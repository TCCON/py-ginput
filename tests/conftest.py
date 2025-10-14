from datetime import datetime
from pathlib import Path
import pytest


_mydir = Path(__file__).parent.resolve()
input_data_dir = _mydir / '..' / 'ginput' / 'testing' / 'test_input_data'
fo2_dir = input_data_dir / 'fo2'


@pytest.fixture
def lat_lon_file():
    return input_data_dir / 'GEOS5124-lat-lon.nc4'


@pytest.fixture
def fo2_pre2025_csv():
    return fo2_dir / 'monthly_o2_ljo.pre2025.csv'


@pytest.fixture
def fo2_v2025_csv():
    return fo2_dir / 'monthly_o2_ljo.v2025.csv'


@pytest.fixture
def fo2_pre2025_pkl():
    return fo2_dir / 'monthly_o2_ljo.pre2025.pkl'


@pytest.fixture
def fo2_v2025_pkl():
    return fo2_dir / 'monthly_o2_ljo.v2025.pkl'


@pytest.fixture
def test_date():
    return datetime(2018, 1, 1)
