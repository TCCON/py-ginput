from __future__ import print_function

from argparse import ArgumentParser
import datetime as dt
from glob import glob
from hashlib import sha1
import os
import shutil
import sys

from ..download import get_GEOS5
from ..common_utils import mod_utils

_mydir = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))

input_data_dir = os.path.join(_mydir, 'test_input_data')
geos_fp_dir = os.path.join(input_data_dir, 'geosfp-it')
geos_sha_file = os.path.join(geos_fp_dir, 'fp_hashes.sha1')
mod_input_dir = os.path.join(input_data_dir, 'mod_files', 'fpit')
vmr_input_dir = os.path.join(input_data_dir, 'vmr_files', 'fpit')
map_input_dir = os.path.join(input_data_dir, 'map_files', 'fpit')
std_vmr_file = os.path.join(input_data_dir, 'summer_35N.vmr')
lat_lon_file = os.path.join(input_data_dir, 'GEOS5124-lat-lon.nc4')

fo2_dir = os.path.join(input_data_dir, 'fo2')
fo2_pre2025_csv = os.path.join(fo2_dir, 'monthly_o2_ljo.pre2025.csv')
fo2_v2025_csv = os.path.join(fo2_dir, 'monthly_o2_ljo.v2025.csv')
fo2_pre2025_pkl = os.path.join(fo2_dir, 'monthly_o2_ljo.pre2025.pkl')
fo2_v2025_pkl = os.path.join(fo2_dir, 'monthly_o2_ljo.v2025.pkl')

output_data_dir = os.path.join(_mydir, 'test_output_data')
mod_output_top_dir = os.path.join(output_data_dir, 'mod_files')
mod_output_dir = os.path.join(mod_output_top_dir, 'fpit')
vmr_output_dir = os.path.join(output_data_dir, 'vmr_files', 'fpit')
map_output_dir = os.path.join(output_data_dir, 'map_files', 'fpit')
map_dry_output_dir = os.path.join(output_data_dir, 'map_files_dry', 'fpit')
test_plots_dir = os.path.join(_mydir, 'plots')

test_date = dt.datetime(2018, 1, 1)
test_site = 'oc'


# Python 2/3 compatibility. Never use "input" in Py2 because it executes whatever input it receives as code
if sys.version_info.major == 2:
    input = raw_input


class InputDataMismatchError(Exception):
    pass


class UserCancelledDownloadError(Exception):
    pass


def hash_file(filename):
    block_size = 2**16
    hash_obj = sha1()
    with open(filename, 'rb') as fobj:
        buf = fobj.read(block_size)
        while len(buf) > 0:
            hash_obj.update(buf)
            buf = fobj.read(block_size)

    return hash_obj.hexdigest()


def write_hash_list(hash_filename, files_to_hash):
    with open(hash_filename, 'w') as wobj:
        for f in files_to_hash:
            sha_sum = hash_file(f)
            # Strip off the leading path to the test data directory, when checking, we add that back on
            # so that the hash file can be used across machines
            local_f = f.replace(input_data_dir, '')
            local_f = local_f.lstrip('/')
            wobj.write('{hash}  {file}\n'.format(hash=sha_sum, file=local_f))


def read_hash_list(hash_filename):
    hash_dict = dict()
    with open(hash_filename, 'r') as fobj:
        for line in fobj:
            hash_obj, filename = line.split()
            filename = os.path.join(input_data_dir, filename)
            hash_dict[filename] = hash_obj

    return hash_dict


def check_hash_list(hash_list_filename):
    hash_dict = read_hash_list(hash_list_filename)
    for filename, hash_hex in hash_dict.items():
        new_hash_hex = hash_file(filename)
        if hash_hex != new_hash_hex:
            return False

    return True


def download_test_geos_data(rebuild=False, rebuild_hash_only=False):
    # Check if we need to download files. We should only need to do this if some of the files are missing. If redownload
    # is True though, that means we're recreating the test data set so we need to delete the existing directories
    # and also recreate the hash file.

    # check hashes, unless we're forcing a rebuild (where we redownload new files and write their hashes to the record)
    if not rebuild and not rebuild_hash_only:
        if not check_hash_list(geos_sha_file):
            print('One or more SHA1 hashes for GEOS FP files do not match expected, or files are missing.\n'
                  'Redownload now (~2 GB)? (y to download, anything else to abort)')
            user_ans = input('')
            if user_ans.lower().strip() != 'y':
                raise UserCancelledDownloadError(
                    'User elected not to download GEOS FPIT data')

        else:
            return

    if not rebuild_hash_only:
        if os.path.exists(os.path.join(geos_fp_dir, 'Nv')):
            shutil.rmtree(os.path.join(geos_fp_dir, 'Nv'))
        if os.path.exists(os.path.join(geos_fp_dir, 'Nx')):
            shutil.rmtree(os.path.join(geos_fp_dir, 'Nx'))

        get_GEOS5.driver(date_range=[test_date, test_date+dt.timedelta(days=1)], mode='FPIT', path=geos_fp_dir,
                         filetypes=('met', 'met', 'chm'), levels=('surf', 'eta', 'eta'))

    if rebuild or rebuild_hash_only:
        surf_files = glob(os.path.join(geos_fp_dir, 'Nx', 'GEOS*.nc4'))
        prof_files = glob(os.path.join(geos_fp_dir, 'Nv', 'GEOS*.nc4'))
        all_files = sorted(surf_files + prof_files)
        write_hash_list(geos_sha_file, all_files)
    elif not check_hash_list(geos_sha_file):
        raise InputDataMismatchError('After downloading the GEOS FP data, the hashes still do not match what is '
                                     'expected. The most likely explanation is that a new version of GEOS-FPIT was '
                                     'released. If you are not a maintainer for GGG, please contact one and provide '
                                     'this error message. If you are a maintainer, then you will need to rerun the '
                                     'test suite with the new GEOS-FPIT version and ensure the correct .mod files are '
                                     'produced, then update the SHA1 hash file.')


def iter_mod_file_pairs(base_dir, test_dir):
    site_dirs = sorted([os.path.basename(p) for p in glob(
        os.path.join(base_dir, '*')) if os.path.isdir(p)])
    for sdir in site_dirs:
        base_site_dir = os.path.join(base_dir, sdir, 'vertical')
        test_site_dir = os.path.join(
            test_dir, sdir, 'vertical') if test_dir is not None else None

        for fpair in _iter_file_pairs('*.mod', base_site_dir, test_site_dir):
            yield fpair


def iter_vmr_file_pairs(base_dir, test_dir):
    for fpair in _iter_file_pairs('*.vmr', base_dir, test_dir):
        yield fpair


def iter_map_file_pairs(base_dir, test_dir, nc):
    pattern = '*.map.nc' if nc else '*.map'
    for fpair in _iter_file_pairs(pattern, base_dir, test_dir):
        yield fpair


def _iter_file_pairs(pattern, base_dir, test_dir):
    all_base_site_files = sorted(glob(os.path.join(base_dir, pattern)))

    for base_file in all_base_site_files:
        if test_dir is None:
            yield base_file
        else:
            test_file = os.path.join(test_dir, os.path.basename(base_file))
            if not os.path.exists(test_file):
                raise InputDataMismatchError('Could not find a test file corresponding to the base file {}. Was '
                                             'looking for {}.'.format(base_file, test_file))
            else:
                yield base_file, test_file


def iter_file_pairs_by_time(pattern, base_dir, test_dir, test_pattern=None):
    def make_file_dict(file_dir, pat):
        files = sorted(glob(os.path.join(file_dir, pat)))
        return {mod_utils.find_datetime_substring(os.path.basename(f), dt.datetime): f for f in files}

    base_site_files = make_file_dict(base_dir, pattern)
    test_site_files = make_file_dict(
        test_dir, pattern if test_pattern is None else test_pattern)

    for base_date, base_file in base_site_files.items():
        if base_date not in test_site_files:
            raise InputDataMismatchError(
                'Could not find a test file in {} for time {}'.format(test_dir, base_date))
        else:
            yield base_file, test_site_files[base_date]


def parse_args():
    p = ArgumentParser(
        description='Download GEOS test data, check the hashes, or update the hashes')
    subp = p.add_subparsers()

    getp = subp.add_parser('get')
    getp.description = 'Download GEOS FPIT data to run the unit test'
    getp.add_argument('--rebuild', action='store_true', help='Rebuild the hash file once the GEOS FPIT data is '
                                                             'downloaded.')
    getp.set_defaults(driver_fxn=download_test_geos_data)

    checkp = subp.add_parser('check')
    checkp.description = 'Check the hashes of the GEOS FPIT data against those stored'
    checkp.set_defaults(driver_fxn=check_hash_list,
                        hash_list_filename=geos_sha_file)

    updatep = subp.add_parser('update', aliases=['up'])
    updatep.description = 'Update the list of SHA hashes for the necessary GEOS FPIT files'
    updatep.set_defaults(driver_fxn=download_test_geos_data,
                         rebuild_hash_only=True)

    return vars(p.parse_args())


def main():
    cl_args = parse_args()
    driver_fxn = cl_args.pop('driver_fxn')
    driver_fxn(**cl_args)


if __name__ == '__main__':
    main()
