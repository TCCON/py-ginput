from argparse import ArgumentParser
import ctypes
from datetime import datetime, timedelta
from glob import glob
import json
import os
import sys
import time

from ..common_utils import mod_utils, writers
from ..mod_maker import mod_maker
from . import tccon_priors

class MKLThreads(object):
    """
    Limit the number of threads used by C/Fortran backends to numpy functions

    Retrieved from https://gist.github.com/technic/80e8d95858b187cd8ff8677bd5cc0fbb on 2019-11-06. User technic is the
    author.
    """
    _mkl_rt = None

    @classmethod
    def _mkl(cls):
        if cls._mkl_rt is None:
            try:
                cls._mkl_rt = ctypes.CDLL('libmkl_rt.so')
            except OSError:
                cls._mkl_rt = ctypes.CDLL('mkl_rt.dll')
        return cls._mkl_rt
    
    @classmethod
    def get_max_threads(cls):
        return cls._mkl().mkl_get_max_threads()

    @classmethod
    def set_num_threads(cls, n):
        assert type(n) == int
        cls._mkl().mkl_set_num_threads(ctypes.byref(ctypes.c_int(n)))

    def __init__(self, num_threads):
        self._n = num_threads
        self._saved_n = 0

    def __enter__(self):
        if self._n > 0:
            self._saved_n = self.get_max_threads()
            self.set_num_threads(self._n)
        return self

    def __exit__(self, type, value, traceback):
        if self._n > 0:
            self.set_num_threads(self._saved_n)
    

class AutomationArgs:
    def __init__(self, **json_dict):
        self.start_date = datetime.strptime(json_dict['start_date'], "%Y-%m-%d")
        self.end_date = datetime.strptime(json_dict['end_date'], '%Y-%m-%d') if 'end_date' in json_dict else self.start_date + timedelta(days=1)
        self.met_path = json_dict['met_path']
        self.chem_path = json_dict['chem_path']
        self.save_path = json_dict['save_path']
        self.site_ids = json_dict['site_ids']
        self.site_lons = json_dict['site_lons']
        self.site_lats = json_dict['site_lats']
        self.site_alts = json_dict['site_alts']

        self.base_vmr_file = json_dict['base_vmr_file']
        self.zgrid_file = json_dict['zgrid_file']

        self.map_file_format = json_dict['map_file_format'].lower()

        self.n_threads = json_dict.get('n_threads', 4)


def _make_mod_files(all_args: AutomationArgs):
    mod_maker.driver(
        date_range=[all_args.start_date, all_args.end_date],
        met_path=all_args.met_path,
        chem_path=all_args.chem_path,
        save_path=all_args.save_path,
        alt=all_args.site_alts,
        lon=all_args.site_lons,
        lat=all_args.site_lats,
        site_abbrv=all_args.site_ids,
        mode='fpit-eta',
        include_chm=True,
        muted=True
    )

def _make_vmr_files(all_args: AutomationArgs):
    mod_files = [t for t in mod_utils.iter_mod_files(os.path.join(all_args.save_path, 'fpit'))]
    # Cannot use the abbreviations defined in the job arguments anymore - there will be at least 8 files per
    # site, so if multiple sites were requested, we'll have n abbreviations and 8*n*ndays files. The prior
    # functions expect either one abbreviation to use for all files or the same number of abbreviations and
    # files.
    mod_sites = mod_utils.extract_mod_site_abbrevs(mod_files)

    tccon_priors.generate_full_tccon_vmr_file(
        mod_data=mod_files,
        utc_offsets=timedelta(0),
        save_dir=all_args.save_path,
        use_existing_luts=True,
        site_abbrevs=mod_sites,
        flat_outdir=False,
        std_vmr_file=all_args.base_vmr_file,
        zgrid=all_args.zgrid_file
    )

def _make_map_files(all_args: AutomationArgs):
    def make_file_dict(file_list):
        dict_out = dict()
        for f in file_list:
            fbase = os.path.basename(f)
            timestr = mod_utils.find_datetime_substring(fbase)
            lonstr = mod_utils.find_lon_substring(fbase)
            latstr = mod_utils.find_lat_substring(fbase)
            dict_out['{}_{}_{}'.format(timestr, lonstr, latstr)] = f
        return dict_out
    
    job_dir = all_args.save_path
    map_fmt = all_args.map_file_format

    if map_fmt == 'none':
        return
    elif map_fmt == 'text':
        map_fmt = 'txt'
    elif map_fmt == 'netcdf':
        map_fmt = 'nc'
    else:
        raise ValueError('"{}" is not an allowed value for map_fmt.'.format(map_fmt))

    sites = sorted(glob(os.path.join(job_dir, 'fpit', '??')))
    for site_dir in sites:
        site_abbrev = os.path.basename(site_dir.rstrip(os.sep))
        mod_files = glob(os.path.join(site_dir, 'vertical', '*.mod'))
        mod_files = make_file_dict(mod_files)
        vmr_files = glob(os.path.join(site_dir, 'vmrs-vertical', '*.vmr'))
        vmr_files = make_file_dict(vmr_files)
        map_dir = os.path.join(site_dir, 'maps-vertical')
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)

        for key in mod_files.keys():
            modf = mod_files[key]
            vmrf = vmr_files[key]

            writers.write_map_from_vmr_mod(vmr_file=vmrf, mod_file=modf, map_output_dir=map_dir, fmt=map_fmt,
                                           site_abbrev=site_abbrev)
            
def _make_simulated_files(all_args: AutomationArgs, delay_time: float):
    time.sleep(delay_time)
    curr_time = all_args.start_date
    site_ids, site_lats, site_lons, _ = mod_utils.check_site_lat_lon_alt(
        all_args.site_ids, all_args.site_lats, all_args.site_lons, all_args.site_alts
    )
    while curr_time < all_args.end_date:
        for (site_id, lat, lon) in zip(site_ids, site_lats, site_lons):
            # .mod files
            mod_dir = os.path.join(all_args.save_path, 'fpit', site_id, 'vertical')
            if not os.path.exists(mod_dir):
                os.makedirs(mod_dir)
            mod_file_name = mod_utils.mod_file_name_for_priors(curr_time, lat, lon)
            with open(os.path.join(mod_dir, mod_file_name), 'w') as f:
                f.write(f'Simulated .mod file for {curr_time}')

            # .vmr files
            vmr_dir = mod_utils.vmr_output_subdir(all_args.save_path, site_id)
            if not os.path.exists(vmr_dir):
                os.makedirs(vmr_dir)
            vmr_file_name = mod_utils.vmr_file_name(curr_time, lon, lat)
            with open(os.path.join(vmr_dir, vmr_file_name), 'w') as f:
                f.write(f'Simulated .vmr file for {curr_time}')

            if all_args.map_file_format != 'none':
                map_dir = os.path.join(all_args.save_path, 'fpit', site_id, 'maps-vertical')
                if not os.path.exists(map_dir):
                    os.makedirs(map_dir)
                map_file_name = mod_utils.map_file_name_from_mod_vmr_files(
                    site_id, mod_file_name, vmr_file_name, all_args.map_file_format
                )
                with open(os.path.join(map_dir, map_file_name), 'w') as f:
                    f.write(f'Simulated .map file for {curr_time}')
                
            
        curr_time += timedelta(hours=3)

def job_driver(json_file, simulate_with_delay=None):
    if json_file is None:
        json_dict = json.loads(sys.stdin.read())
    else:
        with open(json_file) as f:
            json_dict = json.load(f)

    all_args = AutomationArgs(**json_dict)
    if simulate_with_delay is not None:
        _make_simulated_files(all_args, simulate_with_delay)
    else:
        with MKLThreads(all_args.n_threads):
            _make_mod_files(all_args)
            _make_vmr_files(all_args)
            _make_map_files(all_args)

    
def lut_regen_driver():
    # Have each trace gas record use its internal logic to decide if it needs
    # regenerated
    for record in tccon_priors.gas_records.values():
        record()


def parse_cl_args(p=None):
    if p is None:
        p = ArgumentParser()
        i_am_main = True
    else:
        i_am_main = False

    p.description = 'Entry point for ginput intended for calls from priors automation code'
    subp = p.add_subparsers()
    p_run = subp.add_parser('run', help='Run ginput to generate .mod, .vmr, and (optionally) .map files')
    p_run.add_argument('json_file', help='Path to the JSON file containing the information about what priors to generate')
    p_run.add_argument('-s', '--simulate-with-delay', type=float, help='Simulate running ginput, delaying creating output files by the given number of seconds')
    p_run.set_defaults(driver_fxn=job_driver)

    p_lut = subp.add_parser('regen-lut', help='Regenerate the chemical lookup tables used by "run"')
    p_lut.set_defaults(driver_fxn=lut_regen_driver)

    if i_am_main:
        return vars(p.parse_args())