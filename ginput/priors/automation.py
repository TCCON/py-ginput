from argparse import ArgumentParser
import ctypes
from datetime import datetime, timedelta
from glob import glob
import json
from pathlib import Path
import os
import re
import sys
import time

from ..common_utils import mod_utils, writers
from ..common_utils.ggg_logging import logger
from ..mod_maker import mod_maker
from . import tccon_priors, fo2_prep

class MKLThreads(object):
    """
    Limit the number of threads used by C/Fortran backends to numpy functions

    Retrieved from https://gist.github.com/technic/80e8d95858b187cd8ff8677bd5cc0fbb on 2019-11-06. User technic is the
    author of the original. Then modified based on https://stackoverflow.com/a/29582987 on 2023-11-14..
    """
    _thread_rt = None
    _thread_type = None

    @classmethod
    def _threads(cls):
        if cls._thread_rt is None:
            for lib, lib_type in [('libmkl_rt.so', 'MKL'), ('mkl_rt.dll', 'MKL'), ('libopenblas.so', 'BLAS')]:
                try:
                    cls._thread_rt = ctypes.CDLL(lib)
                    cls._thread_type = lib_type
                    logger.info(f'Will limit threads on library {lib}')
                    break
                except OSError:
                    logger.info(f'Tried to limit threads for library {lib}, library not present')
        if cls._thread_rt is None:
            raise OSError('Could not load any of the expected multithreading libraries')
        return cls._thread_rt, cls._thread_type
    
    @classmethod
    def get_num_threads(cls):
        rt, tt = cls._threads()
        if tt == 'MKL':
            # TODO: replace this with the correct function to get current number of threads for MKL
            return rt.mkl_get_max_threads()
        elif tt == 'BLAS':
            return rt.openblas_get_num_threads()
        else:
            raise NotImplementedError(f'library type {tt}')

    @classmethod
    def set_num_threads(cls, n):
        assert type(n) == int
        rt, tt = cls._threads()
        if tt == 'MKL':
            rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(n)))
        elif tt == 'BLAS':
            rt.openblas_set_num_threads(n)
        else:
            raise NotImplementedError(f'library type {tt}')

    def __init__(self, num_threads):
        self._n = num_threads
        self._saved_n = 0

    def __enter__(self):
        if self._n > 0:
            try:
                self._saved_n = self.get_num_threads()
                self.set_num_threads(self._n)
            except OSError:
                logger.warning('Could not set number of MKL/BLAS threads, numpy of numpy threads will not be limited')
                self._n = 0
        return self

    def __exit__(self, type, value, traceback):
        if self._n > 0:
            self.set_num_threads(self._saved_n)
    

class AutomationArgs:
    def __init__(self, **json_dict):
        self.ginput_met_key = json_dict['ginput_met_key']
        self.start_date = self._parse_datestr(json_dict['start_date'])
        self.end_date = self._parse_datestr(json_dict['end_date']) if json_dict.get('end_date') is not None else self.start_date + timedelta(days=1)
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

    @staticmethod
    def _parse_datestr(s):
        if len(s) == 10:
            return datetime.strptime(s, '%Y-%m-%d')
        elif len(s) == 19:
            return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        else:
            raise ValueError(f'Unrecognized datetime format: {s}')


def _make_mod_files(all_args: AutomationArgs, force_file_name_fpit: bool = True):
    mod_maker.driver(
        date_range=[all_args.start_date, all_args.end_date],
        met_path=all_args.met_path,
        chem_path=all_args.chem_path,
        save_path=all_args.save_path,
        alt=all_args.site_alts,
        lon=all_args.site_lons,
        lat=all_args.site_lats,
        site_abbrv=all_args.site_ids,
        mode=all_args.ginput_met_key,
        include_chm=True,
        muted=True
    )

    if force_file_name_fpit:
        logger.info("Renaming any IT files to start with FPIT to work with gsetup")
        subdir = mod_utils.mode_to_product(all_args.ginput_met_key)
        for mod_file in mod_utils.iter_mod_files(os.path.join(all_args.save_path, subdir)):
            mod_file = Path(mod_file)
            new_name = re.sub('^IT', 'FPIT', mod_file.name)
            if new_name != mod_file.name:
                new_file = mod_file.parent / new_name
                logger.debug(f'Renaming {mod_file} to {new_file}')
                mod_file.rename(new_file)

def _make_vmr_files(all_args: AutomationArgs):
    subdir = mod_utils.mode_to_product(all_args.ginput_met_key)
    mod_files = [t for t in mod_utils.iter_mod_files(os.path.join(all_args.save_path, subdir))]
    # Cannot use the abbreviations defined in the job arguments anymore - there will be at least 8 files per
    # site, so if multiple sites were requested, we'll have n abbreviations and 8*n*ndays files. The prior
    # functions expect either one abbreviation to use for all files or the same number of abbreviations and
    # files.
    mod_sites = mod_utils.extract_mod_site_abbrevs(mod_files)

    tccon_priors.generate_full_tccon_vmr_file(
        mod_data=mod_files,
        utc_offsets=timedelta(0),
        save_dir=all_args.save_path,
        product=subdir,
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
    elif map_fmt != 'txtandnc':
        raise ValueError('"{}" is not an allowed value for map_fmt.'.format(map_fmt))

    subdir = mod_utils.mode_to_product(all_args.ginput_met_key)
    sites = sorted(glob(os.path.join(job_dir, subdir, '??')))
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

            if map_fmt == 'txtandnc':
                writers.write_map_from_vmr_mod(
                    vmr_file=vmrf, mod_file=modf, map_output_dir=map_dir, 
                    fmt='txt', site_abbrev=site_abbrev
                )

                writers.write_map_from_vmr_mod(
                    vmr_file=vmrf, mod_file=modf, map_output_dir=map_dir, 
                    fmt='nc', site_abbrev=site_abbrev, no_cfunits=True,
                )
            else:
                writers.write_map_from_vmr_mod(
                    vmr_file=vmrf, mod_file=modf, map_output_dir=map_dir, 
                    fmt=map_fmt, site_abbrev=site_abbrev, no_cfunits=True
                )
            
def _make_simulated_files(all_args: AutomationArgs, delay_time: float):
    time.sleep(delay_time)
    curr_time = all_args.start_date
    subdir = mod_utils.mode_to_product(all_args.ginput_met_key)
    site_ids, site_lats, site_lons, _ = mod_utils.check_site_lat_lon_alt(
        all_args.site_ids, all_args.site_lats, all_args.site_lons, all_args.site_alts
    )
    while curr_time < all_args.end_date:
        for (site_id, lat, lon) in zip(site_ids, site_lats, site_lons):
            # .mod files
            mod_dir = os.path.join(all_args.save_path, subdir, site_id, 'vertical')
            if not os.path.exists(mod_dir):
                os.makedirs(mod_dir)
            mod_file_name = mod_utils.mod_file_name_for_priors(curr_time, lat, lon)
            with open(os.path.join(mod_dir, mod_file_name), 'w') as f:
                f.write(f'Simulated .mod file for {curr_time}')

            # .vmr files
            vmr_dir = mod_utils.vmr_output_subdir(all_args.save_path, site_id, product=subdir)
            if not os.path.exists(vmr_dir):
                os.makedirs(vmr_dir)
            vmr_file_name = mod_utils.vmr_file_name(curr_time, lon, lat)
            with open(os.path.join(vmr_dir, vmr_file_name), 'w') as f:
                f.write(f'Simulated .vmr file for {curr_time}')

            if all_args.map_file_format != 'none':
                map_dir = os.path.join(all_args.save_path, subdir, site_id, 'maps-vertical')
                if not os.path.exists(map_dir):
                    os.makedirs(map_dir)
                    
                if all_args.map_file_format == 'txtandnc':
                    map_file_name = mod_utils.map_file_name_from_mod_vmr_files(
                        site_id, mod_file_name, vmr_file_name, 'txt'
                    )
                    with open(os.path.join(map_dir, map_file_name), 'w') as f:
                        f.write(f'Simulated .map file for {curr_time}')

                    map_file_name = mod_utils.map_file_name_from_mod_vmr_files(
                        site_id, mod_file_name, vmr_file_name, 'nc'
                    )
                    with open(os.path.join(map_dir, map_file_name), 'w') as f:
                        f.write(f'Simulated .map file for {curr_time}')
                else:
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
    # Update the O2 dry mole fraction table first - the LUTs shouldn't depend on this,
    # but it's quick, so may as well do it first.
    fo2_prep.fo2_update_driver()

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
