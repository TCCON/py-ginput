from argparse import ArgumentParser
from netCDF4 import Dataset
import pandas as pd
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

from ..common_utils import mod_utils, readers
from ..common_utils.ggg_logging import logger


TIME_DIM = 'prior_time'
MOD_LEVEL_DIM = 'mod_level'
VMR_LEVEL_DIM = 'vmr_level'
EPOCH = pd.Timestamp(1970,1,1)


def parse_args(p=None):
    if p is None:
        p = ArgumentParser()
    p.description = 'Collate multiple .mod and .vmr files into a netCDF file'
    p.add_argument('--pdb', action='store_true', help='Launch the Python debugger')
    subp = p.add_subparsers()

    patp = subp.add_parser('pattern', help='Deduce tar file names from a curly-brace style pattern and a date range. '
                                           'Useful if you have too many tar files to fit within the allowed number of '
                                           'CLI arguments for your system.')
    patp.add_argument('nc_file', help='Path to the .nc file to write.')
    patp.add_argument('pattern', help='A pattern describing the paths to the tar files, using Python curly brace formatting syntax. '
                                      '{date} will be replaced by the expected date/time of the file and can be formatted with strftime '
                                      'specifiers.')
    patp.add_argument('start_date', help='The first date/time to process, in a string format Pandas understands natively.')
    patp.add_argument('end_date', help='The last date/time to process, in a string format Pandas understands natively.')
    patp.add_argument('--tmp-dir-parent', default='.', help='Directory in which to create the temporary directory where the '
                                                            'tar files will be extracted to. Default is the current directory.')
    patp.set_defaults(driver_fxn=tar_from_pattern)

    filep = subp.add_parser('files', help='Give the list of tar files explicitly')
    filep.add_argument('nc_file', help='Path to the .nc file to write.')
    filep.add_argument('tar_files', nargs='+', help='The list of tar files to collate into the netCDF file')
    filep.add_argument('--tmp-dir-parent', default='.', help='Directory in which to create the temporary directory where the '
                                                             'tar files will be extracted to. Default is the current directory.')
    filep.set_defaults(driver_fxn=tar2nc)



def tar_from_pattern(pattern, start_date, end_date, nc_file, tmp_dir_parent='.', pdb=False):
    if pdb:
        import pdb
        pdb.set_trace()
    tar_files = []
    nmissing = 0
    for date in pd.date_range(start_date, end_date, freq='1d'):
        file = Path(pattern.format(date=date))
        if not file.exists():
            logger.debug(f'File not found: {file}')
            nmissing += 1
        else:
            tar_files.append(file)

    if nmissing > 0:
        logger.warn(f'{nmissing} tar files were not found')
    if len(tar_files) > 0:
        tar2nc(tar_files, nc_file, tmp_dir_parent=tmp_dir_parent)


def tar2nc(tar_files, nc_file, tmp_dir_parent='.', pdb=False):
    if pdb:
        import pdb
        pdb.set_trace()

    with TemporaryDirectory(dir=tmp_dir_parent) as tdir:
        logger.info(f'Extracting {len(tar_files)} TAR files into {tdir}')
        prior_files_df = extract_tarballs(tdir, tar_files)
        prior_files_to_nc(prior_files_df, nc_file, delete_files=True)


def prior_files_to_nc(prior_files_df, nc_file, delete_files=False):
    nfiles = prior_files_df.shape[0]
    logger.info(f'Copying {nfiles} .mod & .vmr files to {nc_file}')
    with Dataset(nc_file, 'w') as ds:
        first = True
        for i, (time, row) in enumerate(prior_files_df.iterrows()):
            if i % 25 == 0:
                logger.info(f'{i+1} of {nfiles} times copied')
            mod_data = readers.read_mod_file(row['mod_file'])
            mod_prof_units = readers.read_mod_file_units(row['mod_file'])
            vmr_data = readers.read_vmr_file(row['vmr_file'])

            if first:
                init_variables(ds, mod_data, mod_prof_units, vmr_data, ntimes=nfiles)
                first = False

            populate_variables(
                ds, itime=i, time=time,
                mod_file_name=Path(row['mod_file']).name, vmr_file_name=Path(row['vmr_file']).name,
                mod_data=mod_data, mod_prof_units=mod_prof_units, vmr_data=vmr_data
            )

            if delete_files:
                # Delete files as we go, rather than relying on the temporary directory
                # at the end - cleaning up 10,000+ files at once is slow.
                Path(row['mod_file']).unlink()
                Path(row['vmr_file']).unlink()
        logger.info(f'Finished copying data to {nc_file}')


def extract_tarballs(tempdir, tar_files):
    nfiles = len(tar_files)
    for ifile, tar_file in enumerate(tar_files):
        if ifile % 25 == 0:
            logger.info(f'{ifile+1} of {nfiles} tar files extracted')
        with tarfile.open(tar_file) as tf:
            tf.extractall(path=tempdir)
    logger.info('Finished extracting tar files')
    mod_files = sorted(Path(tempdir).glob('*.mod'))
    mod_files = {mod_utils.datetime_from_mod_name(f.name): str(f) for f in mod_files}
    vmr_files = sorted(Path(tempdir).glob('*.vmr'))
    vmr_files = {mod_utils.datetime_from_vmr_name(f.name): str(f) for f in vmr_files}
    return pd.DataFrame({'mod_file': mod_files, 'vmr_file': vmr_files})


def init_variables(ds, mod_data, mod_prof_units, vmr_data, ntimes):
    ds.createDimension(TIME_DIM, ntimes)
    ds.createDimension(MOD_LEVEL_DIM, mod_data['profile']['Pressure'].size)
    ds.createDimension(VMR_LEVEL_DIM, vmr_data['profile']['altitude'].size)

    ds.createVariable(TIME_DIM, int, (TIME_DIM,))
    ds[TIME_DIM].units = 'seconds since 1970-01-01 00:00:00'

    ds.createVariable('mod_file_name', str, (TIME_DIM,))
    ds.createVariable('vmr_file_name', str, (TIME_DIM,))

    # Might be able to get rid of this and fold everything into populate_variables
    # Probably should profile the timing to see how slow checking for the existance
    # of variables is first.
    for k in mod_data['scalar'].keys():
        ds.createVariable(f'mod_scalar_{k.lower()}', float, (TIME_DIM,))
    for k in mod_data['profile'].keys():
        vname = f'mod_{k.lower()}'
        ds.createVariable(vname, float, (TIME_DIM, MOD_LEVEL_DIM))
        ds[vname].units = mod_prof_units[k]
    for k in vmr_data['scalar'].keys():
        if k not in {'ginput_version'}: # skip strings
            ds.createVariable(f'vmr_scalar_{k.lower()}', float, (TIME_DIM,))
    for k in vmr_data['profile'].keys():
        vname = f'vmr_{k.lower()}'
        ds.createVariable(vname, float, (TIME_DIM, VMR_LEVEL_DIM))
        if k.lower() == 'altitude':
            ds[vname].units = 'km'
        else:
            ds[vname].units = 'mol/mol'
            ds[vname].description = f'dry mole fraction of {k}'


def populate_variables(ds, itime, time, mod_file_name, vmr_file_name, mod_data, mod_prof_units, vmr_data):
    ds[TIME_DIM][itime] = (time - EPOCH).total_seconds()
    ds['mod_file_name'][itime] = mod_file_name
    ds['vmr_file_name'][itime] = vmr_file_name

    for k, v in mod_data['scalar'].items():
        var = _get_or_init_var(ds, varname=f'mod_scalar_{k.lower()}', dtype=_guess_dtype(v), dims=(TIME_DIM,))
        var[itime] = v
    for k, arr in mod_data['profile'].items():
        var = _get_or_init_var(ds, varname=f'mod_{k.lower()}', dtype=float, dims=(TIME_DIM, MOD_LEVEL_DIM))
        var[itime] = arr
    for k, v in vmr_data['scalar'].items():
        var = _get_or_init_var(ds, varname=f'vmr_scalar_{k.lower()}', dtype=_guess_dtype(v), dims=(TIME_DIM,))
        var[itime] = v
    for k, arr in vmr_data['profile'].items():
        var = _get_or_init_var(ds, varname=f'vmr_{k.lower()}', dtype=float, dims=(TIME_DIM, VMR_LEVEL_DIM))
        var[itime] = arr


def _get_or_init_var(ds, varname, dtype, dims):
    if varname in ds.variables.keys():
        return ds[varname]
    else:
        return ds.createVariable(varname, dtype, dims)

def _guess_dtype(val):
    return str if isinstance(val, str) else float
