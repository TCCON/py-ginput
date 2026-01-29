from datetime import datetime, timedelta
from functools import partial
from glob import glob
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
from pathlib import Path
import pytest
import re

from ginput.common_utils import mod_utils, readers
from ginput.mod_maker.mod_maker import driver as mmdriver
from ginput.priors import tccon_priors, map_maker

MOD_TO_NCMAP = {'profile': {'Height': 'altitude', 'Temperature': 'temp', 'Pressure': 'pressure'}}
MOD_TO_TXTMAP = {'profile': {'Height': 'Height', 'Temperature': 'Temp', 'Pressure': 'Pressure'}}
VMR_TO_MAP = {'profile': {'h2o': 'h2o', 'hdo': 'hdo', 'co2': 'co2', 'n2o': 'n2o', 'co': 'co', 'ch4': 'ch4', 'hf': 'hf', 'o2': 'o2'}}
MAP_UNIT_SCALES = {'mol/mol': 1.0, 'parts': 1.0, 'ppm': 1e-6, 'ppb': 1e-9, 'ppt': 1e-12}
DATE_RANGE_2018 = [datetime(2018, 1, 1), datetime(2018, 1, 2)]
DATE_RANGE_2025 = [datetime(2025, 3, 2, 15), datetime(2025, 3, 3)]

@pytest.mark.slow
def test_mod_files_jan2018(subtests, mod_input_dir, mod_output_dir, test_plots_dir, generate_files_with_defaults):
    """This test confirms that .mod files for Lamont on 1 Jan 2018 are the same as previously produced.
    """
    # generate_files_with_defaults is needed to ensure the output files are created - it's
    # a setup fixture.
    comparison_helper(
        subtests,
        partial(iter_mod_file_pairs, date_range=DATE_RANGE_2018),
        mod_input_dir / 'fpit',
        mod_output_dir / 'fpit',
        plots_dir=test_plots_dir
    )


@pytest.mark.slow
def test_vmr_files_jan2018(subtests, vmr_input_dir, vmr_output_dir, test_plots_dir, generate_files_with_defaults):
    """This test confirms that .vmr files for Lamont on 1 Jan 2018 are the same as previously produced.

    It uses freshly generated .mod files as input, so changes to the .mod files that affect the .vmr files will
    also cause this test to fail.
    """
    # generate_files_with_defaults is needed to ensure the output files are created - it's
    # a setup fixture.
    comparison_helper(
        subtests,
        partial(iter_vmr_file_pairs, date_range=DATE_RANGE_2018),
        vmr_input_dir / 'fpit', 
        vmr_output_dir / 'fpit',
        plots_dir=test_plots_dir
    )


@pytest.mark.slow
def test_map_files_jan2018(subtests, map_input_dir, map_output_dir, test_plots_dir, generate_files_with_defaults):
    """This test confirms that .map and .map.nc files for Lamont on 1 Jan 2018 are the same as previously produced.

    It uses freshly generated .mod and .vmr files as input, so changes to either of those files will
    also cause this test to fail. Note that this only tests the wet .map/.map.nc files.
    """
    # generate_files_with_defaults is needed to ensure the output files are created - it's
    # a setup fixture.
    comparison_helper(
        subtests,
        partial(iter_map_file_pairs, nc=True, date_range=DATE_RANGE_2018),
        map_input_dir / 'fpit',
        map_output_dir / 'fpit',
        plots_dir=test_plots_dir
    )

    comparison_helper(
        subtests,
        partial(iter_map_file_pairs, nc=False, date_range=DATE_RANGE_2018),
        map_input_dir / 'fpit',
        map_output_dir / 'fpit',
        plots_dir=test_plots_dir
    )

@pytest.mark.slow
def test_mod_files_mar2025(subtests, mod_input_dir, mod_output_dir, test_plots_dir, generate_files_smo_gap):
    """This test confirms that a few .mod files for Lamont on 2 Mar 2025 are the same as previously produced.

    This date was chosen since it falls at the beginning of the time period affected by the May 2024 to Jan 2025
    SMO data gap for OCO-2 & OCO-3. When the Feb 2025 data was added, that gap changed from being filled by extrapolation
    to interpolation. Before ginput v1.6.0, interpolation was always done as linear, which meant that the drawdown during
    summer 2024 went from being estimated to ignored, causing a step change in the prior CO2. This test is to confirm
    that the current gap filling strategy has not changed the resulting output.

    While the .mod files should be unaffected, they are needed for the .vmr files. By including this test, we
    have a way to check that changes in the Mar 2025 .vmr files is or is not due to changes in the .mod files.
    """
    # generate_files_smo_gap is needed to ensure the output files are created - it's
    # a setup fixture.
    comparison_helper(
        subtests,
        partial(iter_mod_file_pairs, date_range=DATE_RANGE_2025),
        mod_input_dir / 'it',
        mod_output_dir / 'it',
        plots_dir=test_plots_dir
    )


@pytest.mark.slow
def test_vmr_files_mar2025(subtests, vmr_input_dir, vmr_output_dir, test_plots_dir, generate_files_smo_gap):
    """This test confirms that a few .mod files for Lamont on 2 Mar 2025 are the same as previously produced.

    See :func:`test_mod_files_mar2025` for the rationale for testing this date.
    """
    # generate_files_smo_gap is needed to ensure the output files are created - it's
    # a setup fixture.
    comparison_helper(
        subtests,
        partial(iter_vmr_file_pairs, date_range=DATE_RANGE_2025),
        vmr_input_dir / 'it', 
        vmr_output_dir / 'it',
        plots_dir=test_plots_dir
    )

def test_nc_map_files_against_upstream(
    subtests,
    test_site,
    mod_input_dir,
    vmr_input_dir,
    map_dry_output_dir,
    test_plots_dir,
    generate_dry_map_files
):
    """This test generates dry VMR .map and .map.nc files from the benchmark .mod and .vmr files,
    and compares the .map files' contents to the .mod and .vmr files. Compared to
    :func:`test_map_files_jan2018`, this is checking that the .map.nc files correctly copy
    the .mod and .vmr files' contents, whereas :func:`test_map_files_jan2018` is testing that the
    .map.nc files have not changed from the benchmarks.

    Using dry VMRs here simplifies the comparison to the .vmr files (also dry) and complements the
    wet VMR files tested in :func:`test_map_files_jan2018`.
    """
    mod_iterator = iter_file_pairs_by_time(
        pattern='*.mod',
        test_pattern='*.map.nc',
        base_dir=mod_input_dir / 'fpit' / test_site / 'vertical',
        test_dir=map_dry_output_dir / 'fpit',
        date_range=DATE_RANGE_2018,
    )
    for mod_file, map_file in mod_iterator:
        compare_two_files(
            subtests=subtests,
            check_file=mod_file,
            new_file=map_file,
            plots_dir=test_plots_dir,
            variable_mapping=MOD_TO_NCMAP
        )

    vmr_iterator = iter_file_pairs_by_time(
        pattern='*.vmr',
        test_pattern='*.map.nc',
        base_dir=vmr_input_dir / 'fpit',
        test_dir=map_dry_output_dir / 'fpit',
        date_range=DATE_RANGE_2018,
    )
    for vmr_file, map_file in vmr_iterator:
        vmr_scales = _get_nc_unit_scales(map_file)
        compare_two_files(
            subtests=subtests,
            check_file=vmr_file,
            new_file=map_file,
            plots_dir=test_plots_dir,
            variable_mapping=VMR_TO_MAP,
            variable_scaling=vmr_scales
        )


def test_txt_map_files_against_upstream(
    subtests,
    test_site,
    mod_input_dir,
    vmr_input_dir,
    map_dry_output_dir,
    test_plots_dir,
    generate_dry_map_files
):
    """Like :func:`test_nc_map_files_against_upstream`, but for the text (.map) files.
    """
    mod_iterator = iter_file_pairs_by_time(
        pattern='*.mod',
        test_pattern='*.map',
        base_dir=mod_input_dir / 'fpit' / test_site / 'vertical',
        test_dir=map_dry_output_dir / 'fpit',
        date_range=DATE_RANGE_2018,
    )
    for mod_file, map_file in mod_iterator:
        compare_two_files(
            subtests=subtests,
            check_file=mod_file,
            new_file=map_file,
            plots_dir=test_plots_dir,
            variable_mapping=MOD_TO_TXTMAP
        )

    vmr_iterator = iter_file_pairs_by_time(
        pattern='*.vmr',
        test_pattern='*.map',
        base_dir=vmr_input_dir / 'fpit',
        test_dir=map_dry_output_dir / 'fpit',
        date_range=DATE_RANGE_2018,
    )
    for vmr_file, map_file in vmr_iterator:
        vmr_scales = _get_map_unit_scales(map_file)
        compare_two_files(
            subtests=subtests,
            check_file=vmr_file,
            new_file=map_file,
            plots_dir=test_plots_dir,
            variable_mapping=VMR_TO_MAP,
            variable_scaling=vmr_scales
        )


def comparison_helper(subtests, iter_fxn, input_dir, output_dir, plots_dir):
    for check_file, new_file in iter_fxn(input_dir, output_dir):
        compare_two_files(subtests=subtests, check_file=check_file, new_file=new_file, plots_dir=plots_dir)


@pytest.fixture(scope='session')
def generate_files_with_defaults(
    geos_dir,
    mod_output_dir,
    vmr_output_dir,
    map_output_dir,
    test_site,
    std_vmr_file,
):
    generate_files(
        geos_dir,
        mod_output_dir,
        vmr_output_dir,
        map_output_dir,
        test_site,
        std_vmr_file,
        mode='fpit-eta',
        date_range=DATE_RANGE_2018
    )


@pytest.fixture(scope='session')
def generate_files_smo_gap(
    geos_dir,
    mod_output_dir,
    vmr_output_dir,
    map_output_dir,
    test_site,
    std_vmr_file,
    smo_real_gap_input_dir,
    smo_real_gap_o2_file,
):
    mlo_smo_files = {
        'co2': {
            'mlo_file': smo_real_gap_input_dir / 'co2_mlo_monthly.txt',
            'smo_file': smo_real_gap_input_dir / 'co2_smo_monthly.txt',
        }
    }

    generate_files(
        geos_dir,
        mod_output_dir,
        vmr_output_dir,
        map_output_dir,
        test_site,
        std_vmr_file,
        mode='it-eta',
        date_range=DATE_RANGE_2025,
        vmr_kwargs={'mlo_smo_files': mlo_smo_files, 'o2_mole_fraction_file': smo_real_gap_o2_file}
    )

def generate_files(
    geos_dir,
    mod_output_dir,
    vmr_output_dir,
    map_output_dir,
    test_site,
    std_vmr_file,
    date_range,
    mode,
    vmr_kwargs=dict(),
):
    # Assume that mode will be, e.g., "fpit-eta", so the product will be the part before the dash
    product = mode.split('-')[0]

    # Ensure the output subdirectories exists - needed for the .vmr files at least
    mod_output_subdir = mod_output_dir / product
    vmr_output_subdir = vmr_output_dir / product
    map_output_subdir = map_output_dir / product
    mod_output_subdir.mkdir(exist_ok=True)
    vmr_output_subdir.mkdir(exist_ok=True)
    map_output_subdir.mkdir(exist_ok=True)

    # Clean up any old output files
    _clean_up_files_recursive(mod_output_subdir, r'.*\.mod$')
    _clean_up_files_recursive(vmr_output_subdir, r'.*\.vmr$')
    _clean_up_files_recursive(map_output_subdir, r'.*\.map(.nc)?$')

    mmdriver(
        date_range,
        geos_dir / 'met',
        save_path=mod_output_subdir.parent,
        keep_latlon_prec=False,
        save_in_utc=True,
        site_abbrv=test_site,
        include_chm=True,
        chem_path=geos_dir / 'chm',
        mode=mode
    )


    mod_files = [f for f in iter_mod_file_pairs(mod_output_subdir, None, date_range)]
    tccon_priors.generate_full_tccon_vmr_file(
        mod_files,
        timedelta(hours=0),
        save_dir=vmr_output_subdir,
        std_vmr_file=std_vmr_file,
        site_abbrevs=test_site,
        **vmr_kwargs
    )

    mod_dir = Path(mod_files[0]).parent
    if not map_output_subdir.is_dir():
        map_output_subdir.mkdir()

    common_map_args = dict(
        date_range=date_range,
        mod_dir=mod_dir,
        vmr_dir=vmr_output_subdir,
        save_dir=map_output_subdir,
        dry=False,
        site_abbrev=test_site,
        product=product,
    )
    map_maker.cl_driver(map_fmt='nc', **common_map_args)
    map_maker.cl_driver(map_fmt='txt', **common_map_args)


@pytest.fixture(scope='session')
def generate_dry_map_files(
    test_site,
    mod_input_dir,
    vmr_input_dir,
    map_dry_output_dir,
):
    map_dry_output_subdir = map_dry_output_dir / 'fpit'
    if not map_dry_output_subdir.exists():
        map_dry_output_subdir.mkdir()
    else:
        _clean_up_files(os.path.join(str(map_dry_output_subdir), '*.map*'))

    date_range = DATE_RANGE_2018
    common_opts = dict(
        date_range = date_range,
        mod_dir=mod_input_dir / 'fpit' / test_site / 'vertical',
        vmr_dir=vmr_input_dir / 'fpit',
        save_dir=map_dry_output_subdir,
        dry=True,
        site_abbrev=test_site
    )
    map_maker.cl_driver(map_fmt='nc', **common_opts)
    map_maker.cl_driver(map_fmt='txt', **common_opts)


def compare_two_files(subtests, check_file, new_file, plots_dir, variable_mapping=None, variable_scaling=None):
    """Check that the data contained in two files are identical

    Parameters
    ----------
    subtests
        The subtests fixture.

    check_file : str
        Path to the "true" file, i.e. what the new file must match

    new_file : str
        Path to the newly generated file

    variable_mapping : Optional[dict]
        A dictionary mapping variables in the check file to variables in the new file. The top level must have the
        same categories as the check data and the values must be dictionaries where the key is the check variable
        and the value is the new variable.

    variable_scaling : Optional[dict]
        A dictionary providing factors to multiply the new file's data by to put it in the same units as the old
        file. It must have the same structure as dictionaries read in from the files and `variable_mapping` (i.e.
        category then variable names as keys) and the values of the inner dictionary will be the factor to multiply
        that variable in the new file by.

    Returns
    -------
    bool
        `True` if the two files are the same, `False` otherwise
    """

    # Define absolute tolerances for some variables, keyed off of the check data variable names
    atol_overrides = {
        # A tenth of a degree in equivalent latitude should be trivial, and will avoid issues with
        # small changes near the surface that don't matter anyway (because we only use EqL in the stratosphere).
        'EqL': 0.1  
    }


    # will be set to `False` if any of the subtests fail
    files_match = True
    check_data = read_file(check_file)
    new_data = read_file(new_file)

    if variable_mapping is None:
        variable_mapping = {cat: {k: k for k in d.keys()} for cat, d in check_data.items()}
    if variable_scaling is None:
        # We don't scale if a category or variable is missing, so we can just create an empty dict to indicate
        # not scaling. This is better than creating a dict with all 1s because some of the data can't be multiplied
        # by floats.
        variable_scaling = dict()

    for category_name, category_data in check_data.items():
        if category_name not in variable_mapping:
            continue

        for variable_name, variable_data in category_data.items():
            with subtests.test(check_file=check_file, new_file=new_file, category=category_name, variable=variable_name):
                if variable_name not in variable_mapping[category_name]:
                    continue
                elif variable_name.lower() == 'ginput_version':
                    print('Ignoring GINPUT_VERSION in header for sake of testing')
                    continue

                new_var = variable_mapping[category_name][variable_name]
                this_new_data = new_data[category_name][new_var]
                try:
                    # This will also affect the data in new_data, which is what we want so that the plotting
                    # is correct
                    this_new_data *= variable_scaling[category_name][new_var]
                except KeyError:
                    # Variable does not exist in the scaling dict - so don't scale it
                    pass

                this_atol = atol_overrides.get(variable_name)
                test_result = _test_single_variable(variable_data, this_new_data, atol=this_atol)
                if not test_result:
                    files_match = False
                    try:
                        _plot_helper(check_data=check_data, new_data=new_data, category=category_name,
                                     variable=variable_name, new_file=new_file, plots_dir=plots_dir, new_variable=new_var)
                    except Exception as err:
                        print('Could not generate plot for {} {}, error was {}'.format(new_file, variable_name,
                                                                                       err.args[0]))
                        print(new_data[category_name].keys())
                        print(check_data[category_name].keys())

                msg = f'"{category_name}/{variable_name}" in {new_file} does not match the check data'
                assert test_result, msg

    return files_match


def _test_single_variable(variable_data, this_new_data, atol=None):
    try:
        if atol is None:
            # We need some absolute tolerance, otherwise inconsequential differences cause the test to fail. E.g. as N2O and
            # CH4 go to zero, a difference of 1e-13 parts triggers a failure, which really doesn't matter. We'll make the
            # absolute tolerance equal to 0.01% of the maximum value in the original data, because a 0.01% difference in
            # the prior concentration really shouldn't matter.
            #
            # Some variables will specify an absolute tolerance that better reflects the needed precision in their values.
            atol = 1e-4 * np.abs(np.nanmax(variable_data))
        return np.isclose(variable_data, this_new_data, atol=atol).all()
    except TypeError:
        # Not all variables with be float arrays. If np.isclose() can't coerce the data to a numeric
        # type, it'll raise a TypeError and we fall back on the equality test. nanmax will also throw a TypeError
        # in similar circumstances
        return np.all(variable_data == this_new_data)


def _plot_helper(check_data, new_data, category, variable, new_file, plots_dir, new_variable=None):
    prefix = 'ncmap' if new_file.endswith('.nc') else Path(new_file).suffix.lstrip('.')
    datestr = mod_utils.find_datetime_substring(new_file)
    savename = '{pre}_{var}_{date}Z.pdf'.format(pre=prefix, var=variable, date=datestr)
    savename = os.path.join(plots_dir, savename)
    _plot_failed_test(check_data=check_data, new_data=new_data, category=category, variable=variable,
                      save_as=savename, new_variable=new_variable)


def _plot_failed_test(check_data, new_data, category, variable, save_as=None, new_variable=None):
    def plotting_internal(axs, oldz, newz, oldx, newx, ypres):
        axs[0].plot(oldx, oldz, marker='+', label='Original')
        axs[0].plot(newx, newz, marker='x', linestyle='--', label='New')
        axs[0].legend()
        axs[0].set_xlabel(variable)

        if ypres:
            axs[0].set_ylabel('Pressure (hPa)')
            axs[0].set_yscale('log')
            axs[0].invert_yaxis()
        else:
            axs[0].set_ylabel('Altitude (km)')

        if ypres:
            new_on_old = mod_utils.mod_interpolation_new(oldz, np.flipud(newz), np.flipud(newx), interp_mode='log-log')
            old_on_new = mod_utils.mod_interpolation_new(newz, np.flipud(oldz), np.flipud(oldx), interp_mode='log-log')
        else:
            new_on_old = mod_utils.mod_interpolation_new(oldz, newz, newx, interp_mode='linear')
            old_on_new = mod_utils.mod_interpolation_new(newz, oldz, oldx, interp_mode='linear')

        axs[1].plot(new_on_old - oldx, oldz, marker='+', label='On old z')
        axs[1].plot(newx - old_on_new, newz, marker='x', linestyle='--', label='On new z')
        axs[1].legend()
        axs[1].set_xlabel(r'$\Delta$ {}'.format(variable))

        # Shared y-axes will both be formatted together
        if ypres:
            axs[1].set_yscale('log')
            axs[1].invert_yaxis()

    if category != 'profile':
        return
    if new_variable is None:
        new_variable = variable

    zvar = 'Height' if 'Height' in check_data[category] else 'altitude'
    new_zvar = 'Height' if 'Height' in new_data[category] else 'altitude'
    oldalt = check_data[category][zvar]
    newalt = new_data[category][new_zvar]
    oldval = check_data[category][variable]
    newval = new_data[category][new_variable]

    try:
        oldpres = check_data[category]['Pressure']
        newpres = new_data[category]['Pressure']
    except KeyError:
        oldpres, newpres = None, None
        include_pres = False
        ny = 1
    else:
        include_pres = True
        ny = 2

    fig, all_axs = plt.subplots(ny, 2)
    if include_pres:
        plotting_internal(all_axs[0], oldalt, newalt, oldval, newval, False)
        plotting_internal(all_axs[1], oldpres, newpres, oldval, newval, True)
    else:
        plotting_internal(all_axs, oldalt, newalt, oldval, newval, False)

    fig.set_size_inches(12, 6*ny)

    if save_as:
        plt.savefig(save_as)
        plt.close(fig)

def read_file(filename):
    """Read a .mod or .vmr file

    Parameters
    ----------
    filename : str
        The path to the file to read

    Returns
    -------
    dict
        Results of reading the file
    """
    if filename.endswith('.mod'):
        return readers.read_mod_file(filename)
    elif filename.endswith('.vmr'):
        return readers.read_vmr_file(filename)
    elif filename.endswith('.map') or filename.endswith('.map.nc'):
        return readers.read_map_file(filename, skip_header=True)
    else:
        ext = os.path.splitext(filename)
        raise NotImplementedError('Do not know how to read a "{}" file'.format(ext))


def _clean_up_files_recursive(top_dir, pattern):
    for dirname, _, files in os.walk(top_dir):
        for f in files:
            if re.match(pattern, f):
                fullfile = os.path.join(dirname, f)
                print('Removing', fullfile)
                os.remove(fullfile)

def _clean_up_files(pattern):
    files = sorted(glob(pattern))
    for f in files:
        print('Removing', f)
        os.remove(f)


def iter_mod_file_pairs(base_dir, test_dir, date_range):
    site_dirs = sorted([os.path.basename(p) for p in glob(
        os.path.join(base_dir, '*')) if os.path.isdir(p)])
    for sdir in site_dirs:
        base_site_dir = os.path.join(base_dir, sdir, 'vertical')
        test_site_dir = os.path.join(
            test_dir, sdir, 'vertical') if test_dir is not None else None

        for fpair in _iter_file_pairs('*.mod', base_site_dir, test_site_dir, date_range):
            yield fpair


def iter_vmr_file_pairs(base_dir, test_dir, date_range):
    for fpair in _iter_file_pairs('*.vmr', base_dir, test_dir, date_range):
        yield fpair


def iter_map_file_pairs(base_dir, test_dir, nc, date_range):
    pattern = '*.map.nc' if nc else '*.map'
    for fpair in _iter_file_pairs(pattern, base_dir, test_dir, date_range):
        yield fpair


def _iter_file_pairs(pattern, base_dir, test_dir, date_range):
    all_base_site_files = sorted(glob(os.path.join(base_dir, pattern)))

    for base_file in all_base_site_files:
        file_date = re.search(r'\d{10}Z', os.path.basename(base_file)).group()
        file_date = datetime.strptime(file_date, '%Y%m%d%HZ')
        if date_range is not None and (file_date < date_range[0] or file_date >= date_range[1]):
            continue

        if test_dir is None:
            yield base_file
        else:
            test_file = os.path.join(test_dir, os.path.basename(base_file))
            if not os.path.exists(test_file):
                raise FileNotFoundError(
                    f'Could not find a test file corresponding to the base file {base_file}. '
                    f'Was looking for {test_file}.'
                )
            else:
                yield base_file, test_file


def iter_file_pairs_by_time(pattern, base_dir, test_dir, test_pattern=None, date_range=None):
    def make_file_dict(file_dir, pat):
        files = sorted(glob(os.path.join(file_dir, pat)))
        return {mod_utils.find_datetime_substring(os.path.basename(f), datetime): f for f in files}

    base_site_files = make_file_dict(base_dir, pattern)
    test_site_files = make_file_dict(
        test_dir, pattern if test_pattern is None else test_pattern)

    for base_date, base_file in base_site_files.items():
        if date_range is not None and (base_date < date_range[0] or base_date >= date_range[1]):
            continue
        if base_date not in test_site_files:
            raise FileNotFoundError(
                'Could not find a test file in {} for time {}'.format(test_dir, base_date))
        else:
            yield base_file, test_site_files[base_date]


def _get_nc_unit_scales(map_nc_file):
    with Dataset(map_nc_file) as ds:
        vmr_scales = {v: MAP_UNIT_SCALES[ds.variables[v].full_units] for v in VMR_TO_MAP['profile'].values()}
        return {'profile': vmr_scales}


def _get_map_unit_scales(map_file):
    nhead = mod_utils.get_num_header_lines(map_file)
    with open(map_file) as robj:
        for i in range(nhead-2):
            robj.readline()
        columns = robj.readline().strip().split(',')
        units = robj.readline().strip().split(',')

    units = {c: u for c, u in zip(columns, units)}
    vmr_scales = {c: MAP_UNIT_SCALES[units[c]] for c in VMR_TO_MAP['profile'].values()}
    return {'profile': vmr_scales}
