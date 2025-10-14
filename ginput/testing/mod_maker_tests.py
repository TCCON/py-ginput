from __future__ import print_function, division
import datetime as dt
from glob import glob
from matplotlib import pyplot as plt
import netCDF4 as ncdf
import numpy as np
import os
import re
import unittest

from . import test_utils
from ..common_utils import mod_utils, readers
from ..priors import tccon_priors, map_maker
from ..mod_maker.mod_maker import driver as mmdriver
from .. import __version__


_mydir = os.path.abspath(os.path.dirname(__file__))


class _PseudoSubTest(object):
    """A dummy subtest for the file comparer when not being used as part of a test suite.

    Serves as a fake sub test context manager. If an AssertionError is raised within it, it will print the error message
    and suppress the error.
    """
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None and exc_type is AssertionError:
            print(exc_val)
            return True


class FileComparer(object):
    """Helper class for comparing two .mod or .vmr files

    This class acts as a dummy unit test. It can be used to compare two files manually, in which case plots will be
    created and displayed to the screen for any variables that do not match. It requires no instantiation arguments.
    In most cases, the convenience function :func:`compare_two_files` will be easier to use.
    """
    def __init__(self):
        self._files_match = True

    def subTest(self, *arg, **kwargs):
        """A placeholder subtest method for use when called interactively, instead of as part of a unit test.

        Swallows all input arguments.

        Returns
        -------
        _PseudoSubTest
            A _PseudoSubTest context manager.
        """
        return _PseudoSubTest()

    def assertTrue(self, result, msg):
        """A placeholder assertion method for use when called interactively, instead of as part of a unit test.

        Parameters
        ----------
        result : bool
            Whether the test succeeded or not.

        msg : str
            Message to print if the test failed.

        Raises
        ------
        AssertionError
            if the test failed.
        """
        if not result:
            self._files_match = False
            raise AssertionError(msg)

    @staticmethod
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

    def compare_two_files(self, check_file, new_file, variable_mapping=None, variable_scaling=None):
        """Check that the data contained in two files are identical

        Parameters
        ----------
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

        # This function relies on the method resolution order in Python to perform as expected. The problem was I needed
        # to be able to use the same comparison mechanics I'd written for the unit tests to compare any two arbitrary
        # files, but the subtest and assertion mechanics were too interwoven to effectively encapsulate the test
        # mechanism as its own function outside the unit test framework. So when this class is used on its own, then
        # `self.subTest`, `self._plot_helper`, and `self.assertTrue` all resolve to those methods on *this* class which
        # are set up for manual tests. However, when this is "mixed in" as the second parent class to the test case,
        # these resolve to:
        #   * subTest -> unittest.TestCase.subTest
        #   * assertTrue -> unittest.TestCase.assertTrue
        #   * _plot_helper -> __class__._plot_helper
        # thus behaving as a unit test.

        # will be set to `False` if any of the subtests fail
        self._files_match = True
        check_data = self.read_file(check_file)
        new_data = self.read_file(new_file)

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

                with self.subTest(check_file=check_file, new_file=new_file, category=category_name,
                                  variable=variable_name):
                    test_result = _test_single_variable(variable_data, this_new_data)
                    if not test_result:
                        try:
                            self._plot_helper(check_data=check_data, new_data=new_data, category=category_name,
                                              variable=variable_name, new_file=new_file, new_variable=new_var)
                        except Exception as err:
                            print('Could not generate plot for {} {}, error was {}'.format(new_file, variable_name,
                                                                                           err.args[0]))
                            print(new_data[category_name].keys())
                            print(check_data[category_name].keys())
                    self.assertTrue(test_result, msg='"{category}/{variable}" in {filename} does not match the check data'
                                    .format(category=category_name, variable=variable_name, filename=new_file))

        return self._files_match

    @staticmethod
    def _plot_helper(check_data, new_data, category, variable, new_file, new_variable=None):
        """Create a plot for a variable that differed between the two files

        Parameters
        ----------
        check_data : dict
            dictionary of data read in from the "true" file

        new_data : dict
            dictionary of data read in from the new file

        category : str
            which category (i.e. first level key in the two dictionaries) to get the data from

        variable : str
            which variable (i.e. second level key in the two dictionaries) to get as the data

        new_file : str or None
            path to the new file that had the bad variable. Not actually used, but a necessary input for the unit
            testing version.
        """
        _plot_failed_test(check_data=check_data, new_data=new_data, category=category, variable=variable,
                          new_variable=new_variable)


class TestModMaker(unittest.TestCase, FileComparer):
    @classmethod
    def setUpClass(cls):
        print('Testing version {} from file {}'.format(__version__, __file__))
        # Clean up any old output files
        _clean_up_files_recursive(test_utils.mod_output_dir, r'.*\.mod$')
        _clean_up_files_recursive(test_utils.vmr_output_dir, r'.*\.vmr$')
        _clean_up_files_recursive(test_utils.map_output_dir, r'.*\.map(.nc)?$')

        # Get the GEOS FP/FP-IT data we need for the test if needed, check that the SHA1 sums are what is expected
        test_utils.download_test_geos_data()
        date_range = [test_utils.test_date, test_utils.test_date+dt.timedelta(days=1)]

        # Run mod_maker for the standard test site
        mmdriver(date_range, test_utils.geos_fp_dir, save_path=test_utils.mod_output_top_dir, keep_latlon_prec=False,
                 save_in_utc=True, site_abbrv=test_utils.test_site, include_chm=True, mode='fpit-eta')

        # Run the priors using the new .mod files - we need to make sure that changes to the .mod files do or don't
        # affect the .vmrs.
        mod_files = [f for f in test_utils.iter_mod_file_pairs(test_utils.mod_output_dir, None)]
        tccon_priors.generate_full_tccon_vmr_file(mod_files, dt.timedelta(hours=0), save_dir=test_utils.vmr_output_dir,
                                                  std_vmr_file=test_utils.std_vmr_file, site_abbrevs=test_utils.test_site)

        # Create the wet map files - both netCDF and text
        mod_dir = os.path.dirname(mod_files[0])
        if not os.path.isdir(test_utils.map_output_dir):
            print('Making', test_utils.map_output_dir)
            os.mkdir(test_utils.map_output_dir)
        common_map_args = dict(date_range=date_range, mod_dir=mod_dir, vmr_dir=test_utils.vmr_output_dir,
                               save_dir=test_utils.map_output_dir, dry=False, site_abbrev=test_utils.test_site)
        map_maker.cl_driver(map_fmt='nc', **common_map_args)
        map_maker.cl_driver(map_fmt='txt', **common_map_args)

    def test_mod_files(self):
        self._comparison_helper(test_utils.iter_mod_file_pairs, test_utils.mod_input_dir, test_utils.mod_output_dir)

    def test_vmr_files(self):
        self._comparison_helper(test_utils.iter_vmr_file_pairs, test_utils.vmr_input_dir, test_utils.vmr_output_dir)

    def test_map_files(self):
        self._comparison_helper(lambda b, t: test_utils.iter_map_file_pairs(b, t, nc=True),
                                test_utils.map_input_dir, test_utils.map_output_dir)
        self._comparison_helper(lambda b, t: test_utils.iter_map_file_pairs(b, t, nc=False),
                                test_utils.map_input_dir, test_utils.map_output_dir)

    def _comparison_helper(self, iter_fxn, input_dir, output_dir):
        for check_file, new_file in iter_fxn(input_dir, output_dir):
            self.compare_two_files(check_file, new_file)

    @staticmethod
    def _plot_helper(check_data, new_data, category, variable, new_file, new_variable=None):
        datestr = mod_utils.find_datetime_substring(new_file)
        savename = '{var}_{date}Z.pdf'.format(var=variable, date=datestr)
        savename = os.path.join(test_utils.test_plots_dir, savename)
        _plot_failed_test(check_data=check_data, new_data=new_data, category=category, variable=variable,
                          save_as=savename, new_variable=new_variable)


class TestMapMaker(unittest.TestCase, FileComparer):
    """Map-maker specific tests

    This test case handles checking that the map maker is behaving correctly and that the profiles match their
    respective variables in the .mod and .vmr file. The standard tests, comparing .map files against benchmarks, is
    handled in TestModMaker.
    """
    mod_to_ncmap = {'profile': {'Height': 'altitude', 'Temperature': 'temp', 'Pressure': 'pressure'}}
    mod_to_txtmap = {'profile': {'Height': 'Height', 'Temperature': 'Temp', 'Pressure': 'Pressure'}}
    vmr_to_map = {'profile': {'h2o': 'h2o', 'hdo': 'hdo', 'co2': 'co2', 'n2o': 'n2o', 'co': 'co', 'ch4': 'ch4', 'hf': 'hf', 'o2': 'o2'}}

    _date_range = [test_utils.test_date, test_utils.test_date + dt.timedelta(days=1)]
    _mod_dir = os.path.join(test_utils.mod_input_dir, test_utils.test_site, 'vertical')
    _vmr_dir = test_utils.vmr_input_dir
    _map_dir = test_utils.map_dry_output_dir

    _unit_scales = {'mol/mol': 1.0, 'parts': 1.0, 'ppm': 1e-6, 'ppb': 1e-9, 'ppt': 1e-12}

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls._map_dir):
            print('Making', cls._map_dir)
            os.mkdir(cls._map_dir)
        else:
            _clean_up_files(os.path.join(cls._map_dir, '*.map*'))

        common_opts = dict(date_range=cls._date_range, mod_dir=cls._mod_dir, vmr_dir=cls._vmr_dir,
                           save_dir=cls._map_dir, dry=True, site_abbrev=test_utils.test_site)
        map_maker.cl_driver(map_fmt='nc', **common_opts)
        map_maker.cl_driver(map_fmt='txt', **common_opts)

    def test_nc_map_files(self):

        for mod_file, map_file in test_utils.iter_file_pairs_by_time(pattern='*.mod', test_pattern='*.map.nc', base_dir=self._mod_dir, test_dir=self._map_dir):
            self.compare_two_files(mod_file, map_file, variable_mapping=self.mod_to_ncmap)

        for vmr_file, map_file in test_utils.iter_file_pairs_by_time(pattern='*.vmr', test_pattern='*.map.nc', base_dir=self._vmr_dir, test_dir=self._map_dir):
            vmr_scales = self._get_nc_unit_scales(map_file)
            self.compare_two_files(vmr_file, map_file, variable_mapping=self.vmr_to_map, variable_scaling=vmr_scales)

    def test_txt_map_files(self):
        for mod_file, map_file in test_utils.iter_file_pairs_by_time(pattern='*.mod', test_pattern='*.map', base_dir=self._mod_dir, test_dir=self._map_dir):
            self.compare_two_files(mod_file, map_file, variable_mapping=self.mod_to_txtmap)

        for vmr_file, map_file in test_utils.iter_file_pairs_by_time(pattern='*.vmr', test_pattern='*.map', base_dir=self._vmr_dir, test_dir=self._map_dir):
            vmr_scales = self._get_map_unit_scales(map_file)
            self.compare_two_files(vmr_file, map_file, variable_mapping=self.vmr_to_map, variable_scaling=vmr_scales)

    @classmethod
    def _get_nc_unit_scales(cls, map_file):
        with ncdf.Dataset(map_file) as ds:
            vmr_scales = {v: cls._unit_scales[ds.variables[v].full_units] for v in cls.vmr_to_map['profile'].values()}
            return {'profile': vmr_scales}

    @classmethod
    def _get_map_unit_scales(cls, map_file):
        nhead = mod_utils.get_num_header_lines(map_file)
        with open(map_file) as robj:
            for i in range(nhead-2):
                robj.readline()
            columns = robj.readline().strip().split(',')
            units = robj.readline().strip().split(',')

        units = {c: u for c, u in zip(columns, units)}
        vmr_scales = {c: cls._unit_scales[units[c]] for c in cls.vmr_to_map['profile'].values()}
        return {'profile': vmr_scales}

    @staticmethod
    def _plot_helper(check_data, new_data, category, variable, new_file, new_variable=None):
        prefix = 'ncmap' if new_file.endswith('.nc') else 'map'
        datestr = mod_utils.find_datetime_substring(new_file)
        savename = '{pre}_{var}_{date}Z.pdf'.format(pre=prefix, var=variable, date=datestr)
        savename = os.path.join(test_utils.test_plots_dir, savename)
        _plot_failed_test(check_data=check_data, new_data=new_data, category=category, variable=variable,
                          save_as=savename, new_variable=new_variable)


def _clean_up_files(pattern):
    files = sorted(glob(pattern))
    for f in files:
        print('Removing', f)
        os.remove(f)


def _clean_up_files_recursive(top_dir, pattern):
    for dirname, _, files in os.walk(top_dir):
        for f in files:
            if re.match(pattern, f):
                fullfile = os.path.join(dirname, f)
                print('Removing', fullfile)
                os.remove(fullfile)


def _test_single_variable(variable_data, this_new_data):
    try:
        # We need some absolute tolerance, otherwise inconsequential differences cause the test to fail. E.g. as N2O and
        # CH4 go to zero, a difference of 1e-13 parts triggers a failure, which really doesn't matter. We'll make the
        # absolute tolerance equal to 0.01% of the maximum value in the original data, because a 0.01% difference in
        # the prior concentration really shouldn't matter.
        atol = 1e-4 * np.abs(np.nanmax(variable_data))
        return np.isclose(variable_data, this_new_data, atol=atol).all()
    except TypeError:
        # Not all variables with be float arrays. If np.isclose() can't coerce the data to a numeric
        # type, it'll raise a TypeError and we fall back on the equality test. nanmax will also throw a TypeError
        # in similar circumstances
        return np.all(variable_data == this_new_data)


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


def compare_two_files(old_file, new_file, **kws):
    comparer = FileComparer()
    return comparer.compare_two_files(old_file, new_file, **kws)


if __name__ == '__main__':
    unittest.main()
