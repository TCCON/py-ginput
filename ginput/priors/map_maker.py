from argparse import ArgumentParser
import os
import pandas as pd

from ..common_utils import mod_utils, writers
from ..mod_maker import tccon_sites


def _find_files_in_dirs(mod_dir, vmr_dir, date_range, site_lat, site_lon, product='fpit', keep_latlon_prec=False,
                        skip_missing=False):
    if not os.path.isdir(mod_dir):
        raise IOError('Cannot find expected mod file directory: {}'.format(mod_dir))
    if not os.path.isdir(vmr_dir):
        raise IOError('Cannot find expected vmr file directory: {}'.format(vmr_dir))

    # Go through each expected mod and vmr file for the given date range.
    dates = pd.date_range(date_range[0], date_range[1], freq='3h', inclusive='left')
    round_latlon = not keep_latlon_prec
    all_mod_files = []
    all_vmr_files = []
    for date in dates:
        mod_file = mod_utils.mod_file_name_for_priors(date, site_lat=site_lat, site_lon_180=site_lon, round_latlon=round_latlon, prefix=product.upper())
        mod_file = os.path.join(mod_dir, mod_file)
        if not os.path.isfile(mod_file):
            if skip_missing:
                continue
            else:
                raise IOError('Failed to find mod file {}'.format(mod_file))

        vmr_file = mod_utils.vmr_file_name(date, lon=site_lon, lat=site_lat, keep_latlon_prec=keep_latlon_prec)
        vmr_file = os.path.join(vmr_dir, vmr_file)
        if not os.path.isfile(vmr_file):
            if skip_missing:
                continue
            else:
                raise IOError('Failed to find vmr file {}'.format(vmr_file))

        all_mod_files.append(mod_file)
        all_vmr_files.append(vmr_file)

    return all_mod_files, all_vmr_files


def _find_files_in_dir_tree(root_dir, date_range, site_lat, site_lon, site_abbrev, product, keep_latlon_prec=False,
                            skip_missing=False):
    # Check that `root_dir` contains <product>/<site>/vertical and <product>/<site>/vmrs-vertical
    mod_dir = mod_utils.mod_output_subdir(root_dir, site_abbrev=site_abbrev, product=product)
    vmr_dir = mod_utils.vmr_output_subdir(root_dir, site_abbrev=site_abbrev, product=product)
    return _find_files_in_dirs(mod_dir=mod_dir, vmr_dir=vmr_dir, date_range=date_range,
                               site_lat=site_lat, site_lon=site_lon, product=product,
                               keep_latlon_prec=keep_latlon_prec, skip_missing=skip_missing)


def _cl_get_mod_vmr_files(root_dir, mod_dir, vmr_dir, date_range, site_lat, site_lon, site_abbrev, save_dir,
                          product='fpit', keep_latlon_prec=False, skip_missing=False):
    # 3 cases:
    #   1) no directory specified - use GGG
    #   2) root dir specified - use assumed file structure
    #   3) mod/vmr dirs specified -
    if mod_dir is None != vmr_dir is None:
        raise TypeError('Must specify both or neither of `mod_dir` and `vmr_dir`')

    # For the next step, we rely on the fact that we just checked that mod_dir and vmr_dir are either both None or both
    # not None. We use mod_dir as a proxy for both.
    if root_dir is None and mod_dir is None:
        mod_dir = mod_utils.get_ggg_path(os.path.join('models', 'gnd'), 'model file directory')
        vmr_dir = mod_utils.get_ggg_path(os.path.join('vmrs', 'gnd'), 'vmr file directory')
    elif root_dir is not None and mod_dir is not None:
        raise TypeError('Must specify *either* `root_dir` or `mod_dir`+`vmr_dir`, not both.')

    # Now we've verified that the input is valid: either root_dir was given *or* mod_dir & vmr_dir, but not both. If
    # neither, then we got the paths to our GGG path.
    if root_dir is not None:
        if save_dir is None:
            save_dir = os.path.join(root_dir, product, site_abbrev, 'maps-vertical')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        mods, vmrs = _find_files_in_dir_tree(
            root_dir=root_dir, date_range=date_range, site_lat=site_lat, site_lon=site_lon, site_abbrev=site_abbrev,
            product=product, keep_latlon_prec=keep_latlon_prec, skip_missing=skip_missing
        )
    elif save_dir is None:
        raise TypeError('Cannot infer a save directory when not using root_dir')
    else:
        mods, vmrs = _find_files_in_dirs(
            mod_dir=mod_dir, vmr_dir=vmr_dir, date_range=date_range, site_lat=site_lat, site_lon=site_lon, product=product,
            keep_latlon_prec=keep_latlon_prec, skip_missing=skip_missing
        )

    return mods, vmrs, save_dir


def _lookup_site_lat_lon(site_abbrev, date_range):
    try:
        site_info = tccon_sites.tccon_site_info_for_date_range(date_range=date_range, site_abbrv=site_abbrev)
    except tccon_sites.TCCONNonUniqueTimeSpanError as err:
        # Make a more straightforward error message
        span_end = err.bad_site_spans[site_abbrev][0][1]
        raise tccon_sites.TCCONTimeSpanError(
            'Unable to lookup unique site lat/lon for site {site} across date range {start} to {stop}. Site moved on '
            '{moved}. Either manually specify a lat/lon or limit the date range to end before the move.'.format(
                site=site_abbrev, start=date_range[0], stop=date_range[1], moved=span_end
            )
        )
    return site_info['lat'], site_info['lon_180']


def cl_driver(date_range, root_dir=None, mod_dir=None, save_dir=None, vmr_dir=None, map_fmt='nc', dry=False,
              product='fpit', site_lat=None, site_lon=None, site_abbrev='xx', keep_latlon_prec=False,
              skip_missing=False, req_cfunits=False):

    site_abbrev, site_lat, site_lon, _ = mod_utils.check_site_lat_lon_alt(abbrev=site_abbrev, lat=site_lat,
                                                                          lon=site_lon,
                                                                          alt=None if site_lat is None else 0.0)

    wet_or_dry = 'dry' if dry else 'wet'

    for this_abbrev, this_lat, this_lon in zip(site_abbrev, site_lat, site_lon):
        if this_lat is None:
            # Assuming if lat is None, lon is as well b/c check_site_lat_lon_alt() should guarantee that.
            this_lat, this_lon = _lookup_site_lat_lon(this_abbrev, date_range)

        this_lon = this_lon - 360 if this_lon > 180 else this_lon
        mod_files, vmr_files, this_save_dir = _cl_get_mod_vmr_files(
            root_dir=root_dir, mod_dir=mod_dir, vmr_dir=vmr_dir, save_dir=save_dir, date_range=date_range,
            site_lon=this_lon, site_lat=this_lat, site_abbrev=this_abbrev, product=product,
            keep_latlon_prec=keep_latlon_prec, skip_missing=skip_missing
        )

        for modf, vmrf in zip(mod_files, vmr_files):
            writers.write_map_from_vmr_mod(vmr_file=vmrf, mod_file=modf, map_output_dir=this_save_dir, fmt=map_fmt,
                                           wet_or_dry=wet_or_dry, site_abbrev=this_abbrev, no_cfunits=not req_cfunits)


def parse_cl_args(p: ArgumentParser):
    p.description = 'Generate .map files from .mod & .vmr files'
    p.add_argument('date_range', type=mod_utils.parse_date_range,
                   help='The range of dates to generate .map files for. May be given as YYYYMMDD-YYYYMMDD, or '
                        'YYYYMMDD_HH-YYYYMMDD_HH, where the ending date is exclusive. A single date may be given, '
                        '(YYYYMMDD) in which case the ending date is assumed to be one day later.')

    iogrp = p.add_argument_group('I/O', 'Arguments for input and output control')
    iogrp.add_argument('mod_dir', nargs='?',
                       help='Directory where the .mod files can be found. See note for --root-dir.')
    iogrp.add_argument('vmr_dir', nargs='?',
                       help='Directory where the .vmr files can be found. See note for --root-dir.')
    iogrp.add_argument('-r', '--root-dir',
                       help='Top directory of the standard .mod/.vmr directory tree, that is, a directory containing '
                            'fpit/<site>/vertical and fpit/<site>/vmrs-vertical. Specify *either* this argument *or* '
                            'mod_dir and vmr_dir, *not* both. ')
    iogrp.add_argument('-s', '--save-dir',
                       help='Directory to save the .map files to. If not given, but passing --root-dir (and not '
                            'mod_dir+vmr_dir), then .map files are automatically saved in the root directory under '
                            'fpit/<site>/maps-vertical.')
    iogrp.add_argument('--product', default='fpit', choices=('fp', 'fpit'),
                       help='Which meteorology product you used. "fp" = GEOS-FP, "fpit" = GEOS-FPIT. Only required if '
                            'specifying --root-dir instead of mod_dir+vmr_dir. Default is %(default)s')
    iogrp.add_argument('-k', '--keep-latlon-prec', action='store_true',
                       help='Use 2 decimal places for lat/lon in the names of the .mod files. This must match the '
                            'format of your .mod file names, the default is to round to the nearest degree.')

    sitegrp = p.add_argument_group('Location', 'Arguments specifying which site/location to generate .map files for')
    valid_site_ids = list(tccon_sites.tccon_site_info().keys())
    sitegrp.add_argument('--site', default='xx', choices=valid_site_ids, dest='site_abbrev',
                         help='Which site to generate priors for. Used to set the lat/lon looked for in the file name. '
                              'If an explicit lat and lon are given, those override this.')
    sitegrp.add_argument('--lat', type=float, dest='site_lat', help='Latitude to generate prior for. If given, '
                                                                    '--lon must be given as well.')
    sitegrp.add_argument('--lon', type=float, dest='site_lon', help='Longitude to generate prior for. If given, '
                                                                    '--lat must be given as well.')

    fmtgrp = p.add_argument_group('Format', 'Control the format of the output file')
    fmtgrp.add_argument('-f', '--map-fmt', choices=('nc', 'txt'), default='nc',
                        help='Select the output format for the .map files, "nc" for netCDF for "txt" for the legacy '
                             'text format. Default is "%(default)s".')
    fmtgrp.add_argument('-d', '--dry', action='store_true',
                        help='Save the priors as dry mole fraction instead of wet. Note that TCCON uses wet mole '
                             'fractions in the retrieval. If you have questions about which to use for your '
                             'application, please see the TCCON wiki (https://tccon-wiki.caltech.edu/) or contact '
                             'the ginput maintainer.')

    othergrp = p.add_argument_group('Other', 'Additional arguments')
    othergrp.add_argument('-m', '--skip-missing', action='store_true',
                          help='If there are missing .mod or .vmr files in your requested date range, skip generating '
                               'those .map files. Otherwise, an error is raised.')
    othergrp.add_argument('-c', '--req-cfunits', action='store_true',
                          help='Require that CFUnits be successfully imported. This is used to enforce CF unit '
                               'conventions in netCDF output files. In some cases, CFUnits cannot be imported due to '
                               'a C-library incompatibility. Use this flag if following CF unit conventions is '
                               'necessary for your use of the .map files and you do not get a warning about CFUnits '
                               'failing to import.')
    p.set_defaults(driver_fxn=cl_driver)
