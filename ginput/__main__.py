import argparse
import sys

from .priors import acos_interface as aci, tccon_priors, map_maker, mlo_smo_prep, fo2_prep, automation
from .mod_maker import mod_maker
from .download import get_GEOS5, get_NOAA_flask_data, get_fo2_data
from .tools import tar2nc


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    driver = args.pop('driver_fxn')
    driver(**args)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Call various pieces of ginput')
    subparsers = parser.add_subparsers(help='The following subcommands execute different parts of ginput')

    oco_parser = subparsers.add_parser('oco', help='Generate .h5 file for input into the OCO algorithm')
    aci.parse_args(oco_parser, instrument='oco')
    gosat_parser = subparsers.add_parser('acos', help='Generate .h5 file for input into the GOSAT algorithm')
    aci.parse_args(gosat_parser, instrument='gosat')
    geocarb_parser = subparsers.add_parser('geocarb', help='Generate .h5 file for input into the GeoCarb algorithm')
    aci.parse_args(geocarb_parser, instrument='geocarb')

    noaa_parser = subparsers.add_parser('update_hourly', help='Update monthly input CO2 files with new NOAA hourly data')
    mlo_smo_prep.parse_args(noaa_parser)

    o2_parser = subparsers.add_parser('update_fo2', help='Update or create the O2 mole fraction file')
    fo2_prep.parse_args(o2_parser)

    mm_parser = subparsers.add_parser('mod', help='Generate .mod (model) files for GGG')
    mod_maker.parse_args(mm_parser)

    mm_tccon_parser = subparsers.add_parser('tccon-mod', help='Generate .mod (model) files appropriate for use with '
                                                              'TCCON GGG2020 retrievals.')
    mod_maker.parse_vmr_args(mm_tccon_parser)

    mmrl_parser = subparsers.add_parser('rlmod', help='Generate .mod (model) files for spectra enumerated in a runlog')
    mod_maker.parse_runlog_args(mmrl_parser)

    mmrl_tccon_parser = subparsers.add_parser('tccon-rlmod', help='Generate .mod (model) files appropriate for use with'
                                                                  ' TCCON GGG2020 retrievals for spectra enumerated in'
                                                                  ' a runlog.')
    mod_maker.parse_vmr_args(mmrl_tccon_parser, backend=mod_maker.parse_runlog_args)

    vmr_parser = subparsers.add_parser('vmr', help='Generate full .vmr files for GGG')
    tccon_priors.parse_args(vmr_parser)

    vmr_rl_parser = subparsers.add_parser('rlvmr', help='Generate .vmr files from a runlog')
    tccon_priors.parse_runlog_args(vmr_rl_parser)

    get_g5_parser = subparsers.add_parser('getg5', help='Download GEOS5 FP or FP-IT data')
    get_GEOS5.parse_args(get_g5_parser)

    get_rlg5_parser = subparsers.add_parser('get-rl-g5', help='Download GEOS5 FP or FP-IT data for spectra in a runlog')
    get_GEOS5.parse_runlog_args(get_rlg5_parser)

    get_noaa_parser = subparsers.add_parser('getnoaa', help='Download NOAA flask data')
    get_NOAA_flask_data.parse_args(get_noaa_parser)

    get_fo2_parser = subparsers.add_parser('getfo2', help='Download input data required to calculate f(O2)')
    get_fo2_data.parse_args(get_fo2_parser)

    map_parser = subparsers.add_parser('map', help='Generate .map (a priori) files.')
    map_maker.parse_cl_args(map_parser)

    auto_parser = subparsers.add_parser('auto', help='Entry point for running ginput in an automation environment')
    automation.parse_cl_args(auto_parser)

    tar2nc_parser = subparsers.add_parser('tar2nc', help='Collate multiple .mod and .vmr files into a single .nc file')
    tar2nc.parse_args(tar2nc_parser)

    clargs = vars(parser.parse_args(args))
    if 'driver_fxn' not in clargs:
        parser.print_usage()
    else:
        return clargs
