#!/usr/bin/env python
import argparse

from ginput.priors import acos_interface as aci, tccon_priors, map_maker, fo2_prep, automation
from ginput.mod_maker import mod_maker
from ginput.download import get_GEOS5, get_fo2_data


def parse_args():
    parser = argparse.ArgumentParser(description='Call various pieces of ginput')
    subparsers = parser.add_subparsers(help='The following subcommands execute different parts of ginput')

    oco_parser = subparsers.add_parser('oco', help='Generate .h5 file for input into the OCO algorithm')
    aci.parse_args(oco_parser, oco_or_gosat='oco')
    gosat_parser = subparsers.add_parser('acos', help='Generate .h5 file for input into the GOSAT algorithm')
    aci.parse_args(gosat_parser, oco_or_gosat='gosat')
    
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

    get_fo2_parser = subparsers.add_parser('getfo2', help='Download input data required to calculate f(O2)')
    get_fo2_data.parse_args(get_fo2_parser)

    map_parser = subparsers.add_parser('map', help='Generate .map (a priori) files.')
    map_maker.parse_cl_args(map_parser)

    auto_parser = subparsers.add_parser('auto', help='Entry point for running ginput in an automation environment')
    automation.parse_cl_args(auto_parser)

    clargs = vars(parser.parse_args())
    if 'driver_fxn' not in clargs:
        parser.print_usage()
    else:
        return clargs


def main():
    args = parse_args()
    driver = args.pop('driver_fxn')
    driver(**args)


if __name__ == '__main__':
    main()
