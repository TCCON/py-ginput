#!~/anaconda2/bin/python
# -*- coding: utf-8 -*-

"""
README

mod_maker10.f translated into python

What's different?

#########################################################################################################################################################################

OLD: used to generate MOD files using ncep (like the IDL code), or using merra or fp or fpit data

python mod_maker.py arg1 site=arg2 mode=arg3 time=arg4 step=arg5 lat=arg6 lon=arg7 alt=arg8 save_path=arg9 ncdf_path=arg10

arg1: date range (YYYYMMDD-YYYYMMDD, second one not inclusive, so you don't have to worry about end of months) or single date (YYYYMMDD)
arg2: two letter site abbreviation (e.g. 'oc' for Lamont, Oklahoma; see the "site_dict" dictionary)
arg3: mode ('ncep', 'merradap42', 'merradap72', 'merraglob', 'fpglob', 'fpitglob'), ncep and 'glob' modes require local files
the 'merradap' modes require a .netrc file in your home directory with credentials to connect to urs.earthdata.nasa.gov
arg4: (optional, default=12:00)  hour:minute (HH:MM) for the starting time in local time
arg5: (optional, default=24) time step in hours (can be decimal)
arg6: (optional) latitude in [-90,90] range
arg7: (optional) longitude in [0,360] range
arg8: (optional) altitude (meters)
arg9: (optional) full path to directory where data will be saved (save_path/fpit will be created), defaults to GGGPATH/models/gnd
arg10: (optional) full path to the directory where ncep/geos/merra files are located, defaults to GGGPATH/ncdf

A custom site location can be given, in that case arg6,arg7, and arg8 must be specified and a site name must be made up for arg2

add 'mute' in the command line (somewhere after arg1) and there will be no print statements other than warnings and error messages 

MOD files will be saved under save_path/xx/site.
With xx either 'ncep', 'merra', 'fp', or 'fpit'
The merradap modes require an internet connection and EarthData credentials
The ncep mode requires the global NCEP netcdf files of the given year to be present in GGGPATH/ncdf

The fpglob or fpitglob modes expect two files in 'ncdf_path' containing concatenated 3-hourly files for surface and multi-level data, the concatenated files need to be generated beforehand
e.g.
GEOS_fpit_asm_inst3_2d_asm_Nx_GEOS5124.20171210_20171217.nc4 # surface data
GEOS_fpit_asm_inst3_2d_asm_Np_GEOS5124.20171210_20171217.nc4 # multi-level data

The merraglob mode works like the fpglob and fpitglob modes and will expect two files:
e.g.
MERRA2_asm_inst3_2d_asm_Nx_GEOS5124.20171210_20171217.nc4 # surface data
MERRA2_asm_inst3_2d_asm_Np_GEOS5124.20171210_20171217.nc4 # multi-level data

The ncep mode should produce files identical to the IDL mod_maker if 'time' and 'step' are kept as default

#########################################################################################################################################################################

NEW: used to generate MOD files on GEOS5-FP-IT times for all TCCON sites at once using GEOS5-FP-IT 3-hourly files

python mod_maker.py arg1 geos_path=arg2 site=arg3 lat=arg4 lon=arg5 alt=arg6 save_path=arg7

arg1: date range (YYYYMMDD-YYYYMMDD, second one not inclusive, so you don't have to worry about end of months) or a single date (YYYYMMDD) in which case the end date is +24h
You can also give YYYYMMDD_HH instead to specify the hour, but these must be exact GEOS5 times (UTC times 3 hourly from 00)
arg2: full path to directory containing the daily GEOS5-FP-IT files
arg3: (optional) two letter site abbreviation
arg4: (optional) latitude in [-90,90] range
arg5: (optional) longitude in [0,360] range
arg6: (optional) altitude (meters)
arg7: (optional) full path to directory where data will be saved (save_path/fpit will be created), defaults to GGGPATH/models/gnd

If arg3 is specified, MOD files will only be produced for that one site. See the dictionary in tccon_sites.py for site abbreviations of existing sites.

A custom site location can be given, in that case arg3,arg4,arg5, and arg6 must be specified

add 'mute' in the command line (somewhere after arg1) and there will be no print statements other than warnings and error messages

add 'slant' in the command line (somewhere after arg1) to generate both vertical and slant MOD files.

two folders are expected in the geos_path directory:
in geos_path/Np you must have all the 42 levels GEOS5-FP-IT files
in geos_path/Nx you must have all the surface data files

Running the code like this will generate MOD files for ALL sites withtin the date range on GEOS5 times (every 3 hours) using GEOS5-FP-IT 3-hourly files
MOD files will be generate both along the vertical and along the sun ray

They will be saved under save_path/fpit/xx/yy
with xx the two letter site abbreviation and yy either 'vertical' or 'slant'

The slant .mod files are only generated when the SZA is above 90 degrees.
#########################################################################################################################################################################

There is dictionary of sites with their respective lat/lon in tccon_sites.py, so this works for all TCCON sites, lat/lon values were taken from the wiki page of each site.
"""
import argparse
import glob
import os, sys
import numpy.ma as ma
import pandas as pd
from scipy.interpolate import interp1d, interp2d
import netCDF4 # netcdf I/O
import re # used to parse strings
import time
import netrc # used to connect to earthdata
from astropy.time import Time # this is essentialy like datetime, but with better methods for conversion of datetime to / from julian dates, can also be converted to datetime
from pydap.cas.urs import setup_session # used to connect to the merra opendap servers
import xarray
import warnings

from ..common_utils import mod_utils, run_utils
from ..common_utils.mod_utils import gravity, check_site_lat_lon_alt
from ..common_utils.mod_constants import ratio_molec_mass as rmm, p_ussa, t_ussa, z_ussa, mass_dry_air
from ..common_utils.versioning import GeosVersion
from ..common_utils.ggg_logging import logger
from .slantify import * # code to make slant paths
from .tccon_sites import site_dict, tccon_site_info, tccon_site_info_for_date


####################
# Module constants #
####################

_old_modmaker_modes = ('ncep','merradap42','merradap72','merraglob','fpglob','fpitglob')
_new_fixedp_modes = ('fpit', 'fp')
_new_native_modes = ('fpit-eta', 'fp-eta', 'it-eta')
_new_modmaker_modes = _new_fixedp_modes + _new_native_modes
_default_mode = 'fpit-eta'


def shell_error(msg, ecode=1):
    print(msg)
    sys.exit(ecode)


def compute_h2o_dmf(qv, rmm):
    """
    compute h2o dry mole fraction from specific humidity
    """
    return rmm*qv/(1-qv)

def compute_h2o_wmf(h2o_dmf):
    """
    compute h2o wet mole fraction from h2o dry mole fraction
    """
    return h2o_dmf/(1+h2o_dmf)

def compute_rh(t,h2o_wmf,p):
    """
    compute relative humidity from h2o wet mole fraction, pressure and temperature
    """
    svp = svp_wv_over_ice(t)
    return 100*h2o_wmf*p/svp

def compute_mmw(h2o_wmf):
    """
    compute mean molecular weight of air from h2o wet mole fraction
    """
    # mass_dry_air in kg/mol, want g/mol
    return 1e3*mass_dry_air*(1-h2o_wmf)+18.02*h2o_wmf

def svp_wv_over_ice(temp):
    """
    Uses the Goff-Gratch equation to calculate the saturation vapor
    pressure of water vapor over ice at a user-specified temperature.
        Input:  temp (K)
        Output: svp (mbar)
    """
    t0 = 273.16	# triple point temperature
    tr = t0/temp
    yy = -9.09718*(tr-1)-3.56654*np.log10(tr)+0.876793*(1-1/tr)
    svp = 6.1173*10**yy # saturation vapor pressure over ice (mbar)

    return svp

mod_var_fmt_info = {'lev':     {'total_width': 13, 'format': '9.3e',  'scale': 1,   'name': 'Pressure', 'units': 'mbar'},
                    'T':       {'total_width': 13, 'format': '11.3f',  'scale': 1,   'name': 'Temperature', 'units': 'Kelvin'},
                    'H':       {'total_width': 9,  'format': '7.3f',  'scale': 1,   'name': 'Height', 'units': 'km'},
                    'mmw':     {'total_width': 12, 'format': '7.4f',  'scale': 1,   'name': 'MMW', 'units': 'g/mole'},
                    'H2O_DMF': {'total_width': 12, 'format': '10.3e', 'scale': 1,   'name': 'H2O', 'units': 'DMF'},
                    'RH':      {'total_width': 8,  'format': '>6.1f', 'scale': 100, 'name': 'RH', 'units': '%'},
                    'EPV':     {'total_width': 15, 'format': '10.3e', 'scale': 1,   'name': 'EPV', 'units': 'K.m+2/kg/s'},
                    'PT':      {'total_width': 11, 'format': '8.3f',  'scale': 1,   'name': 'PT', 'units': 'Kelvin'},
                    'EL':      {'total_width': 11, 'format': '7.3f',  'scale': 1,   'name': 'EqL', 'units': 'degrees'},
                    'O3':      {'total_width': 11, 'format': '9.3e',  'scale': 1,   'name': 'O3', 'units': 'kg/kg'},
                    'CO':      {'total_width': 11, 'format': '9.3e',  'scale': 1, 'name': 'CO', 'units': 'mol/mol'}}


def build_mod_fmt_strings(var_order):
    # Units and names just need to have the right total width and be centered
    header_fmt = ''
    data_fmt = ''
    var_names = dict()
    var_units = dict()

    spaces = '    '

    for v in var_order:
        var_info = mod_var_fmt_info[v]

        this_fmt = var_info['format']
        # Assuming the format is something like "9.3e" or "7.3f" the total width of the format is the number before
        # the decimal. Get that and subtract from total width to figure out how many spaces to add to the end of the
        # column.
        fmt_width = int(re.search(r'\d+(?=\.)', this_fmt).group())
        if fmt_width > var_info['total_width']:
            raise NotImplementedError('The format width is greater than the total column width.')

        full_fmt = '{{{}:{}}}'.format(v, this_fmt) + spaces
        data_fmt += full_fmt

        header_fmt += '{{{}:^{}}}'.format(v, fmt_width) + spaces
        var_names[v] = var_info['name']
        var_units[v] = var_info['units']

    header_names = header_fmt.format(**var_names) + '\n'
    header_units = header_fmt.format(**var_units) + '\n'
    var_name_mapping = {v: k for k, v in var_names.items()}
    data_fmt += '\n'
    return header_names, header_units, var_name_mapping, data_fmt


def write_mod(mod_path, version, site_lat, data=0, surf_data=0, func=None, muted=False, slant=False, chem_vars=False, geos_versions=None):
    """
    Creates a GGG-format .mod file
    INPUTS:
        mod_path: full path to write the .mod file
        version: the mod_maker version
        site_lat: site latitude (-90 to 90)
        data: dictionary of the inputs
        surf_data: dictionary of the surface inputs (for merra/geos5)
    """
    if geos_versions is None:
        geos_versions = dict()
    # Output scaling: define factors to multiply values by before writing to the .mod file. If a column name is not
    # here, 1.0 is assumed. Only used for GEOS-type mod files currently.
    output_scaling = {'RH': 100.0}

    # Define US Standard Atmosphere (USSA) for use above 10 mbar (imported from mod_constants)

    # Define constants common to both NCEP and GEOS files: earth's radius, ecc2 (?), site latitude, surface gravity,
    # profile bottom altitude, and base pressure. Tropopause pressure will be added under the NCEP/GEOS part of the code
    # b/c we use different variables under those cases
    # TODO: these should be defined in a constants file and just referenced here, at least the ones that are truly
    #  constant
    mod_constant_names = ('earth_radius', 'ecc2', 'obs_lat', 'surface_gravity', 'profile_base_geometric_alt',
                          'base_pressure', 'tropopause_pressure')
    mod_constants = [6378.137, 6.000E-05, site_lat, 9.81, data['H'][0], 1013.25]
    if type(surf_data)==int: # ncep mode
        # NCEP and GEOS provide different tropopause variables that need to be added
        mod_constants.append(data['TROPP'])
        # The head of the .mod file
        fmt = '{:8.3f} {:11.4e} {:7.3f} {:5.3f} {:8.3f} {:8.3f} {:8.3f}\n'
        mod_content = []
        mod_content+=[	'5  6\n',
                          fmt.format(*mod_constants),
                          version+'\n',
                          ' mbar        Kelvin         km      g/mole      DMF       %\n',
                          'Pressure  Temperature     Height     MMW        H2O      RH\n',	]

        fmt = '{:9.3e}    {:7.3f}    {:7.3f}    {:7.4f}    {:9.3e}{:>6.1f}\n' # format for writting the lines

        # Export the Pressure, Temp and SHum for lower levels (1000 to 300 mbar)
        for k,elem in enumerate(data['H2O_DMF']):
            svp = svp_wv_over_ice(data['T'][k])
            h2o_wmf = compute_h2o_wmf(data['H2O_DMF'][k]) # wet mole fraction of h2o
            frh = h2o_wmf*data['lev'][k]/svp # Fractional relative humidity

            # Relace H2O mole fractions that are too small
            if (frh < 30./data['lev'][k]):
                if not muted:
                    print('Replacing too small H2O ',mod_path, data['lev'][k],h2o_wmf,svp*30./data['lev'][k]/data['lev'][k],frh,30./data['lev'][k])
                frh = 30./data['lev'][k]
                h2o_wmf = svp*frh/data['lev'][k]
                data['H2O_DMF'][k] = h2o_wmf/(1-h2o_wmf)

            # Relace H2O mole fractions that are too large (super-saturated)  GCT 2015-08-05
            if (frh > 1.0):
                if not muted:
                    print('Replacing too large H2O ',mod_path,data['lev'][k],h2o_wmf,svp/data['lev'][k],frh,1.0)
                frh=1.0
                h2o_wmf = svp*frh/data['lev'][k]
                data['H2O_DMF'][k] = h2o_wmf/(1-h2o_wmf)

            mmw = compute_mmw(h2o_wmf)

            mod_content += [fmt.format(data['lev'][k], data['T'][k],data['H'][k],mmw,data['H2O_DMF'][k],100*frh)]

        # Export Pressure and Temp for middle levels (250 to 10 mbar)
        # which have no SHum reanalysis
        ptop = data['lev'][k] # Top pressure level
        frh_top = frh  # remember the FRH at the top (300 mbar) level

        for k in range(len(data['H2O_DMF']),len(data['T'])):
            zz = np.log10(data['lev'][k])  # log10[pressure]
            strat_wmf = 7.5E-06*np.exp(-0.16*zz**2)
            svp = svp_wv_over_ice(data['T'][k])
            trop_wmf = frh_top*svp/data['lev'][k]
            wt = (data['lev'][k]/ptop)**3
            avg_wmf = trop_wmf*wt + strat_wmf*(1-wt)
            avg_frh = avg_wmf*data['lev'][k]/svp
            if (avg_frh > 1.0):
                if not muted:
                    print('Replacing super-saturated H2O ',mod_path, data['lev'][k],avg_wmf,svp*avg_frh/data['lev'][k],avg_frh,1.0)
                avg_frh = 1.0
                avg_wmf = svp*avg_frh/data['lev'][k]

            mmw = compute_mmw(avg_wmf)

            mod_content += [fmt.format(data['lev'][k],data['T'][k],data['H'][k],mmw,avg_wmf/(1-avg_wmf),100*avg_frh)]

        # Get the difference between the USSA and given site temperature at 10 mbar,
        Delta_T=data['T'][16]-t_ussa[0]

        # Export the P-T profile above 10mbar
        for k in range(1,len(t_ussa)):
            Delta_T=Delta_T/2
            zz = np.log10(p_ussa[k])  # log10[pressure]
            strat_wmf = 7.5E-06*np.exp(-0.16*zz**2)
            svp = svp_wv_over_ice(data['T'][k])
            mmw = compute_mmw(strat_wmf)
            mod_content += [fmt.format(p_ussa[k],t_ussa[k]+Delta_T,z_ussa[k],mmw,strat_wmf,100*strat_wmf*p_ussa[k]/svp)]

        output_dict = dict()
        print('Warning: output dictionary for NCEP mode not implemented, will just be empty')

    else: # merra/geos mode
        surf_var_order = ['PS', 'T2M', 'H', 'MMW', 'H2O_DMF', 'RH', 'SLP', 'TROPPB', 'TROPPV', 'TROPPT', 'TROPT', 'SZA']
        prof_var_order = ['lev', 'T', 'H', 'mmw', 'H2O_DMF', 'RH', 'EPV', 'PT', 'O3']
        computed_keys = ['mmw', 'PT', 'EL']
        if func is not None:
            # want equivalent latitude before O3, CO, etc.
            prof_var_order.insert(-1, 'EL')
        if chem_vars:
            prof_var_order.append('CO')

        mod_constants.append(surf_data['TROPPB'])
        # The head of the .mod file
        constants_fmt = '{:8.3f} {:11.4e} {:7.3f} {:5.3f} {:8.3f} {:8.3f} {:8.3f}\n'
        surface_fmt = '{:9.3e}    {:7.3f}    {:7.3f}    {:7.4f}    {:9.3e}{:>6.1f}    {:9.3e}    {:9.3e}    {:9.3e}    {:9.3e}    {:7.3f}    {:7.3f}\n'
        header_names, header_units, final_data_keys, fmt = build_mod_fmt_strings(prof_var_order)

        mod_content = []

        # number of header rows, number of data columns
        nheader = 7 + len(geos_versions)
        mod_content.append('{}  {}\n'.format(nheader, len(prof_var_order)))
        # constants
        mod_content.append(constants_fmt.format(*mod_constants))
        # surface variables
        mod_content.append('Pressure  Temperature     Height     MMW        H2O      RH         SLP        TROPPB        TROPPV      TROPPT       TROPT       SZA\n')
        mod_content.append(surface_fmt.format(*[surf_data[key] for key in surf_var_order]))
        # version info
        mod_content.append(version+'\n')
        for key, version in geos_versions.items():
            mod_content.append(f'GEOS source : {key} : {version}\n')
        # profile data headers
        mod_content.append(header_units)
        mod_content.append(header_names)

        # not sure if merra needs all the filters/corrections used for ncep data?

        # Export the Pressure, Temp and SHum
        prototype_array = np.full((len(data['H2O_DMF'],)), np.nan, dtype=float)

        output_dict = dict()
        for key in final_data_keys.keys():
            output_dict[key] = prototype_array.copy()

        for k, elem in enumerate(data['H2O_DMF']):
            # will use to output the final data to the .mod file and the returned dict
            line_dict = dict()

            #############################################
            # Check for and fix non-physical quantities #
            #############################################

            svp = svp_wv_over_ice(data['T'][k])
            h2o_wmf = compute_h2o_wmf(data['H2O_DMF'][k]) # wet mole fraction of h2o

            if 300 <= data['lev'][k] <= 1000:
                # Replace H2O mole fractions that are too small
                if data['RH'][k] < 30./data['lev'][k]:
                    if not muted:
                        print('Replacing too small H2O at {:.2f} hPa; H2O_WMF={:.3e}; {:.3e}; RH={:.3f}'.format(data['lev'][k],h2o_wmf,svp/data['T'][k],data['RH'][k],1.0))
                    data['RH'][k] = 30./data['lev'][k]
                    h2o_wmf = svp*data['RH'][k]/data['lev'][k]
                    data['H2O_DMF'][k] = h2o_wmf/(1-h2o_wmf)
                    if not muted:
                        print('svp,h2o_wmf,h2o_dmf',svp,h2o_wmf,data['H2O_DMF'][k],data['RH'][k])

            # Replace H2O mole fractions that are too large (super-saturated)  GCT 2015-08-05
            if data['RH'][k] > 1.0:
                if not muted:
                    print('Replacing too large H2O at {:.2f} hPa; H2O_WMF={:.3e}; {:.3e}; RH={:.3f}'.format(data['lev'][k],h2o_wmf,svp/data['T'][k],data['RH'][k],1.0))
                data['RH'][k] = 1.0
                h2o_wmf = svp*data['RH'][k]/data['lev'][k]
                data['H2O_DMF'][k] = h2o_wmf/(1-h2o_wmf)

            for key in prof_var_order:
                if key not in computed_keys:
                    line_dict[key] = data[key][k]

            #################################
            # Calculated derived quantities #
            #################################

            line_dict['mmw'] = compute_mmw(h2o_wmf)
            # compute potential temperature
            line_dict['PT'] = data['T'][k]*(1000.0/data['lev'][k])**0.286
            if func is None:
                line_dict['EL'] = None
            else:
                # compute equivalent latitude; 1e6 converts EPV to PVU (1e-6 K . m2 / kg / s)
                line_dict['EL'] = func(line_dict['EPV']*1e6, line_dict['PT'])[0]

            for key in line_dict.keys():
                scale = mod_var_fmt_info[key]['scale']
                line_dict[key] *= scale

            mod_content += [fmt.format(**line_dict)]
            for outkey, linekey in final_data_keys.items():
                output_dict[outkey][k] = line_dict[linekey]

    output_dict['constants'] = {k: v for k, v in zip(mod_constant_names, mod_constants)}

    with open(mod_path,'w') as outfile:
        outfile.writelines(mod_content)

    if not muted:
        print(mod_path)

    return output_dict

def trilinear_interp(DATA,varlist,site_lon_360,site_lat,site_tim):
    """
    Evaluates  fout = fin(xx,yy,*,tt)
    Result is a 1-vector
    """
    INTERP_DATA = {}

    dx = DATA['lon'][1]-DATA['lon'][0]
    dy = DATA['lat'][1]-DATA['lat'][0]
    dt = DATA['time'][1]-DATA['time'][0]

    xx = (site_lon_360-DATA['lon'][0])/dx
    yy = (site_lat-DATA['lat'][0])/dy
    tt = (site_tim-DATA['time'][0])/dt

    nxx =  len(DATA['lon'])
    nyy =  len(DATA['lat'])
    ntt =  len(DATA['time'])

    index_xx = int(xx)
    ixpomnxx = (index_xx+1) % nxx
    fr_xx = xx-index_xx

    index_yy = int(yy)
    if index_yy > nyy-2:
        index_yy = nyy-2  #  avoid array-bound violation at SP
    fr_yy = yy-index_yy

    index_tt = int(tt)

    if index_tt < 0:
        index_tt = 0          # Prevent Jan 1 problem
    if index_tt+1 > ntt-1:
        index_tt = ntt-2  # Prevent Dec 31 problem

    fr_tt=tt-index_tt  #  Should be between 0 and 1 when interpolating in time

    if (fr_tt < -1) or (fr_tt > 2):
        print('Excessive time extrapolation:',fr_tt,' time-steps   =',fr_tt*dt,' days')
        print(' tt= ',tt,'  index_tt=',index_tt,'  fr_tt=',fr_tt)
        print('input file does not cover the full range of dates')
        print('site_tim',site_tim)
        print('tim_XX',DATA['time'])
        input() # will hold the program until something is typed in commandline

    if (fr_xx < 0) or (fr_xx > 1):
        print('Excessive longitude extrapolation:',fr_xx,' steps   =',fr_xx*dx,' deg')
        print(' xx= ',xx,'  index_xx=',index_xx,'  fr_xx=',fr_xx)
        print('input file does not cover the full range of longitudes')
        input() # will hold the program until something is typed in commandline

    if (fr_yy < 0) or (fr_yy > 1):
        print('Excessive latitude extrapolation:',fr_yy-1,' steps   =',(fr_yy-1)*dy,' deg')
        print(' yy= ',yy,'  index_yy=',index_yy,'  fr_yy=',fr_yy)
        print('input file does not cover the full range of latitudes')
        input() # will hold the program until something is typed in commandline

    if (fr_tt < 0) or (fr_tt > 1):
        print(' Warning: time extrapolation of ',fr_tt,' time-steps')
    if (fr_xx < 0) or (fr_xx > 1):
        print(' Warning: longitude extrapolation of ',fr_xx,' steps')
    if (fr_yy < 0) or (fr_yy > 1):
        print(' Warning: latitude extrapolation of ',fr_yy,' steps')

    for varname in varlist:

        fin = DATA[varname]

        if fin.ndim==4:
            fout =	((fin[index_tt,:,index_yy,index_xx]*(1.0-fr_xx) \
                        + fin[index_tt,:,index_yy,ixpomnxx]*fr_xx)*(1.0-fr_yy) \
                       + (fin[index_tt,:,index_yy+1,index_xx]*(1.0-fr_xx) \
                          + fin[index_tt,:,index_yy+1,ixpomnxx]*fr_xx)*fr_yy)*(1.0-fr_tt) \
                      + ((fin[index_tt+1,:,index_yy,index_xx]*(1.0-fr_xx) \
                          + fin[index_tt+1,:,index_yy,ixpomnxx]*fr_xx)*(1.0-fr_yy) \
                         + (fin[index_tt+1,:,index_yy+1,index_xx]*(1.0-fr_xx) \
                            + fin[index_tt+1,:,index_yy+1,ixpomnxx]*fr_xx)*fr_yy)*fr_tt
        elif fin.ndim==3: # for data that do not have the vertical dimension
            fout =	((fin[index_tt,index_yy,index_xx]*(1.0-fr_xx) \
                        + fin[index_tt,index_yy,ixpomnxx]*fr_xx)*(1.0-fr_yy) \
                       + (fin[index_tt,index_yy+1,index_xx]*(1.0-fr_xx) \
                          + fin[index_tt,index_yy+1,ixpomnxx]*fr_xx)*fr_yy)*(1.0-fr_tt) \
                      + ((fin[index_tt+1,index_yy,index_xx]*(1.0-fr_xx) \
                          + fin[index_tt+1,index_yy,ixpomnxx]*fr_xx)*(1.0-fr_yy) \
                         + (fin[index_tt+1,index_yy+1,index_xx]*(1.0-fr_xx) \
                            + fin[index_tt+1,index_yy+1,ixpomnxx]*fr_xx)*fr_yy)*fr_tt
        else:
            print('Data has unexpected dimensions, ndim =',fin.ndim)
            sys.exit()

        INTERP_DATA[varname] = fout*DATA['scale_factor_'+varname] + DATA['add_offset_'+varname]

    return INTERP_DATA

def read_data(dataset, varlist, lat_lon_box=0):
    """
    for ncep files "dataset" is the full path to the netcdf file

    for merra files "dataset" is a pydap.model.DatasetType object
    """
    DATA = {}

    opendap = type(dataset)!=netCDF4._netCDF4.Dataset

    if lat_lon_box == 0: # ncep mode

        varlist += ['level','lat','lon','time']

        for varname in varlist:
            DATA[varname] = dataset[varname][:]

            for attribute in ['add_offset','scale_factor']:
                try:
                    DATA['add_offset_'+varname] = dataset[varname].getncattr('add_offset')
                except:
                    DATA['add_offset_'+varname] = 0.0
                    DATA['scale_factor_'+varname] = 1.0

    else: # merra/geos5 mode

        min_lat_ID, max_lat_ID, min_lon_ID, max_lon_ID = lat_lon_box

        DATA['lat'] = dataset['lat'][min_lat_ID:max_lat_ID] 	# Read in variable 'lat'
        DATA['lon'] = dataset['lon'][min_lon_ID:max_lon_ID] 	# Read in variable 'lon'
        DATA['time'] = dataset['time'][:]	# Read in variable 'time'

        try:
            dataset['lev']
        except:
            pass
        else:
            if dataset['lev'].shape[0] == 72:
                DATA['lev'] = dataset['PL'][:,:,min_lat_ID:max_lat_ID,min_lon_ID:max_lon_ID]	# the merra 72 mid level pressures are not fixed
            elif dataset['lev'].shape[0] == 42:
                DATA['lev'] = dataset['lev'][:]	# the 42 levels data is on a fixed pressure grid
            else:
                DATA['lev'] = dataset['PS'][:,min_lat_ID:max_lat_ID,min_lon_ID:max_lon_ID] # surface data doesn't have a 'lev' variable

        if opendap:
            for varname in ['time','lev','lat','lon']:
                try:
                    DATA[varname] = DATA[varname].data
                except KeyError as IndexError:
                    pass

        # get longitudes as 0 -> 360 instead of -180 -> 180, needed for trilinear_interp
        for i,elem in enumerate(DATA['lon']):
            if elem < 0:
                DATA['lon'][i] = elem + 360.0

        for varname in varlist:

            if dataset[varname].ndim == 4:
                DATA[varname] = dataset[varname][:,:,min_lat_ID:max_lat_ID,min_lon_ID:max_lon_ID] 	# Read in variable varname
            else:
                DATA[varname] = dataset[varname][:,min_lat_ID:max_lat_ID,min_lon_ID:max_lon_ID] 	# Read in variable varname
            if opendap or ('Masked' in str(type(DATA[varname]))):
                DATA[varname] = DATA[varname].data

            for attribute in ['add_offset','scale_factor']:
                try:
                    DATA[attribute+'_'+varname] = dataset[varname].getncattr(attribute)
                except:
                    DATA['add_offset_'+varname] = 0.0
                    DATA['scale_factor_'+varname] = 1.0

            print(varname, DATA[varname].shape)

    time_units = dataset['time'].units  # string containing definition of time units

    # two lines to parse the date (no longer need to worry about before/after 2014)
    date_list = re.findall(r"[\w]+",time_units.split(' since ')[1])
    common_date_format = '{:0>4}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}'.format(*date_list)

    start_date = datetime.strptime(common_date_format,'%Y-%m-%d %H:%M:%S')
    with warnings.catch_warnings(): # silence the astropy erfa warning that triggers for dates prior to UTC (datasets can have epoch as 1800-01-01)
        warnings.simplefilter("ignore")
        astropy_start_date = Time(start_date)

    DATA['julday0'] = astropy_start_date.jd # gives same results as IDL's JULDAY function that is used in mod_maker.pro

    return DATA

def nearest(array,value):
    """
    return index of element in array that is closest to value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def querry_indices(dataset,site_lat,site_lon_180,box_lat_half_width,box_lon_half_width):
    """
    Get the lat/lon IDs of the grid cell that contain the site lat/lon
    """

    # read the latitudes and longitudes from the merra file
    if type(dataset)==netCDF4._netCDF4.Dataset:
        merra_lon = dataset['lon'][:]
        merra_lat = dataset['lat'][:]
    elif type(dataset)==list:
        merra_lat = dataset[0]
        merra_lon = dataset[1]
    else: # for opendap datasets
        merra_lon = dataset['lon'][:].data
        merra_lat = dataset['lat'][:].data


    nearest_lat_ID = nearest(merra_lat,site_lat)
    if (site_lat-merra_lat[nearest_lat_ID])>0: # if exact same latitude, the second latitude for the grid square will be south
        min_lat_ID = nearest_lat_ID
        max_lat_ID = nearest_lat_ID+1
    else:
        min_lat_ID = nearest_lat_ID-1
        max_lat_ID = nearest_lat_ID

    nearest_lon_ID = nearest(merra_lon,site_lon_180)
    if (site_lon_180-merra_lon[nearest_lon_ID])>0: # if exact same longitude, the second longitude for the grid square will be west
        min_lon_ID = nearest_lon_ID
        if nearest_lon_ID == len(merra_lon)-1: # positive longitude edge case
            max_lon_ID = 0
        else:
            max_lon_ID = nearest_lon_ID+1
    else:
        # no need for a negative longitude edge case because negative indices go in reverse from the end of the list e.g. for a list with N elements: list[-1] = list[N-1]
        min_lon_ID = nearest_lon_ID-1
        max_lon_ID = nearest_lon_ID

    if not merra_lat[min_lat_ID]<site_lat<merra_lat[max_lat_ID]:
        print('min_lat','site_lat','max_lat',merra_lat[min_lat_ID],site_lat,merra_lat[max_lat_ID])
    if not merra_lon[min_lon_ID]<site_lon_180<merra_lon[max_lon_ID]:
        print('min_lon','site_lon','max_lon',merra_lon[min_lon_ID],site_lon_180,merra_lon[max_lon_ID])

    IDs = [min_lat_ID, max_lat_ID, min_lon_ID, max_lon_ID]

    return IDs
# ncep has geopotential height profiles, not merra(?, only surface), so I need to convert geometric heights to geopotential heights
# the idl code uses a fixed radius for the radius of earth (6378.137 km), below the gravity routine of gsetup is used
# also the surface geopotential height of merra is in units of m2 s-2, so it must be divided by surface gravity

def read_merradap(username,password,mode,site_lon_180,site_lat,gravity_at_lat,date,end_date,time_step,varlist,surf_varlist,muted):
    """
    Read MERRA2 data via opendap.

    This has to connect to the daily netcdf files, and then concatenate the subsetted datasets.

    This is EXTREMELY slow, should probably make that use separate to generate local files, and then use to files in mod_maker
    """
    DATA = {}
    SURF_DATA = {}

    if '42' in mode:
        letter = 'P'
    elif '72' in mode:
        letter = 'V'
        varlist += ['PL']

    old_UTC_date = ''
    urllist = []
    surface_urllist = []
    if not muted:
        print('\n\t-Making lists of URLs')
    while date < end_date:
        UTC_date = date + timedelta(hours = -site_lon_180/15.0) # merra times are in UTC, so the date may be different than the local date, make sure to use the UTC date to querry the file
        if (UTC_date.strftime('%Y%m%d') != old_UTC_date):
            if not muted:
                print('\t\t',UTC_date.strftime('%Y-%m-%d'))
            urllist += ['https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2I3N{}ASM.5.12.4/{:0>4}/{:0>2}/MERRA2_400.inst3_3d_asm_N{}.{:0>4}{:0>2}{:0>2}.nc4'.format(letter,UTC_date.year,UTC_date.month,letter.lower(),UTC_date.year,UTC_date.month,UTC_date.day)]
            surface_urllist += ['https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2I1NXASM.5.12.4/{:0>4}/{:0>2}/MERRA2_400.inst1_2d_asm_Nx.{:0>4}{:0>2}{:0>2}.nc4'.format(UTC_date.year,UTC_date.month,UTC_date.year,UTC_date.month,UTC_date.day)]
            if old_UTC_date == '':
                session = setup_session(username,password,check_url=urllist[0]) # just need to setup the authentication session once
        old_UTC_date = UTC_date.strftime('%Y%m%d')
        date = date + time_step

    # multi-level data
    if not muted:
        print('\nNow doing multi-level data')
        print('\t-Connecting to datasets ...')
    store_list = [xarray.backends.PydapDataStore.open(url,session) for url in urllist]
    dataset_list = [xarray.open_dataset(store) for store in store_list]
    if not muted:
        print('\t-Datasets opened')
    min_lat_ID,max_lat_ID,min_lon_ID,max_lon_ID = querry_indices(dataset_list[0],site_lat,site_lon_180,2.5,2.5) # just need to get the lat/lon box once
    subsest_dataset_list = [dataset[{'lat':list(range(min_lat_ID,max_lat_ID+1)),'lon':list(range(min_lon_ID,max_lon_ID+1))}] for dataset in dataset_list]
    if not muted:
        print('\t-Datasets subsetted')
        print('\t-Merging datasets (time consuming)')
    merged_dataset = xarray.concat(subsest_dataset_list,'time')
    merged_dataset = merged_dataset.fillna(1e15)

    # single-level data
    if not muted:
        print('\nNow doing single-level data')
        print('\t-Connecting to datasets ...')
    surface_store_list = [xarray.backends.PydapDataStore.open(url,session) for url in surface_urllist]
    surface_dataset_list = [xarray.open_dataset(store) for store in surface_store_list]
    if not muted:
        print('\t-Datasets opened')
    subsest_surface_dataset_list = [dataset[{'lat':list(range(min_lat_ID,max_lat_ID+1)),'lon':list(range(min_lon_ID,max_lon_ID+1))}] for dataset in surface_dataset_list]
    if not muted:
        print('\t-Datasets subsetted')
        print('\t-Merging datasets (time consuming)')
    merged_surface_dataset = xarray.concat(subsest_surface_dataset_list,'time')
    merged_surface_dataset = merged_surface_dataset.fillna(1e15)

    for varname in varlist:
        DATA[varname] = merged_dataset[varname].data
        DATA['add_offset_'+varname] = 0.0
        DATA['scale_factor_'+varname] = 1.0
    for varname in surf_varlist:
        SURF_DATA[varname] = merged_surface_dataset[varname].data
        SURF_DATA['add_offset_'+varname] = 0.0
        SURF_DATA['scale_factor_'+varname] = 1.0

    for varname in ['time','lat','lon']:
        DATA[varname] = merged_dataset[varname].data
        SURF_DATA[varname] = merged_surface_dataset[varname].data
    DATA['lev'] = merged_dataset['lev'].data

    DATA['PHIS'] = DATA['PHIS'] / gravity_at_lat # convert from m2 s-2 to m

    delta_time = [(i-DATA['time'][0]).astype('timedelta64[h]') / np.timedelta64(1,'h') for i in DATA['time']] # hours since base time
    surf_delta_time = [(i-SURF_DATA['time'][0]).astype('timedelta64[h]') / np.timedelta64(1,'h') for i in SURF_DATA['time']] # hours since base time

    DATA['julday0'] = Time(str(DATA['time'][0]),format="isot").jd
    SURF_DATA['julday0'] = Time(str(SURF_DATA['time'][0]),format="isot").jd

    DATA['time'] = delta_time
    SURF_DATA['time'] = surf_delta_time

    # get longitudes as 0 -> 360 instead of -180 -> 180, needed for trilinear_interp
    for i,elem in enumerate(DATA['lon']):
        if elem < 0:
            DATA['lon'][i] = elem + 360.0
    for i,elem in enumerate(SURF_DATA['lon']):
        if elem < 0:
            SURF_DATA['lon'][i] = elem + 360.0

    return DATA, SURF_DATA

def read_ncep(ncdf_path,year):
    """
    Read data from yearly NCEP netcdf files and return it in one dictionary
    """

    # path to the netcdf files
    ncdf_AT_file = os.path.join(ncdf_path,'.'.join(['air','{:0>4}'.format(year),'nc']))
    ncdf_GH_file = os.path.join(ncdf_path,'.'.join(['hgt','{:0>4}'.format(year),'nc']))
    ncdf_SH_file = os.path.join(ncdf_path,'.'.join(['shum','{:0>4}'.format(year),'nc']))

    print('Read global',year,'NCEP data ...')
    # Air Temperature
    DATA = read_data(netCDF4.Dataset(ncdf_AT_file,'r'), ['air'])
    if len(DATA['air']) < 17:
        print('Need 17 levels of AT data: found only ',len(lev_AT))

    # Specific Humidity
    SHUM_DATA = read_data(netCDF4.Dataset(ncdf_SH_file,'r'), ['shum'])
    if len(SHUM_DATA['level']) <  8:
        print('Need  8 levels of SH data: found only ',len(lev_SH))

    if list(SHUM_DATA['level'])!=list(DATA['level'][:len(SHUM_DATA['level'])]):
        print('Warning: air and shum do not share the same lower pressure levels')

    DATA.update(SHUM_DATA)

    # Geopotential Height
    GH_DATA = read_data(netCDF4.Dataset(ncdf_GH_file,'r'), ['hgt'])
    if len(GH_DATA['level']) < 17:
        print('Need 17 levels of GH data: found only ',len(lev_GH))

    DATA.update(GH_DATA)

    for key in DATA:
        if 'air' in key:
            DATA[key.replace('air','T')] = DATA[key]
            del DATA[key]
        if 'hgt' in key:
            DATA[key.replace('hgt','H')] = DATA[key]
            del DATA[key]
        if 'shum' in key:
            DATA[key.replace('shum','QV')] = DATA[key]
            del DATA[key]

    DATA['lev'] = DATA['level']
    del DATA['level']

    return DATA

def read_global(ncdf_path,mode,site_lat,site_lon_180,gravity_at_lat,varlist,surf_varlist,muted):
    """
    Read data from GEOS5 and MERRA2 datasets

    This assumes those are saved locally in GGGPATH/ncdf with two files per dataset (inst3_3d_asm_np and inst3_2d_asm_nx)
    """

    key_dict = {'merraglob':'MERRA','fpglob':'_fp_','fpitglob':'_fpit_'}

    # assumes only one file with all the data exists in the GGGPATH/ncdf folder
    # path to the netcdf file
    ncdf_list = [i for i in os.listdir(ncdf_path) if key_dict[mode] in i]

    ncdf_file = [i for i in ncdf_list if '3d' in i][0]
    dataset = netCDF4.Dataset(os.path.join(ncdf_path,ncdf_file),'r')

    surf_file = [i for i in ncdf_list if '2d' in i][0]
    surface_dataset = netCDF4.Dataset(os.path.join(ncdf_path,surf_file),'r')

    if not muted:
        print(ncdf_file)
        print(surf_file)

    # get the min/max lat-lon indices of merra lat-lon that lies within a given box.
    # geos5-fp has a smaller grid than merra2 amd geos5-fp-it
    box_lat_half_width = float(dataset.LatitudeResolution)
    box_lon_half_width = float(dataset.LongitudeResolution)
    lat_lon_box = querry_indices(dataset,site_lat,site_lon_180,box_lat_half_width,box_lon_half_width)

    # multi-level data
    if not muted:
        print('Read global',mode,'multi-level data ...')
    DATA = read_data(dataset,varlist,lat_lon_box)
    DATA['PHIS'] = DATA['PHIS'] / gravity_at_lat # convert from m2 s-2 to m

    # single level data
    if not muted:
        print('Read global',mode,'single-level data ...')
    SURF_DATA = read_data(surface_dataset,surf_varlist,lat_lon_box)

    # merra/geos time is minutes since base time, need to convert to hours
    DATA['time'] = DATA['time'] / 60.0
    SURF_DATA['time'] = SURF_DATA['time'] / 60.0

    return DATA,SURF_DATA

def equivalent_latitude_functions(ncdf_path,mode,start=None,end=None,muted=False):
    """
    Inputs:
        - dataset: global dataset for fp, fp-it, or merra

    Outputs:
        - func_dict: list of functions, at each dataset time, to get equivalent latitude for a given PV and PT

    e.g. for the ith time, to get equivalent latitude for PV and PT: eq_lat = func_dict[i](PV,PT)

    takes ~ 3-4 minutes per date
    """

    key_dict = {'merraglob':'MERRA','fpglob':'_fp_','fpitglob':'_fpit_'}

    ncdf_list = [i for i in os.listdir(ncdf_path) if key_dict[mode] in i]

    ncdf_file = [i for i in ncdf_list if '3d' in i][0]
    dataset = netCDF4.Dataset(os.path.join(ncdf_path,ncdf_file),'r')

    if not muted:
        print('\nGenerating equivalent latitude functions ...')

    lat = dataset['lat'][:]
    lat[180] = 0.0
    lon = dataset['lon'][:]
    pres = dataset['lev'][:]
    date = netCDF4.num2date(dataset['time'][:],dataset['time'].units)

    EPV = (dataset['EPV'][0]*1e6).data

    ntim,nlev,nlat,nlon = [dataset.dimensions[i].size for i in dataset.dimensions]
    if not muted:
        print('time,lev,lat,lon',(ntim,nlev,nlat,nlon))

    select_dates = date[date>=start]
    select_dates = select_dates[select_dates<=end]
    date_inds = [np.where(date==np.datetime64(i))[0][0] for i in select_dates]
    ntim = len(date_inds)

    # Get the area of each grid cell
    lat_res = float(dataset.LatitudeResolution)
    lon_res = float(dataset.LongitudeResolution)

    lon_res = np.radians(lon_res)
    lat_half_res = 0.5*lat_res

    area = np.zeros([nlat,nlon])
    earth_area = 0
    for j in range(nlat):
        Slat = lat[j]-lat_half_res
        Nlat = lat[j]+lat_half_res

        Slat = np.radians(Slat)
        Nlat = np.radians(Nlat)
        for i in range(nlon):
            area[j,i] = lon_res*np.abs(sin(Slat)-sin(Nlat))

    earth_area = np.sum(area)

    if abs(np.sum(area)-earth_area)>0.0001:
        area = area*4*np.pi/earth_area

    # used to compute potential temperature PT = T*(P0/P)**0.286; this is the (P0/P)**0.286 which is computed once here instead of many times in the dates loop
    coeff = (1000.0/pres)**0.286
    coeff_mat = np.zeros([nlev,nlat,nlon])
    for i in range(nlat):
        for j in range(nlon):
            coeff_mat[:,i,j] = coeff

    nmin = [0.125]
    func_dict = {} # dictionary mapping each time to the corresponding equivalent latitude function
    total_start = time.time()
    for t in date_inds: # loop over dates
        start = time.time()
        if not muted:
            sys.stdout.write('\r\tDate {:4d} / {:4d} ; finish in about {:.1f} minutes'.format(t+1,ntim,np.mean(nmin)*(ntim-t)))
            sys.stdout.flush()

        # Compute potential temperature
        PT = (dataset['T'][t]*coeff_mat).data

        EPV = (dataset['EPV'][t].data)*1e6 # Potential vorticity in PVU = 1e-6 K . m2 / kg / s

        # Get rid of fill values, this fills the bottom of profiles with the first valid value
        PT[PT>1e4]=np.nan
        EPV[EPV>1e8]=np.nan
        for i in range(nlat):
            pd.DataFrame(PT[:,i,:]).fillna(method='bfill',axis=0,inplace=True)
            pd.DataFrame(EPV[:,i,:]).fillna(method='bfill',axis=0,inplace=True)

        # Define a fixed potential temperature grid, with increasing spacing
        #fixed_PT = np.arange(np.min(PT),np.max(PT),20) # fixed potential temperature grid
        fixed_PT = sorted(list(set(list(range(int(np.min(PT)),300,2))+list(range(300,350,5))+list(range(350,500,10))+list(range(500,750,20))+list(range(750,1000,30))+list(range(1000,int(np.max(PT)),100)))))
        new_nlev = len(fixed_PT)

        # Get PV on the fixed PT levels
        new_EPV = np.zeros([new_nlev,nlat,nlon])
        for i in range(nlat):
            for j in range(nlon):
                new_EPV[:,i,j] = np.interp(fixed_PT,PT[:,i,j],EPV[:,i,j])

        # Compute equivalent latitudes
        EL = np.zeros([new_nlev,100])
        EPV_thresh = np.zeros([new_nlev,100])
        for k in range(new_nlev): # loop over potential temperature levels
            maxPV = np.max(new_EPV[k]) # global max PV
            minPV = np.min(new_EPV[k]) # global min PV

            # define 100 PV values between the min and max PV
            EPV_thresh[k] = np.linspace(minPV,maxPV,100)

            for l,thresh in enumerate(EPV_thresh[k]):
                area_total = np.sum(area[new_EPV[k]>=thresh])
                EL[k,l] = arcsin(1-area_total/(2*np.pi))*90.0*2/np.pi

        # Define a fixed potentital vorticity grid, with increasing spacing away from 0
        #fixed_PV = np.arange(np.min(EPV_thresh),np.max(EPV_thresh)+10,10) # fixed PV grid
        fixed_PV = sorted(list(set(list(range(int(np.min(EPV_thresh)-50),-1000,50))+list(range(-1000,-500,20))+list(range(-500,-100,10))+list(range(-100,-10,1))+list(np.arange(-10,-1,0.1))+list(np.arange(-1,1,0.01))+list(np.arange(1,10,0.1))+list(range(10,100,1))+list(range(100,500,10))+list(range(500,1000,20))+list(range(1000,int(np.max(EPV_thresh)+50),50)))))
        if 0.0 not in fixed_PV: # need a point at 0.0 for the interpolations to work better
            fixed_PV = np.sort(np.append(fixed_PV,0.0))

        # Generate interpolating function to get EL for a given PV and PT
        interp_EL = np.zeros([new_nlev,len(fixed_PV)])
        for k in range(new_nlev):
            interp_EL[k] = np.interp(fixed_PV,EPV_thresh[k],EL[k])

        func_dict[date[t]] = interp2d(fixed_PV,fixed_PT,interp_EL)

        end = time.time()
        nmin.append(int(end-start)/60.0)

    actual_time = (time.time()-total_start)/60.0
    predicted_time = 0.125*ntim
    if not muted:
        print('\nPredicted to finish in {:.1f} minutes\nActually finished in {:.1f} minutes'.format(predicted_time,actual_time))

    dataset.close()

    return func_dict


def _add_common_args(parser):
    def english_join(conj, seq):
        seq = ['"{}"' for el in seq]
        if len(seq) == 1:
            return seq[0]

        seq[-1] = '{} {}'.format(conj, seq[-1])
        if len(seq) == 2:
            return ' '.join(seq)
        else:
            return ', '.join(seq)

    parser.add_argument('met_path',
                        help='Path to the meteorology FP(-IT) netCDF files. Must be directory with subdirectories '
                             'Nx and Np or Nv, containing surface and profile paths respectively.')
    parser.add_argument('--chem-path', default=None, help='Path to the chemistry FP(-IT) files. Must be a directory '
                                                          'with subdirectory Nv containing the chm netCDF files. If '
                                                          'not given, it is assumed that these files are stored with '
                                                          'the regular met data.')
    parser.add_argument('-s', '--save-path',
                        help='Location to save .mod files to. Subdirectories organized by met type, '
                             'site, and vertical/slant .mod files will be created. If not given, '
                             'will attempt to save files under $GGGPATH/models/gnd')
    parser.add_argument('--keep-latlon-prec', action='store_true',
                        help='Retain lat/lon to 2 decimals in the .mod file names')
    parser.add_argument('--save-in-local', action='store_false', dest='save_in_utc',
                        help='Use local time in .mod file name, instead of UTC')
    parser.add_argument('-c', '--include-chem', action='store_true', dest='include_chm',
                        help='Include chemistry variables (CO) in the .mod files. Note that this is necessary if the '
                             '.mod files are to be used to generate GGG2019+ .vmr files.')
    parser.add_argument('-q', '--quiet', dest='muted', action='store_true', help='Suppress log output to command line.')
    parser.add_argument('--slant', action='store_true', help='Generate slant .mod files, in addition to vertical .mod '
                                                             'files. Not compatible with --flat-outdir.')
    parser.add_argument('--mode', choices=_old_modmaker_modes + _new_modmaker_modes, default=_default_mode,
                        help='If one of {old} is chosen, mod_maker uses the old code with time interpolation. If one '
                             'of {new} is chosen, it uses the new code that generates 8x mod files per day. {fixedp} '
                             'expect fixed pressure level GEOS files; {eta} expects native 72 level GEOS files.'
                        .format(old=', '.join(_old_modmaker_modes), new=', '.join(_new_modmaker_modes),
                                fixedp=english_join('or', _new_fixedp_modes), eta=english_join('or', _new_native_modes)))
    parser.add_argument('-f', '--flat-outdir', action='store_true',
                        help='Write the .mod files directly to the specified output directory, rather than organizing '
                             'by product/site/vertical or slant.')


def parse_args(parser=None):
    """
    parse commandline arguments (see code header or README.md)
    """
    # For Py3 compatibility, convert keys iterator into an explicit list.
    valid_site_ids = list(site_dict.keys())

    description = 'Generate TCCON .mod files'
    if parser is None:
        parser = argparse.ArgumentParser(description=description)
        am_i_main = True
    else:
        parser.description = description
        am_i_main = False

    parser.add_argument('date_range', type=mod_utils.parse_date_range,
                        help='The range of dates to generate .mod files for. May be given as YYYYMMDD-YYYYMMDD, or '
                             'YYYYMMDD_HH-YYYYMMDD_HH, where the ending date is exclusive. A single date may be given, '
                             'in which case the ending date is assumed to be one day later.')
    parser.add_argument('--alt', type=float, help='Site altitude in meters, if defining a custom site.')
    parser.add_argument('--lon', type=float, help='Site longitude in degrees east, if defining a custom site. Values '
                                                  'should be positive; i.e. 90 W should be given as 270.')
    parser.add_argument('--lat', type=float, help='Site latitude, in degrees (north = positive, south = negative).')
    parser.add_argument('--site', dest='site_abbrv', choices=valid_site_ids, help='Two-letter site abbreviation. '
                                                                                  'Providing this will produce .mod '
                                                                                  'files only for that site.')
    _add_common_args(parser)

    if am_i_main:
        arg_dict = vars(parser.parse_args())

        # Error checking and some splitting of variables
        arg_dict['start_date'], arg_dict['end_date'] = arg_dict['date_range']
        if arg_dict['end_date'] < arg_dict['start_date']:
            shell_error('Error: end of date range (if given) must be after the start')
        if not os.path.exists(arg_dict['met_path']):
            shell_error('Given GEOS data path ({}) does not exist'.format(arg_dict['met_path']))

        return arg_dict
    else:
        parser.set_defaults(driver_fxn=driver)


def parse_runlog_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
        am_i_main = True
    else:
        am_i_main = False
    parser.description = 'Generate .mod files for all spectra in a given runlog'
    parser.add_argument('runlog', help='Path to the runlog file. .mod files will be created for each unique lat/lon/'
                                       'date combination in the runlog (with date rounded to the nearest GEOS time).')
    parser.add_argument('--site', dest='site_abbrv', default=None,
                        help='The two letter site ID to use when organizing the output .mod files into folders. The '
                             'default behavior is to take the first two letters of each spectrum as the site ID. Pass '
                             'a single ID with this option to override that. Currently there is no way to pass '
                             'multiple IDs from the command line.')
    parser.add_argument('--first-date', default='2000-01-01',
                        help='First date to generate .mod files for. Default is %(default)s, due to the availability '
                             'of GEOS-FPIT data.')

    _add_common_args(parser)
    if am_i_main:
        arg_dict = vars(parser.parse_args())
        if not os.path.exists(arg_dict['met_path']):
            shell_error('Given GEOS data path ({}) does not exist'.format(arg_dict['met_path']))
        return arg_dict
    else:
        parser.set_defaults(driver_fxn=runlog_driver)


def parse_vmr_args(parser, backend=parse_args):
    """

    :param parser:
    :type parser: :class:`argparse.ArgumentParser`
    :return:
    """
    # configure all the options
    backend(parser)

    parser.set_defaults(mode='fpit-eta', include_chm=True)
    parser.epilog = "Defaults have been set to produce the right format of file for TCCON GGG2020 use " \
                    "(--mode=fpit-eta, --include-chem)."


def GEOS_files(GEOS_path, start_date, end_date, chm=False):

    # all GEOS5-FPIT Np/Nx files and their dates. Use 'glob' to avoid listing other files (e.g. the download link list)
    # in the directory. Whether glob.glob() and os.listdif() returns a sorted list is platform dependendent. Since the
    # main mod_maker logic depends on the profile and surface file lists being in the same order, we need to sort them.
    ncdf_list = sorted(glob.glob(os.path.join(GEOS_path, 'GEOS*.nc4')))
    if chm:
        ncdf_list = np.array([f for f in ncdf_list if 'chm' in os.path.basename(f)])
    else:
        ncdf_list = np.array([f for f in ncdf_list if 'chm' not in os.path.basename(f)])
    ncdf_basenames = [os.path.basename(f) for f in ncdf_list]
    ncdf_dates = np.array([mod_utils.datetime_from_geos_filename(elem) for elem in ncdf_basenames])

    # just the one between the 'start_date' and 'end_date' dates
    select_files = ncdf_list[(ncdf_dates>=start_date) & (ncdf_dates<end_date)]
    select_dates = ncdf_dates[(ncdf_dates>=start_date) & (ncdf_dates<end_date)]

    if len(select_dates) == 0:
        raise IOError('No GEOS files between {} and {}'.format(start_date,end_date))

    return select_files,select_dates


def equivalent_latitude_functions_geos(GEOS_path, start_date=None, end_date=None, muted=False, **kwargs):
    """
    Inputs:
        - GEOS_path: full path to the folder containing GEOS5-fpit files, an 'Np' folder with 3-hourly files is expected under that path
        - start_date: datetime object
        - end_date: datetime object (exclusive)
        - muted: if True there will be no print statements
    Outputs:
        - func_dict: list of functions, at each dataset time, to get equivalent latitude for a given PV and PT

    e.g. for the ith time, to get equivalent latitude for PV and PT: eq_lat = func_dict[i](PV,PT)

    takes ~ 3-4 minutes per date
    """

    GEOS_path = os.path.join(GEOS_path,'Np')

    select_files, select_dates = GEOS_files(GEOS_path,start_date,end_date)

    if not muted:
        print('\nGenerating equivalent latitude functions for {} times'.format(len(select_dates)))

    return equivalent_latitude_functions_from_geos_files(select_files, select_dates, muted=muted)


def equivalent_latitude_functions_native_geos(GEOS_path, start_date=None, end_date=None, muted=False, **kwargs):
    """
    Generate equivalent latitude interpolators from native (72 eta level) GEOS files.

    :param GEOS_path: full path to the folder containing GEOS5-fpit native level files, an 'Nv' folder with 3-hourly
     files is expected under that path.
    :type GEOS_path: str

    :param start_date: the first datetime to generate eq. lat. interpolators for
    :type start_date: datetime-like

    :param end_date: the last datetime to generate eq. lat. interpolators for
    :type end_date: datetime-like

    :param muted: set to ``True`` to disable logging to the console.
    :type muted: bool

    :param kwargs: unused, swallows extra keyword arguments.

    :return: dictionary of equivalent latitude intepolators, the keys will be the datetime of the interpolators
    :rtype: dict
    """
    GEOS_path = os.path.join(GEOS_path, 'Nv')
    select_files, select_dates = GEOS_files(GEOS_path, start_date, end_date)

    if not muted:
        print('\nGenerating equivalent latitude functions for {} native GEOS files'.format(len(select_dates)))

    return equivalent_latitude_functions_from_native_geos_files(select_files, select_dates, muted=muted)


def equivalent_latitude_functions_from_geos_files(geos_np_files, geos_dates, muted=False):
    # Use any file for stuff that is the same in all files
    with netCDF4.Dataset(geos_np_files[0], 'r') as dataset:
        lat = dataset['lat'][:]
        lat[180] = 0.0
        lon = dataset['lon'][:]
        pres = dataset['lev'][:]
        ntim, nlev, nlat, nlon = dataset['EPV'].shape

        # Get the area of each grid cell
        lat_res = float(dataset.LatitudeResolution)
        lon_res = float(dataset.LongitudeResolution)

    area = mod_utils.calculate_area(lat, lon, lat_res, lon_res, muted=muted)

    # pre-compute pressure coefficients for calculating potential temperature, this is the (Po/P)^(R/Cp) term
    # TODO: test replacement with mod_utils potential temperature function
    coeff = (1000.0 / pres) ** 0.286
    coeff_mat = np.zeros([nlev, nlat, nlon])
    for i in range(nlat):
        for j in range(nlon):
            coeff_mat[:, i, j] = coeff

    ntim = len(geos_dates)
    nmin = [0.125]
    func_dict = {}
    total_start = time.time()
    for date_ID, date in enumerate(geos_dates):

        start = time.time()
        if not muted:
            sys.stdout.write('\r\tDate {:4d} / {:4d} ; finish in about {:.1f} minutes'.format(date_ID + 1, ntim,
                                                                                              np.mean(nmin) * (
                                                                                                      ntim - date_ID)))
            sys.stdout.flush()

        with netCDF4.Dataset(geos_np_files[date_ID]) as dataset:
            PT = (dataset['T'][0] * coeff_mat).data  # Compute potential temperature
            EPV = (dataset['EPV'][0].data) * 1e6  # Potential vorticity in PVU = 1e-6 K . m2 / kg / s

        func_dict[date] = mod_utils.calculate_eq_lat(EPV, PT, area)

        end = time.time()
        nmin.append(int(end - start) / 60.0)
    # end of loop over dates

    actual_time = (time.time() - total_start) / 60.0
    predicted_time = 0.125 * ntim
    if not muted:
        print('\nPredicted to finish in {:.1f} minutes\nActually finished in {:.1f} minutes'.format(predicted_time,
                                                                                                    actual_time))

    return func_dict


def equivalent_latitude_functions_from_native_geos_files(geos_nv_files, geos_dates, muted=False):
    """
    Generate equivalent latitude interpolators from native GEOS FP(-IT) files

    :param geos_nv_files: a list of the native GEOS files to construct eq. lat. interpolators for
    :type geos_nv_files: list(str)

    :param geos_dates: the datetimes of the GEOS files given as the first argument. Must be in the same order, i.e.
     ``geos_dates[i]`` must be the datetime of ``geos_nv_files[i]``.
    :type geos_dates: list(datetime-like)

    :param muted: set to ``True`` to disable some logging to console.
    :type muted: bool

    :return: a dictionary of equivalent latitude interpolators. THe keys will be the dates of the GEOS files, there will
     be one interpolator per GEOS file.
    :rtype: dict
    """
    func_dict = dict()
    start = time.time()
    for idx, (geos_file, date) in enumerate(zip(geos_nv_files, geos_dates)):
        with netCDF4.Dataset(geos_file, 'r') as dataset:
            logger.info('Calculating equivalent latitudes for {}/{} GEOS files'.format(idx+1, len(geos_nv_files)))
            lat = dataset['lat'][:]
            lat[np.abs(lat) < 0.001] = 0.0
            lon = dataset['lon'][:]
            pres = mod_utils.convert_geos_eta_coord(dataset['DELP'][0])
            EPV = dataset['EPV'][0] * 1e6
            PT = mod_utils.calculate_potential_temperature(pres, dataset['T'][0])

            # Get the area of each grid cell
            lat_res = float(dataset.LatitudeResolution)
            lon_res = float(dataset.LongitudeResolution)
            area = mod_utils.calculate_area(lat, lon, lat_res, lon_res, muted=muted)

        # The native 72-level geos files are ordered space-to-surface. The equivalent latitude calculation *may* be okay
        # with that, but I felt it was safer to just go ahead and flip them.
        func_dict[date] = mod_utils.calculate_eq_lat(np.flip(EPV, axis=0), np.flip(PT, axis=0), area)
    print("It took {:.1f} minutes to generate equivalent latitude functions for {} GEOS files".format((time.time()-start)/60.0,len(geos_nv_files)))

    return func_dict


def add_equivalent_latitude_to_native_geos_file(geos_nv_file, muted=False):
    """
    Add an 'eqlat' variable to a native GEOS FP(-IT) file with the equivalent latitudes.

    :param geos_nv_file: full path to a native GEOS files to which an 'eqlat' variable will be added
    :type geos_nv_file: str

    :param muted: set to ``True`` to disable some logging to console.
    :type muted: bool
    """
    with netCDF4.Dataset(geos_nv_file, 'r+') as dataset:
        logger.info(f'Calculating equivalent latitudes for {geos_nv_file}')
        lat = dataset['lat'][:]
        lat[np.abs(lat) < 0.001] = 0.0
        lon = dataset['lon'][:]
        pres = mod_utils.convert_geos_eta_coord(dataset['DELP'][0])
        EPV = dataset['EPV'][0] * 1e6
        PT = mod_utils.calculate_potential_temperature(pres, dataset['T'][0])

        if 'eqlat' not in dataset.variables:
            dataset.createVariable('eqlat',np.float32,('time','lev','lat','lon'))
            att_dict = {
                'units':'degrees_north',
                'long_name':'equivalent latitude',
                'standard_name':'equivalent_latitude',
                'scale_factor':1.0,
                'add_offset':0.0,
            }
            dataset['eqlat'].setncatts(att_dict)

        # Get the area of each grid cell
        lat_res = float(dataset.LatitudeResolution)
        lon_res = float(dataset.LongitudeResolution)
        area = mod_utils.calculate_area(lat, lon, lat_res, lon_res, muted=muted)

        eqlat = mod_utils.calculate_eq_lat_field(np.flip(EPV, axis=0), np.flip(PT, axis=0), area)

        dataset['eqlat'][0] = np.flip(eqlat, axis=0)
            

def lat_lon_interp(data_old,lat_old,lon_old,lat_new,lon_new,IDs_list):
    """
    Use RectSphereBivariateSpline to interpolate in a latitude-longitude grid (rectangle over of sphere)

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.html

    lat_old: array of latitude within [-90,90] degrees
    lon_old: array of longitudes within [-180,+180[ degrees (+180 excluded)

    lat_new: array of latitudest o interpolate to [-90,90]
    lon_new: array of longitudes to interpolate to [-180,180[
    """

    # make copies of input arrays to not modify them in place
    data_old = data_old.copy()
    lat_old = lat_old.copy()
    lon_old = lon_old.copy()
    lat_new = lat_new.copy()
    lon_new = lon_new.copy()

    """	
    # Using interpolation in rectangles on a sphere
    lat_old = deg2rad(lat_old+90)[1:-1]
    lon_old = deg2rad(lon_old)

    data_old = data_old[1:-1,:]

    func = RectSphereBivariateSpline(lat_old,lon_old,data_old)

    for i,elem in enumerate(lon_new):
        if elem<0:
            lon_new[i] = elem + 360

    data_new = func.ev(deg2rad(lat_new+90),deg2rad(lon_new))
    """

    data_new = []
    count = 0
    for IDs in IDs_list:
        lat1,lat2,lon1,lon2 = IDs

        lat = np.array([lat_old[lat1],lat_old[lat2]])
        lon = np.array([lon_old[lon1],lon_old[lon2]])

        data = np.array([[data_old[lat1,lon1],data_old[lat1,lon2]],[data_old[lat2,lon1],data_old[lat2,lon2]]])

        data = ma.masked_where(np.isnan(data),data)

        func = interp2d(lon,lat,data)

        data_new.append(func(lon_new[count],lat_new[count]))

        count+=1

    return data_new

def show_interp(data,x,y,interp_data,ilev,pres):

    max = data[ilev].max()
    min = data[ilev].min()

    pl.imshow(data[ilev],extent=(-180,180,90,-90),vmin=min,vmax=max)
    try:
        pl.scatter(x,y,c=np.diag(interp_data[ilev]),vmax=max,vmin=min,edgecolor='black')
    except:
        pl.scatter(x,y,c=interp_data[ilev],vmax=max,vmin=min,edgecolor='black')
    pl.gca().invert_yaxis()
    pl.xlabel('Longitude')
    pl.ylabel('Latitude')
    pl.title('Level {}: {} hPa'.format(ilev+1,pres[ilev]))
    pl.colorbar()
    pl.show()


def load_chem_variables(geos_file, geos_vars, target_site_dicts, pres_levels=None,
                        muted=False, extrapolate='no'):
    if not mod_utils.is_geos_on_native_grid(geos_file):
        raise NotImplementedError('GEOS chemistry file ({}) does not appear to be on the native eta grid. This case '
                                  'has not been implemented.')
    # Load the chemical variables
    with netCDF4.Dataset(geos_file, 'r') as dataset:
        box_lat_half_width = 0.5 * float(dataset.LatitudeResolution)
        box_lon_half_width = 0.5 * float(dataset.LongitudeResolution)

        lat = dataset['lat'][:]
        lon = dataset['lon'][:]

        geos_data = dict()
        for var in geos_vars:
            geos_data[var] = dataset[var][0]
            if geos_data[var].shape[0] == 72:
                # The vertical dimension should be first if present. Flip native variables
                # to be surface-to-space.
                geos_data[var] = np.flipud(geos_data[var])

        geos_pres = mod_utils.convert_geos_eta_coord(dataset['DELP'][0].filled(np.nan))
        geos_data['pres'] = np.flipud(geos_pres)

    # Handle the lat/lon interpolation
    if pres_levels is None:
        nlevels = geos_data['pres'].shape[0]
    elif isinstance(pres_levels, dict):
        nlevels = set(d['prof']['lev'].size for d in pres_levels.values())
        if len(nlevels) != 1:
            raise NotImplementedError('Differently sized profile data')
        nlevels = nlevels.pop()
    else:
        nlevels = np.size(pres_levels)
    nsites = len(target_site_dicts)
    site_data = {v: np.full([nlevels, nsites], np.nan) for v in geos_vars}

    for site, subdict in target_site_dicts.items():
        slat = subdict['lat']
        slon = subdict['lon_180']
        target_site_dicts[site]['IDs'] = querry_indices([lat, lon], site_lat=slat, site_lon_180=slon,
                                                        box_lat_half_width=box_lat_half_width,
                                                        box_lon_half_width=box_lon_half_width)

    interp_geos_data = interp_geos_data_to_sites(geos_data, lat, lon, target_site_dicts, muted=muted)

    # Interpolate to the standard pressure levels. Do this in log-log space since pressure and concentration typically
    # vary exponentially with altitude. If no pressure levels given, then assume we are working with the native files
    # for met as well and can just leave CO on the standard levels.
    #
    # Note that this is only really tested with a single site in the target_site_dicts, since this is normally run
    # with the iteration over sites done by `driver`.
    if pres_levels is not None:
        target_site_ids = list(target_site_dicts)
        for i in range(nsites):
            if isinstance(pres_levels, dict):
                # Means we passed in the INTERP_DATA dictionary from the main function, so we need to extract the
                # pressure levels for this site.
                std_pres_log = np.log(pres_levels[target_site_ids[i]]['prof']['lev'])
            else:
                std_pres_log = np.log(pres_levels)
                
            pres_log = np.flipud(np.log(interp_geos_data['pres'][:, i]))
            for var in geos_vars:
                var_log = np.flipud(np.log(interp_geos_data[var][:, i]))
                # Other parts of modmaker use scipy's interp1d without issue; but I'm more comfortable with the
                # straightforward linear interpolation np.interp does. Sometime scipy's interpolators behave strangely.
                if extrapolate == 'no':
                    left, right = np.nan, np.nan
                elif extrapolate == 'const':
                    left, right = var_log[0], var_log[-1]
                else:
                    NotImplementedError(f'extrapolate = {extrapolate}')
                var_log = np.interp(std_pres_log, pres_log, var_log, left=left, right=right)
                site_data[var][:, i] = np.exp(var_log)
    else:
        for var in geos_vars:
            site_data[var][:] = interp_geos_data[var]

    site_data = combine_profile_surface_data(site_data, dict(), target_site_dicts)

    return site_data


def interp_geos_data_to_sites(DATA, lat, lon, site_dict, varlist=None, muted=False):
    """
    Interpolate GEOS data to the lat/lon of the sites where .mod files are needed.

    :param DATA: dictionary of GEOS arrays arranged levels-by-lat-by-lon. The keys must be the GEOS variable names.
    :type DATA: dict

    :param lat: the latitude vector for the GEOS arrays.
    :type lat: :class:`numpy.ndarray`

    :param lon: the longitude vector for the GEOS arrays.
    :type lon: :class:`numpy.ndarray`

    :param site_dict: the dictionary organized by site. Keys will be the site abbreviation, values must themselves
     be dictionaries which contain the key "IDs" which are the grid cell indices returned by :func:`querry_indices`.
    :type site_dict: dict

    :param varlist: the list of variables from the GEOS data that should be interpolated to the site lat/lons.
    :type varlist: list(str)

    :param muted: set to ``True`` to silence progress messages
    :type muted: bool

    :return: a dictionary of GEOS variables as masked arrays, interpolated to the site lat/lons. The arrays will be
     nlevels-by-nsites.
    :rtype: dict
    """
    if varlist is None:
        varlist = list(DATA.keys())

    ids_list = [site_dict[site]['IDs'] for site in site_dict]
    new_lats = np.array([site_dict[site]['lat'] for site in site_dict])
    new_lons = np.array([site_dict[site]['lon_180'] for site in site_dict])

    nsite = len(site_dict)
    default_geos_var = list(DATA.keys())[0]
    nlev = DATA[default_geos_var].shape[0]

    if not muted:
        print('\t-Interpolate to (lat,lon) of sites ...')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interp_data = dict()
        for var in varlist:
            if not muted:
                sys.stdout.write('\r\t\tNow doing : {:<10s}'.format(var))
                sys.stdout.flush()

            if DATA[var].ndim == 2:
                interp_data[var] = lat_lon_interp(DATA[var], lat, lon, new_lats, new_lons, ids_list)
            else:
                interp_data[var] = np.zeros([nlev, nsite])
                for ilev, level_data in enumerate(DATA[var]):
                    # JLL 2023-008-29: for whatever reason, numpy 1.24.4 doesn't like assigning a list of arrays
                    # to an array element, but it is okay converting that list to an equivalent 2D array and assigning that.
                    interp_data[var][ilev] = np.asarray(lat_lon_interp(level_data, lat, lon, new_lats, new_lons, ids_list))

    # setup masks
    for var in varlist:
        for i in range(nsite):
            interp_data[var] = ma.masked_where(np.isnan(interp_data[var]), interp_data[var])

    return interp_data


def combine_profile_surface_data(interp_profile_data, interp_surf_data, local_site_dict):
    """
    Merge profile and surface data into a single dictionary organized by site.

    :param interp_profile_data: a dictionary of profile data, where each key is a variable name and the arrays are
     nlevels-by-nsites.
    :type interp_profile_data: dict

    :param interp_surf_data: a dictionary of surface data, where each key is a variable name and the arrays are
     nlevels-long vectors.
    :type interp_surf_data: dict

    :param local_site_dict: a dictionary with the abbreviations for the sites in the first two dictionaries as keys.
    :type local_site_dict: dict

    :return: a dictionary organized by site, then 'prof'/'surf', then variable.
    :rtype: dict
    """
    # restructure the data from two dictionaries organized with variables as keys and the variables as arrays including
    # all sites to one dictionary organized by site, 'prof'/'surf', and variable.
    temp_data = dict()
    for i, site in enumerate(local_site_dict.keys()):
        temp_data[site] = {}
        temp_data[site]['prof'] = {}
        for var in interp_profile_data:
            if var == 'PHIS':
                continue
            else:
                temp_data[site]['prof'][var] = interp_profile_data[var][:, i]

        temp_data[site]['surf'] = {}
        for var in interp_surf_data:
            if var == 'H':
                temp_data[site]['surf'][var] = mod_utils.geopotential_height_to_altitude(
                    interp_surf_data[var][i][0], local_site_dict[site]['lat'], local_site_dict[site]['alt'] / 1000.0
                )
            else:
                temp_data[site]['surf'][var] = interp_surf_data[var][i]
    return temp_data


def extrapolate_to_surface(var_to_interp, INTERP_DATA, SLANT_DATA=None):
    """
    Extend GEOS variables to fill out all pressure levels

    :param var_to_interp: a dictionary specifying which variables in INTERP_DATA need to be extended. The keys must be
     the profile variables to extend and the values the surface variables to use to extrapolate those variables. If
     there is no surface variable, then use ``None`` as the value. This will cause the extrapolation to just use the
     bottom value of the profile for all levels below it.
    :type var_to_interp: dict

    :param INTERP_DATA: a dictionary of vertical profile and surface data. Expected to be a dict of dict of dicts; the
     top level keys are the site abbreviations, the second level being 'prof' or 'surf' for profile and surface
     variables, respectively, and the third level being the variable names.
    :type INTERP_DATA: dict

    :param SLANT_DATA: a dictionary of slant profiles and surface data. Must have same format as ``INTERP_DATA``.
    :type SLANT_DATA: dict

    :return: None. Modified INTERP_DATA and SLANT_DATA in-place.
    """
    def get_and_check_mask(data_dict):
        chk_var = list(var_to_interp.keys())[0]
        chk_mask = data_dict[chk_var].mask.copy()  # make a copy to avoid changing this mask as we fill values
        for v in var_to_interp:
            # using array_equiv here should directly handle cases where the check mask is a scalar `False`
            # but the variable's mask is a vector of all `False` values, since the former can be broadcast
            # to be the same as the latter.
            if not np.array_equiv(chk_mask, data_dict[v].mask):
                raise ValueError('All variables to interpolate in input data must have the same mask')
        return chk_mask

    for site in INTERP_DATA.keys():
        var_with_surf_data = {prof: surf for prof, surf in var_to_interp.items() if surf is not None}
        var_without_surf_data = {prof: surf for prof, surf in var_to_interp.items() if surf is None}

        if SLANT_DATA is not None:
            all_data = [SLANT_DATA[site], INTERP_DATA[site]['prof']]
        else:
            all_data = [INTERP_DATA[site]['prof']]

        for elem in all_data:
            # Assume that the height variable
            H_mask = get_and_check_mask(elem)

            patch = {'surf': {}, 'first': {}}
            if np.any(elem['lev'].mask):
                raise NotImplementedError('The "lev" variable has masked elements, this is not supported')
            # This function expects 1D pressure levels. Since we're interpolating pressure like other 3D variables now
            # to support native GEOS files, pressure levels come in as nlev-by-1 arrays that need squeezed down or we
            # get an error at the `elem[var][H_mask] = f(missing_p)` line.
            level_pres = elem['lev'].data.squeeze()
            missing_p = level_pres[H_mask]  # pressure levels with masked data
            if len(missing_p) != 0:
                surf_p = INTERP_DATA[site]['surf']['PS']  # surface pressure

                first_p = level_pres[~H_mask][0]  # first valid pressure level
                first_ID = np.where(level_pres == first_p)[0]

                for prof_var, surf_var in var_with_surf_data.items():
                    patch['surf'][prof_var] = INTERP_DATA[site]['surf'][surf_var]
                    patch['first'][prof_var] = elem[prof_var][first_ID].data[0]

                # interpolate/extrapolate using the first valid level and surface data
                if surf_p > first_p:  # interpolate between the surface pressure and first valid pressure level
                    patch_p = [surf_p, first_p]
                    for var in var_with_surf_data:
                        patch_var = [patch['surf'][var], patch['first'][var]]
                        f = interp1d(patch_p, patch_var, fill_value='extrapolate')
                        elem[var][H_mask] = f(missing_p)
                else:  # use the surface value and the first level above it to extrapolate down to 1000 hPa
                    valid_p = level_pres[~H_mask]
                    first_p = valid_p[valid_p < surf_p][0]
                    patch_p = np.array([surf_p, first_p])  # surface pressure and first level above it
                    for var in var_with_surf_data:
                        valid_var = elem[var][~H_mask]
                        patch_var = np.array([patch['surf'][var], valid_var[valid_p < surf_p][0]])
                        f = interp1d(patch_p, patch_var, fill_value='extrapolate')
                        elem[var][:np.where(level_pres == first_p)[0][0]] = f(level_pres[:np.where(level_pres == first_p)[0][0]])

                # for variables with no surface data, just use the value of the first valid level above the surface
                for var in var_without_surf_data:
                    elem[var][:np.where(level_pres==first_p)[0][0]] = elem[var][np.where(level_pres==first_p)][0]


def mod_maker_new(start_date=None, end_date=None, func_dict=None, GEOS_path=None, chem_path=None, locations=site_dict,
                  slant=False, muted=False, lat=None, lon=None, alt=None, site_abbrv=None, save_path=None, product='fpit',
                  keep_latlon_prec=False, save_in_utc=True, native_files=False, chem_variables=tuple(), flat_outdir=False, **kwargs):
    """
    This code only works with GEOS-5 FP-IT data.
    It generates MOD files for all sites between start_date and end_date on GEOS-5 times (every 3 hours)

    Inputs:
        - start_date: datetime object for first date, YYYYMMDD_HH, _HH is optional and defaults to _00
        - end_date:  datetime object for last date, YYYYMMDD_HH, _HH is optional and defaults to _00
        - func_dict: output of equivalent_latitude_functions
        - GEOS_path: full path to the directory containing all the GEOS5-FP-IT files, the directory must contain a 'Np' folder with profile data, and a 'Nx' folder with surface data
        - locations: dictionary of sites, defaults to the one in tccon_sites.py
        - slant: if True both slant and vertical .mod files will be generated
        - muted: if True there will be no print statements except for warnings and errors
        - (optional) lat: latitude in [-90,90] range
        - (optional) lon: longitude in [0,360] range
        - (optional) alt: altitude (meters)
        - (optional) site_abbrv: two letter site abbreviation
    Outputs:
        - .mod files at every GEOS5 time within the given date range

    If any of alt/lat/lon is given, the other two must be given too as well as site_abbrv

    When giving dates with _HHMM, dates must correspond exactly to GEOS5 times, so 3 hourly UTC times starting at HHMM=0000
    """
    
    if lat is not None: # a custom location was given
        site_abbrv = 'xx' if site_abbrv is None else site_abbrv
        locations = {site_abbrv:{'name':'custom site','loc':'custom loc','lat':lat,'lon':lon,'alt':alt}}
    elif site_abbrv and site_abbrv != 'all': # if not custom location is given, but a site abbreviation is given, just do that one site
        locations = {site_abbrv:locations[site_abbrv]}

    if chem_path is None:
        # Assume that the chemistry files are in the same folder as the met files
        chem_path = GEOS_path

    if save_path is None:
        GGGPATH = os.environ['GGGPATH']
        if GGGPATH is None:
            raise RuntimeError('No custom save_path provided, and the GGGPATH environmental variable is not set. '
                               'One of these must be provided.')
        save_path = os.path.join(GGGPATH,'models','gnd')
        
    if flat_outdir and slant:
        # Currently the vertical and slant files get the same names, so if we're not allowed to save into separate
        # subdirectories, they will overwrite each other
        raise NotImplementedError('Cannot request slant path files when .mod files be saved directly to `save_dir` (`flat_outdir=True`)')
    elif flat_outdir:
        mod_path = save_path
    else:
        mod_path = os.path.join(save_path,product)
    
    if not os.path.exists(mod_path):
        if not muted:
            print('Creating',mod_path)
        os.makedirs(mod_path)

    do_load_chem = len(chem_variables) > 0
    if slant and do_load_chem:
        raise NotImplementedError('Slant path chemistry variables have not yet been implemented')

    varlist = ['T','QV','RH','H','EPV','O3','PHIS', 'lev']
    surf_varlist = ['T2M','QV2M','PS','SLP','TROPPB','TROPPV','TROPPT','TROPT']

    profile_subdir = 'Nv' if native_files else 'Np'
    select_files, select_dates = GEOS_files(os.path.join(GEOS_path, profile_subdir),start_date,end_date)
    select_surf_files, select_surf_dates = GEOS_files(os.path.join(GEOS_path,'Nx'),start_date,end_date)
    if do_load_chem:
        # Assumes that chemistry files are the only ones in the Nv directory
        select_chem_files, select_chem_dates = GEOS_files(os.path.join(chem_path, 'Nv'), start_date, end_date, chm=True)
        if len(select_chem_dates) != len(select_dates) or any(d1 != d2 for d1, d2 in zip(select_chem_dates, select_dates)):
            raise RuntimeError('Dates for the chemistry files do not match the dates for the met file. Something '
                               'went wrong when looking for these files.')

    nsite = len(locations)

    start = time.time()
    mod_dicts = dict()

    for date_ID, UTC_date in enumerate(select_dates):
        site_dict = tccon_site_info_for_date(UTC_date, site_dict_in=locations)
        mod_dicts[UTC_date] = dict()
        start_it = time.time()

        geos_versions = {
            'Met3d': GeosVersion.from_nc_file(select_files[date_ID]),
            'Met2d': GeosVersion.from_nc_file(select_surf_files[date_ID])
        }
        if do_load_chem:
            geos_versions['Chm3d'] = GeosVersion.from_nc_file(select_chem_files[date_ID])
            
        DATA = {}
        if not muted:
            print('\nNOW DOING date {:4d} / {} :'.format(date_ID+1,len(select_dates)),UTC_date.strftime("%Y-%m-%d %H:%M"),' UTC')
            print('\t-Read global data ...')

        file_is_native = mod_utils.is_geos_on_native_grid(select_files[date_ID])
        if file_is_native != native_files:
            raise RuntimeError('Loaded a native level GEOS file but expected a fixed pressure file, or vice versa')

        with netCDF4.Dataset(select_files[date_ID],'r') as dataset:
            for var in varlist:
                if var == 'lev':
                    # 'lev' needs handle specially because we want it to always be pressure, but in the native files
                    # it is eta.
                    continue

                # Taking dataset[var][0] is equivalent to dataset[var][0,:,:,:], which since there's only one time per
                # file just cuts the data from 4D to 3D
                DATA[var] = dataset[var][0]
                if file_is_native and DATA[var].shape[0] == 72:
                    # The native 72 eta level files are organized space-to-surface vertically; the 42 fixed pressure
                    # level files are surface-to-space. We want the latter so we need to flip the vertical dimension
                    # if it is a native file. The vertical dimension, if present, should be first and have 72 levels.
                    DATA[var] = np.flipud(DATA[var])

            if file_is_native:
                pres_levels = mod_utils.convert_geos_eta_coord(dataset['DELP'][0])
                pres_levels = np.flipud(pres_levels)
            else:
                pres_levels = dataset['lev'][:]
                pres_levels = np.broadcast_to(pres_levels.reshape(-1, 1, 1), DATA[varlist[0]].shape)
            DATA['lev'] = pres_levels

            lat = dataset['lat'][:]
            lon = dataset['lon'][:]
            nlev = dataset.dimensions['lev'].size
            if date_ID == 0:
                box_lat_half_width = 0.5*float(dataset.LatitudeResolution)
                box_lon_half_width = 0.5*float(dataset.LongitudeResolution)

        for site in site_dict:
            if 'time_spans' in site_dict[site].keys(): # instruments with different locations for different time periods
                for time_span in site_dict[site]['time_spans']:
                    if time_span[0]<=UTC_date<time_span[1]:
                        site_dict[site]['IDs'] = querry_indices([lat,lon],site_dict[site]['time_spans'][time_span]['lat'],site_dict[site]['time_spans'][time_span]['lon_180'],box_lat_half_width,box_lon_half_width)
                        site_dict[site]['lat'] = site_dict[site]['time_spans'][time_span]['lat']
                        site_dict[site]['lon'] = site_dict[site]['time_spans'][time_span]['lon']
                        site_dict[site]['lon_180'] = site_dict[site]['time_spans'][time_span]['lon_180']
                        site_dict[site]['alt'] = site_dict[site]['time_spans'][time_span]['alt']
                        break
            else:
                site_dict[site]['IDs'] = querry_indices([lat,lon],site_dict[site]['lat'],site_dict[site]['lon_180'],box_lat_half_width,box_lon_half_width)

        SURF_DATA = {}
        with netCDF4.Dataset(select_surf_files[date_ID],'r') as dataset:
            for var in surf_varlist:
                SURF_DATA[var] = dataset[var][0]

        if not muted:
            print('\t-Interpolate to (lat,lon) of sites ...')

        # interpolate pressure levels along with the rest of the 3D variables. If we're using a native file, we're not
        # on fixed pressure levels, and need to interpolate anyway. If using a fixed pressure level file, we've
        # broadcast the pressure levels to be the same size as the rest of the 3D variables.
        INTERP_DATA = interp_geos_data_to_sites(DATA, lat=lat, lon=lon, site_dict=site_dict, varlist=varlist,
                                                muted=muted)

        INTERP_SURF_DATA = interp_geos_data_to_sites(SURF_DATA, lat=lat, lon=lon, site_dict=site_dict,
                                                     varlist=surf_varlist, muted=muted)

        ##############################################################################
        # Handle some variable conversions/custom calculations for the met variables #
        ##############################################################################

        # Ensure that the surface variables are 1D
        for var in surf_varlist:
            INTERP_SURF_DATA[var] = INTERP_SURF_DATA[var].reshape(nsite)

        # Convert pressure fields to hPa. 'lev' is already in hPa.
        for varname in ['PS','SLP','TROPPB','TROPPV','TROPPT']:
            INTERP_SURF_DATA[varname] = INTERP_SURF_DATA[varname] / 100.0  # convert Pa to hPa

        # Convert specific humidity, a wet mass mixing ratio, to dry mole fraction
        INTERP_DATA['H2O_DMF'] = rmm*INTERP_DATA['QV']/(1-INTERP_DATA['QV'])
        INTERP_DATA['H'] = INTERP_DATA['H']/1000.0  # Convert m to km
        INTERP_DATA['PHIS'] = INTERP_DATA['PHIS'] / 1000.0

        INTERP_SURF_DATA['H2O_DMF'] = compute_h2o_dmf(INTERP_SURF_DATA['QV2M'],rmm)

        # compute surface relative humidity
        svp = svp_wv_over_ice(INTERP_SURF_DATA['T2M'])
        INTERP_SURF_DATA['H2O_WMF'] = compute_h2o_wmf(INTERP_SURF_DATA['H2O_DMF'])  # wet mole fraction of h2o
        INTERP_SURF_DATA['RH'] = compute_rh(INTERP_SURF_DATA['T2M'],INTERP_SURF_DATA['H2O_WMF'],INTERP_SURF_DATA['PS'])/100 # Fractional relative humidity
        INTERP_SURF_DATA['MMW'] = compute_mmw(INTERP_SURF_DATA['H2O_WMF'])
        INTERP_SURF_DATA['H'] = INTERP_DATA['PHIS']

        # Combine the profile and surface data into a new dictionary organized by site/levels/variable.
        INTERP_DATA = combine_profile_surface_data(INTERP_DATA, INTERP_SURF_DATA, site_dict)

        # If requested, load the chemistry data and incorporate it into the existing dictionaries.
        if do_load_chem:
            if native_files and geos_versions['Met3d'] == geos_versions['Chm3d']:
                # Using the eta levels and both 3D files are from the same version of GEOS, so we 
                # can assume that the lat/lon interpolation will result in chem data on the same levels.
                chem_plevs = None
                chem_extrap = 'no'
            elif native_files:
                # Otherwise, with different versions of GEOS, the level pressures might change slightly,
                # so we do need to interpolate. As elsewhere, use constant value extrapolation to ensure
                # that all levels receive a value, but we won't get runaway interpolation.
                chem_plevs = INTERP_DATA
                chem_extrap = 'const'
            else:
                # This is a carry-over from earlier versions of mod maker that used fixed pressure levels.
                # I think we did not want to extrapolate b/c levels outside the native pressure levels should
                # get fill values (i.e. they are below the terrain, usually).
                chem_plevs = mod_utils._std_model_pres_levels
                chem_extrap = 'no'
            CHEM_DATA = load_chem_variables(select_chem_files[date_ID], chem_variables, site_dict, pres_levels=chem_plevs, extrapolate=chem_extrap)
            for site in INTERP_DATA.keys():
                INTERP_DATA[site]['prof'].update(CHEM_DATA[site]['prof'])

        # add a mask for temperature = 0 K
        for site in site_dict:
            for var in INTERP_DATA[site]['prof']:
                if var not in ['T','lev']:
                    INTERP_DATA[site]['prof'][var] = ma.masked_where(INTERP_DATA[site]['prof']['T']==0,INTERP_DATA[site]['prof'][var])
            INTERP_DATA[site]['prof']['T'] = ma.masked_where(INTERP_DATA[site]['prof']['T']==0,INTERP_DATA[site]['prof']['T'])

            INTERP_DATA[site]['surf']['SZA'] = rad2deg(sun_angles(UTC_date,deg2rad(site_dict[site]['lat']),deg2rad(site_dict[site]['lon_180']),site_dict[site]['alt'],INTERP_DATA[site]['surf']['PS'],INTERP_DATA[site]['surf']['T2M'])[0])

        if not slant:
            SLANT_DATA = None
        else:
            # get slant path coordinates corresponding to the altitude levels above each site
            if not muted:
                print('\t-Slantify:')
            for i,site in enumerate(site_dict.keys()): # loops over sites
                if not muted:
                    sys.stdout.write('\r\t\t site {:3d} / {}  {:>20}'.format(i+1,nsite,site_dict[site]['name']))
                    sys.stdout.flush()

                site_alt = site_dict[site]['alt']
                site_lat = site_dict[site]['lat']
                site_lon = site_dict[site]['lon_180']

                # vertical grid above site
                H = INTERP_DATA[site]['prof']['H']*1000.0
                pres = INTERP_DATA[site]['surf']['PS'] # surface pressure (hPa)
                temp = INTERP_DATA[site]['surf']['T2M']-273.15 # surface temperature (celsius)


                # get the (lat,lon,alt) of points on sunray correspondings to the vertical altitudes
                site_dict[site]['slant_coords'] = slantify(UTC_date,site_lat,site_lon,site_alt,H,pres=pres,temp=temp)
                for var in ['lat','lon','alt','vertical','slant']:
                    site_dict[site]['slant_coords'][var] = ma.masked_where(H.mask,site_dict[site]['slant_coords'][var])
            if not muted:
                print('\r\t\t{:<40s}'.format('DONE'))

            # Set two lists with all the latitudes and longitudes of all sites at all slant levels
            slant_lat = []
            slant_lon = []
            slat_slon = []
            for site in site_dict:
                if site_dict[site]['slant_coords']['sza']<90: # only make profiles where sun is above the horizon
                    slat = site_dict[site]['slant_coords']['lat']
                    slon = site_dict[site]['slant_coords']['lon']
                    slant_lat.extend(slat)
                    slant_lon.extend(slon)
                    for i in range(len(slat)):
                        if slat[i] is ma.masked:
                            continue
                        if (slat[i],slon[i]) not in slat_slon:
                            slat_slon.append((slat[i],slon[i]))

            IDs_list = np.array([querry_indices([lat,lon],slat,slon,box_lat_half_width,box_lon_half_width) for slat,slon in slat_slon])

            slant_lat = np.array([slat for slat,slon in slat_slon])
            slant_lon = np.array([slon for slat,slon in slat_slon])

            # Interpolate to each slant level (lat,lon)
            # This will give a vertical profile at every (lat,lon) of all the slant levels
            if not muted:
                print('\t-Interpolate to each slant level (lat,lon) ...')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                NEW_INTERP_DATA = {}
                for var in varlist:
                    if not muted:
                        sys.stdout.write('\r\t\tNow doing : {:<10s}'.format(var))
                        sys.stdout.flush()
                    if DATA[var].ndim==2:
                        NEW_INTERP_DATA[var] = lat_lon_interp(DATA[var],lat,lon,slant_lat,slant_lon,IDs_list)
                        continue

                    NEW_INTERP_DATA[var] = np.zeros([nlev,len(IDs_list)])
                    for ilev,level_data in enumerate(DATA[var]):
                        NEW_INTERP_DATA[var][ilev] = lat_lon_interp(level_data,lat,lon,slant_lat,slant_lon,IDs_list)
                if not muted:
                    print('\r\t\t{:<40s}'.format('DONE'))
            # setup masks
            for var in set(varlist)-set(['PHIS']):
                NEW_INTERP_DATA[var] = ma.masked_where(np.isnan(NEW_INTERP_DATA[var]),NEW_INTERP_DATA[var])

            # Now just get the data along the slant paths
            if not muted:
                print('\t-Get data along slant paths ...')
            SLANT_DATA = {}
            for site in site_dict: # for each site
                if site_dict[site]['slant_coords']['sza']<90:
                    SLANT_DATA[site] = site_dict[site]['slant_coords']
                    SLANT_DATA[site]['H'] = SLANT_DATA[site]['alt']
                    SLANT_DATA[site]['lev'] = INTERP_DATA[site]['prof']['lev']
                    for var in set(varlist)-set(['H','PHIS']): # for each variable
                        SLANT_DATA[site][var] = np.array([])
                        for i in range(len(SLANT_DATA[site]['H'])): # for each slant point
                            slat,slon = SLANT_DATA[site]['lat'][i] , SLANT_DATA[site]['lon'][i]
                            try:
                                ID = slat_slon.index((slat,slon))
                            except ValueError:
                                SLANT_DATA[site][var] = np.append(SLANT_DATA[site][var],np.nan)
                            else:
                                SLANT_DATA[site][var] = np.append(SLANT_DATA[site][var],NEW_INTERP_DATA[var][i,ID])

                        SLANT_DATA[site][var] = ma.masked_where(np.isnan(SLANT_DATA[site][var]),SLANT_DATA[site][var])
        # end of 'if slant'

        if not muted:
            print('\t-Patch fill values ...')

        if not file_is_native:
            # Fixed pressure level files will result in NaNs for pressure levels below the surface. We want to fill
            # those in. Native files are terrain following so this should not be an issue.
            extrap_vars = {'RH': 'RH', 'QV': 'QV2M', 'T': 'T2M', 'H': 'H', 'EPV': None, 'O3': None}
            extrap_vars.update({k: None for k in chem_variables})
            extrapolate_to_surface(extrap_vars, INTERP_DATA, SLANT_DATA=SLANT_DATA)

        for site in site_dict:
            if slant:
                all_data = [SLANT_DATA[site],INTERP_DATA[site]['prof']]
            else:
                all_data = [INTERP_DATA[site]['prof']]

            # Needed this to be a fraction for the extrapolation so that it is in the same units as the profile. Now
            # convert back to percent. Can't convert the profile RH here, it's used for certain calculations throughout
            # that assume it is a fraction. It will get scaled in the write_mod function.
            INTERP_DATA[site]['surf']['RH'] *= 100

            for elem in all_data:
                # Convert specific humidity, a wet mass mixing ratio, to dry mole fraction
                elem['H2O_DMF'] = compute_h2o_dmf(elem['QV'], rmm)

        if not muted:
            if slant:
                print('\t\t {:3d} / {} sites with SZA<90'.format(len(SLANT_DATA.keys()),nsite))
            print('\t-Write mod files ...')

        # write the .mod files
        version = 'mod_maker.py   2019-06-20   SR/JL'

        for site in INTERP_DATA:
            mod_dicts[UTC_date][site] = dict()
            site_lat = site_dict[site]['lat']
            site_lon_180 = site_dict[site]['lon_180']

            utc_offset = timedelta(hours=site_dict[site]['lon_180']/15.0) if not save_in_utc else timedelta(hours=0)
            local_date = UTC_date + utc_offset

            vertical_mod_path = mod_path if flat_outdir else os.path.join(mod_path,site,'vertical')
            if not os.path.exists(vertical_mod_path):
                os.makedirs(vertical_mod_path)

            if slant:
                # We already check at the beginning of this function that flat_outdir = False if slant = True
                # so we don't need to handle the flat_outdir = True case here.
                slant_mod_path =  os.path.join(mod_path,site,'slant')
                if not os.path.exists(slant_mod_path):
                    os.makedirs(slant_mod_path)

            # directions for .mod file name
            if site_lat >= 0:
                ns = 'N'
            else:
                ns = 'S'

            if site_lon_180 >= 0:
                ew = 'E'
            else:
                ew = 'W'

            mod_name = mod_utils.mod_file_name(product.upper(), local_date, timedelta(hours=3), site_lat, site_lon_180, ew, ns, mod_path, round_latlon=not keep_latlon_prec, in_utc=save_in_utc)
            if not muted:
                print('\t\t\t{:<20s} : {}'.format(site_dict[site]['name'], mod_name))

            # write vertical mod file
            mod_file_path = os.path.join(vertical_mod_path,mod_name)
            vertical_mod_dict = write_mod(mod_file_path,version,site_lat,data=INTERP_DATA[site]['prof']
                                          ,surf_data=INTERP_DATA[site]['surf'],func=func_dict[UTC_date],
                                          muted=muted,slant=slant,chem_vars=do_load_chem,geos_versions=geos_versions)

            if slant:
                # write slant mod_file
                if site in SLANT_DATA.keys():
                    if not muted:
                        print('\t\t\t{:>20s} + slant'.format(''))
                    mod_file_path = os.path.join(slant_mod_path,mod_name)
                    slant_mod_dict = write_mod(mod_file_path,version,site_lat,data=SLANT_DATA[site],surf_data=INTERP_DATA[site]['surf'],func=func_dict[UTC_date],muted=muted,slant=slant)
            else:
                slant_mod_dict = dict()

            mod_dicts[UTC_date][site]['vertical'] = vertical_mod_dict
            mod_dicts[UTC_date][site]['slant'] = slant_mod_dict
        if not muted:
            print('\ndate {:4d} / {} DONE in {:.0f} seconds'.format(date_ID+1,len(select_dates),time.time()-start_it))
    if not muted:
        print('It took {:.1f} minutes to generate .mod files for {} dates'.format((time.time()-start)/60.0,len(select_dates)))

    return mod_dicts


def mod_maker(site_abbrv=None,start_date=None,end_date=None,mode=None,locations=site_dict,HH=12,MM=0,time_step=24,muted=False,lat=None,lon=None,alt=None,save_path=None,ncdf_path=None,keep_latlon_prec=False,**kwargs):
    """
    Inputs:
        - site_abbvr: two letter site abbreviation
        - start_date: YYYYMMDD (set from date_range in parse_args)
        - end_date: YYYYMMDD  (set from date_range in parse_args)
        - mode: one of ncep, merradap42, merradap72, merraglob, fpglob, fpitglob
        - (optional) HH: local hour (default: 12)
        - (optional) MM: minute (default: 0)
        - (optional) time step in hours (default: 24)
        - (optional) muted: if True there won't be print statements expect for warninsg and errors
        - (optional) lat: latitude in [-90,90] range
        - (optional) lon: longitude in [0,360] range
        - (optional) alt: altitude (meters)
    Outputs:
        - .mod files at every time_step within the given date range

    If any of alt/lat/lon is given, the other two must be given too
    """
    if 'merra' in mode:
        prefix = 'MERRA2'
    elif 'ncep' in mode:
        prefix = 'NCEP'
    else:
        prefix = 'FPIT'

    if 'merradap' in mode: # get the earthdata credentials
        try:
            username,account,password = netrc.netrc().authenticators('urs.earthdata.nasa.gov')
        except:
            print('When using MERRA mode, you need a ~/.netrc file to connect to urs.earthdata.nasa.gov')
            sys.exit()

    if lat is not None: # a custom location was given
        locations = {site_abbrv:{'name':'custom site','loc':'custom loc','lat':lat,'lon':lon,'alt':alt}}

    site_dict = tccon_site_info(locations)

    try:
        print('Site:',site_dict[site_abbrv]['name'],site_dict[site_abbrv]['loc'])
    except KeyError:
        print('Wrong 2 letter site abbreviation (check the site_dict dictionary)')
        sys.exit()

    if not muted:
        print('lat,lon,masl:',site_dict[site_abbrv]['lat'],site_dict[site_abbrv]['lon'],site_dict[site_abbrv]['alt'])

    try:
        GGGPATH = os.environ['GGGPATH'] # reads the GGGPATH environment variable
    except:
        if not save_path or not ncdf_path:
            print('No GGGPATH environment variable')
            sys.exit()
    else:
        if not ncdf_path: # if a data path is not given try with GGGPATH/ncdf
            ncdf_path = os.path.join(GGGPATH,'ncdf')

    simple = {'merradap42':'merra','merradap72':'merra','merraglob':'merra','ncep':'ncep','fpglob':'fp','fpitglob':'fpit'}
    if save_path: # using user specified destination folder
        mod_path = os.path.join(save_path,simple[mode],site_abbrv) # .mod files will be saved here
    else:
        if not muted:
            print('GGGPATH =',GGGPATH)
        mod_path = os.path.join(GGGPATH,'models','gnd',simple[mode],site_abbrv)	# .mod files will be saved here

    if not os.path.exists(mod_path):
        os.makedirs(mod_path)
    if not muted:
        print('MOD files will be saved in:',mod_path)

    local_date = start_date + timedelta(hours=HH,minutes=MM) # date with local time
    astropy_date = Time(local_date)
    if not muted:
        print('Starting local time for interpolation:',local_date.strftime('%Y-%m-%d %H:%M'))

    time_step = timedelta(hours=time_step) # time step between mod files; will need to change the mod file naming and gsetup to do sub-3-hourly files
    if not muted:
        print('Time step:',time_step.total_seconds()/3600.0,'hours')

    total_time = end_date-start_date
    n_step = int(total_time.total_seconds()/time_step.total_seconds())
    local_date_list = np.array([start_date+timedelta(hours=HH,minutes=MM)+i*time_step for i in range(n_step)])

    site_moved = False
    if 'time_spans' in site_dict[site_abbrv].keys(): # instruments with different locations for different time periods
        site_moved = True
        for time_span in site_dict[site]['time_spans']:
            if time_span[0]<=UTC_date<time_span[1]:
                site_lat = site_dict[site]['time_spans'][time_span]['lat']
                site_lon_360 = site_dict[site]['time_spans'][time_span]['lon']
                site_lon_180 = site_dict[site]['time_spans'][time_span]['lon_180']
                site_alt = site_dict[site]['time_spans'][time_span]['alt']
                break
    else:
        site_lat = site_dict[site_abbrv]['lat']
        site_lon_360 = site_dict[site_abbrv]['lon']
        site_lon_180 = site_dict[site_abbrv]['lon_180']
        site_alt = site_dict[site_abbrv]['alt']

    rmm = 28.964/18.02	# Ratio of Molecular Masses (Dry_Air/H2O)
    gravity_at_lat, earth_radius_at_lat = gravity(site_lat, site_alt / 1000.0) # used in merra/fp mode

    if 'ncep' in mode:
        DATA = read_ncep(ncdf_path,start_date.year)
        varlist = ['T','H','QV']
    elif 'glob' in mode:
        varlist = ['T','QV','RH','H','EPV','O3','PHIS']
        surf_varlist = ['T2M','QV2M','PS','SLP','TROPPB','TROPPV','TROPPT','TROPT']
        DATA,SURF_DATA = read_global(ncdf_path,mode,site_lat,site_lon_180,gravity_at_lat,varlist,surf_varlist,muted)
    elif 'merradap' in mode: # read all the data first, this could take a while ...
        if not muted:
            print('Reading MERRA2 data via opendap')
        varlist = ['T','QV','RH','H','EPV','O3','PHIS']
        surf_varlist = ['T2M','QV2M','PS','SLP','TROPPB','TROPPV','TROPPT','TROPT']
        DATA,SURF_DATA = read_merradap(username,password,mode,site_lon_180,site_lat,gravity_at_lat,local_date,end_date,time_step,varlist,surf_varlist,muted)

    for local_date in local_date_list:

        if site_moved:
            for time_span in site_dict[site]['time_spans']:
                if time_span[0]<=local_date<time_span[1]:
                    site_lat = site_dict[site]['time_spans'][time_span]['lat']
                    site_lon_360 = site_dict[site]['time_spans'][time_span]['lon']
                    site_lon_180 = site_dict[site]['time_spans'][time_span]['lon_180']
                    site_alt = site_dict[site]['time_spans'][time_span]['alt']
                    break

        astropy_date = Time(local_date)

        utc_offset = timedelta(hours=site_lon_180/15.0)
        UTC_date = local_date - utc_offset

        """
        Interpolation time:
            julday0 is the fractional julian day number of the base time of the dataset: dataset times are in hours since base UTC time
            astropy_date.jd is the fractional julian day number of the current local day
            (astropy_date.jd-julday0)*24.0 = local hours since julday0
        """
        site_tim = (astropy_date.jd-DATA['julday0'])*24.0 - utc_offset.total_seconds()/3600.0 # UTC hours since julday0
        # interpolate the data to the site's location and the desired time
        INTERP_DATA = trilinear_interp(DATA,varlist,site_lon_360,site_lat,site_tim)

        if 'ncep' in mode:
            INTERP_DATA['lev'] = np.copy(DATA['lev'])
            INTERP_DATA['TROPP'] = 0  # tropopause pressure not used with NCEP data
            INTERP_DATA['RH'] = 0 # won't be used, just to feed something to write_mod frh
        else: # merra/geos5
            if 'lev' not in varlist:
                INTERP_DATA['lev'] = np.copy(DATA['lev'])

            # get rid of fill values
            without_fill_IDs = np.where(INTERP_DATA['T']<1e10) # merra/geos fill value is 1e15
            for varname in list(set(varlist+['lev'])):
                try:
                    INTERP_DATA[varname] = INTERP_DATA[varname][without_fill_IDs]
                except IndexError:
                    pass

        if ('merradap' in mode) or ('glob' in mode):
            site_tim = (astropy_date.jd-SURF_DATA['julday0'])*24.0 - utc_offset.total_seconds()/3600.0 # UTC hours since julday0
            INTERP_SURF_DATA = trilinear_interp(SURF_DATA,surf_varlist,site_lon_360,site_lat,site_tim)
            for varname in ['PS','SLP','TROPPB','TROPPV','TROPPT']:
                INTERP_SURF_DATA[varname] = INTERP_SURF_DATA[varname] / 100.0 # convert Pa to hPa

            if 'merradap72' in mode: # merra42 and ncep go from high pressure to low pressure, but merra 72 does the reverse
                # reverse merra72 profiles
                INTERP_DATA['lev'] = INTERP_DATA['PL'] / 100.0
                for varname in list(set(varlist+['lev'])):
                    try:
                        INTERP_DATA[varname] = INTERP_DATA[varname][::-1]
                    except IndexError:
                        pass

        INTERP_DATA['H2O_DMF'] = rmm*INTERP_DATA['QV']/(1-INTERP_DATA['QV']) # Convert specific humidity, a wet mass mixing ratio, to dry mole fraction
        INTERP_DATA['H'] = INTERP_DATA['H']/1000.0	# Convert m to km

        if ('merradap' in mode) or ('glob' in mode):
            INTERP_SURF_DATA['H2O_DMF'] = compute_h2o_dmf(INTERP_SURF_DATA['QV2M'],rmm)
            INTERP_DATA['PHIS'] = INTERP_DATA['PHIS']/1000.0
            # compute surface relative humidity
            INTERP_SURF_DATA['H2O_WMF'] = compute_h2o_wmf(INTERP_SURF_DATA['H2O_DMF']) # wet mole fraction of h2o
            INTERP_SURF_DATA['RH'] = compute_rh(INTERP_SURF_DATA['T2M'],INTERP_SURF_DATA['H2O_WMF'],INTERP_SURF_DATA['PS']) # Fractional relative humidity
            INTERP_SURF_DATA['MMW'] = compute_mmw(INTERP_SURF_DATA['H2O_WMF'])
            INTERP_SURF_DATA['H'] = INTERP_DATA['PHIS']

        ## write the .mod file
        # directions for .mod file name
        if site_lat > 0:
            ns = 'N'
        else:
            ns = 'S'

        if site_lon_180>0:
            ew = 'E'
        else:
            ew = 'W'

        # use the local date for the name of the .mod file
        mod_name = mod_utils.mod_file_name(prefix, local_date, time_step, site_lat, site_lon_180, ew, ns, mod_path, round_latlon=not keep_latlon_prec)
        mod_file_path = os.path.join(mod_path,mod_name)
        if not muted:
            print('\n',mod_name)

        version = 'mod_maker.py   2019-06-20   SR/JL'
        if 'ncep' in mode:
            write_mod(mod_file_path,version,site_lat,data=INTERP_DATA,muted=muted)
        else:
            write_mod(mod_file_path,version,site_lat,data=INTERP_DATA,surf_data=INTERP_SURF_DATA,muted=muted)

        if ((UTC_date+time_step).year!=UTC_date.year) and ('ncep' in mode):
            DATA = read_ncep(ncdf_path,(UTC_date+time_step).year)

    if not muted:
        print(len(local_date_list),'mod files written')


def runlog_driver(runlog, site_abbrv=None, first_date='2000-01-01', **kwargs):
    total_geos_files = 0
    for drange, abbrv, lon, lat, alt in run_utils.iter_runlog_args(runlog, first_date=first_date, site_abbrv=site_abbrv):
        total_geos_files += (drange[1]-drange[0])/timedelta(hours=3)
    print("Equivalent latitude functions will be calculated for a total of {} GEOS files".format(int(total_geos_files)))

    for drange, abbrv, lon, lat, alt in run_utils.iter_runlog_args(runlog, first_date=first_date, site_abbrv=site_abbrv):
        driver(date_range=drange, alt=alt, lon=lon, lat=lat, site_abbrv=abbrv, **kwargs)


def driver(date_range, met_path, chem_path=None, save_path=None, keep_latlon_prec=False, save_in_utc=True, muted=False,
           slant=False, alt=None, lon=None, lat=None, site_abbrv=None, mode=_default_mode, include_chm=True, flat_outdir=False, **kwargs):
    """
    Function that when called executes the full mod maker process as if called from the command line

    The parameters ``alt``, ``lat``, ``lon``, and ``site_abbrv`` set where the .mod file should be made for. They can be
    used in one of three combinations:

        * If none of them are given, .mod files for all of the standard TCCON sites are made.
        * If site_abbrv is given, just the .mod files for that TCCON site are made.
        * If ``alt``, ``lat``, and ``lon`` are also given, the .mod file will be made for those locations. In this mode,
          ``site_abbrv`` just specifies what abbreviation is used in the .mod file names.

    :param date_range: a two-element collection that specifies the start and end datetime of the period to generate
     .mod files for. The end datetime is exclusive.
    :type date_range: list(datetime-like)

    :param met_path: the path to the met files. Different expectations based on ``mode``:
        * If running in "fpit" mode, must contain Nx and Np subdirectories holding the surface and profile files.
        * If running in "fpit-eta" mode, must contain Nx and NV subdirectories holding the surface and profile files.
        * If running in "ncep" mode, must contain "air.yyyy.nc", "hgt.yyyy.nc" and "shum.yyyy.nc" files, where "yyyy" is
          the year. May be omitted (set to ``None``), in which case it defaults to ``$GGGPATH/ncdf`` (``$GGGPATH`` being
          an environmental variable.)
        * If running in "merraglob", "fpglob", or "fpitglob" modes, must contain files with "MERRA", "_fp_" or "_fpit_",
          respectively, in the file names.
        * If running in "merradap" mode, this is not used.
    :type met_path: str

    :param chem_path: the path to the chm (chemistry) files. Only used in "fpit" and "fpit-eta" mode. In either case,
     an "Nv" subdirectory must exist in that directory containing the GEOS FP-IT chm files. If not given, assumes that
     this is the same as the ``met_path``.
    :type chem_path: str or None

    :param save_path: path to save the .mod files to. If ``mode`` is "fpit" or "fpit-eta", then the mod files will be
     saved in "<save_path>/fpit/<site_id>/<vertical or slant>". If one of the other modes, the subdirectories will be
     the same except there will not be a vertical or slant directory. If this parameter is not specified (is ``None``),
     then it will try to use ``$GGGPATH/models/gnd``.
    :type save_path: str

    :param keep_latlon_prec: if ``True``, lat/lon precision in the .mod file names is kept at 2 decimal places. If
     ``False``, it is rounded to the nearest degree.
    :type keep_latlon_prec: bool

    :param save_in_utc: if ``True``, then the time in the .mod file name will be in UTC. If ``False``, it will be in
     local time, estimated using the site longitude.
    :type save_in_utc: bool

    :param muted: set to ``True`` to suppress most logging to the console.
    :type muted: bool

    :param slant: set to ``True`` to output .mod files along a the slant path following the solar zenith angle. Only has
     an effect using the new mod_maker code, i.e. if ``mode`` is "fpit" or "fpit-eta".
    :type slant: bool

    :param mode: which met data mode to use. Options are:

        * "ncep" - uses yearly NCEP met files
        * "merraglob", "fpglob", "fpitglob" - uses MERRA, GEOS-FP, or GEOS-FPIT files with old-style time interpolation.
        * "merradap42", "merradap72" - uses MERRA data via DAP remote access protocol.
        * "fpit", "fpit-eta" - uses GEOS-FPIT files on fixed pressure levels or eta levels, respectively, with new-style
          8x per day files.
    :type mode: str

    :param alt: altitude above sea (?) level in meters. If not given, the standard altitude specified for the TCCON
     site(s) is used.
    :type alt: None or float

    :param lon: the longitude at which the .mod file should be produced. If not given, the standard longitude specified
     for the TCCON site(s) is used.
    :type lon: None or float

    :param lat: the latitude at which the .mod file should be produced. If not given, the standard latitude specified
     for the TCCON site(s) is used.
    :type lat: None or float

    :param site_abbrv: the two-letter abbreviation of the TCCON site for which to generate .mod files. If not specified
     and ``lat`` is not specified, then mod files are generated for all predefined TCCON sites. If specified and
     ``lat`` not specified, then just .mod files for that TCCON site are made. If ``lat`` is specified, then this is the
     abbreviation used in the .mod file names. If not given in this case, it defaults to "xx".
    :type site_abbrv: None or str

    :param include_chm: set to ``True`` to include variables from the "chm" files as well. Currently this will only be
     CO, but requires that you have the "chm" files available. See ``chem_path`` for the rules about specifying where
     thos files are located.
    :type include_chm: bool

    :param flat_outdir: set to ``True`` to save .mod files directly into `save_path` rather than organizing into 
     subdirectories by product, site, and vertical/slant.
    :type flat_outdir: bool

    :param kwargs: unused, swallows extra keyword arguments

    :return: nothing, writes .mod files to the output directory.
    """
    start_date, end_date = date_range
    site_abbrv, lat, lon, alt = check_site_lat_lon_alt(site_abbrv, lat=lat, lon=lon, alt=alt)

    # The modmaker functions are not set up to allow multiple custom lat/lon/alts to be passed, so if there are multiple
    # lat/lon/alts to be made, then we have to iterate over them. In the case of the new mod maker, we want to only
    # generate the eq. lat. interpolation functions once, because those take a long time.
    if mode in _old_modmaker_modes:
        for this_abbrv, this_lat, this_lon, this_alt in zip(site_abbrv, lat, lon, alt):
            mod_maker(site_abbrv=this_abbrv, start_date=start_date, end_date=end_date, locations=site_dict,
                      HH=12, MM=0, time_step=24, muted=muted, lat=this_lat, lon=this_lon, alt=this_alt,
                      save_path=save_path, ncdf_path=met_path, keep_latlon_prec=keep_latlon_prec, mode=mode)
    elif mode in _new_modmaker_modes:
        if mode in _new_fixedp_modes:
            eqlat_fxn = equivalent_latitude_functions_geos
            native_files = False
        elif mode in _new_native_modes:
            eqlat_fxn = equivalent_latitude_functions_native_geos
            native_files = True
        else:
            raise NotImplementedError('No equivalent latitude function defined for mode == "{}"'.format(mode))

        chem_vars = ('CO',) if include_chm else tuple()
        product = mod_utils.mode_to_product(mode)
        func_dict = eqlat_fxn(GEOS_path=met_path, start_date=start_date, end_date=end_date, muted=muted)

        for this_abbrv, this_lat, this_lon, this_alt in zip(site_abbrv, lat, lon, alt):
            mod_maker_new(start_date=start_date, end_date=end_date, func_dict=func_dict, GEOS_path=met_path,
                          chem_path=chem_path, chem_variables=chem_vars, slant=slant, locations=site_dict, muted=muted,
                          lat=this_lat, lon=this_lon, alt=this_alt, site_abbrv=this_abbrv, save_path=save_path, product=product,
                          keep_latlon_prec=keep_latlon_prec, save_in_utc=save_in_utc, native_files=native_files, flat_outdir=flat_outdir)
    else:
        raise ValueError('mode "{}" is not one of the allowed values: {}'.format(
            mode, ', '.join(_old_modmaker_modes + _new_modmaker_modes)
        ))


if __name__ == "__main__": # this is only executed when the code is used directly (e.g. not executed when imported from another python code)

    arguments = parse_args()
    driver(**arguments)
