from datetime import datetime as dtime
from itertools import product
from netCDF4 import Dataset
import numpy as np

from ginput.common_utils import mod_utils
from ginput.mod_maker import mod_maker, tccon_sites


def test_git_info_graceful_exit():
    # All that this test requires is that this does not crash -
    # we deliberately tell it to use a git path that will not
    # exist
    mod_utils._git_commit_info(git_exec='/not/a/bin/git')
    mod_utils._git_is_commit_clean(git_exec='/not/a/bin/git')


def test_hg_info_graceful_exit():
    # All that this test requires is that this does not crash -
    # we deliberately tell it to use an hg path that will not
    # exist
    mod_utils._hg_commit_info(hg_exec='/not/a/bin/hg')
    mod_utils._hg_is_commit_clean(hg_exec='/not/a/bin/hg')


def test_find_date_substring():
    # Test ideal cases of the three formats of datestring,
    # then also test on an example .mod and .vmr filename
    test_datestrs = {'20180101': '%Y%m%d',
                     '2018010112': '%Y%m%d%H', '20180101_1200': '%Y%m%d_%H%M'}
    test_dates = {k: dtime.strptime(k, v)
                  for k, v in test_datestrs.items()}
    test_filenames = {k: 'afile_{}.txt'.format(k) for k in test_datestrs}

    vmr_date = dtime(2018, 2, 1, 6)
    vmr_datestr = vmr_date.strftime('%Y%m%d%H')
    vmr_name = mod_utils.vmr_file_name(vmr_date, 0., 0.)

    mod_date = dtime(2018, 3, 1, 21)
    mod_datestr = mod_date.strftime('%Y%m%d%H')
    mod_name = mod_utils.mod_file_name_for_priors(mod_date, 0.0, 0.0)

    test_filenames.update({vmr_datestr: vmr_name, mod_datestr: mod_name})
    test_dates.update({vmr_datestr: vmr_date, mod_datestr: mod_date})

    for dstr, fname in test_filenames.items():
        date = test_dates[dstr]
        rdstr = mod_utils.find_datetime_substring(fname)
        rdate = mod_utils.find_datetime_substring(fname, out_type=dtime)
        assert dstr == rdstr
        assert date == rdate


def test_find_latlon_substring():
    test_lats = [-45., -5.,  5., 45, -30.75, -30.25, 30.25, 30.75]
    test_lons = [-150., -50., -5, 5., 50.,
                 150., -30.75, -30.25, 30.25, 30.75]

    for lon, lat in product(test_lons, test_lats):
        for keep_prec in (True, False):
            mod_name = mod_utils.mod_file_name_for_priors(
                dtime(2018, 1, 1), lat, lon, round_latlon=not keep_prec)
            vmr_name = mod_utils.vmr_file_name(
                dtime(2018, 1, 1), lon, lat, keep_latlon_prec=keep_prec)
            mod_lon = mod_utils.find_lon_substring(
                mod_name, to_float=True)
            mod_lat = mod_utils.find_lat_substring(
                mod_name, to_float=True)
            vmr_lon = mod_utils.find_lon_substring(
                vmr_name, to_float=True)
            vmr_lat = mod_utils.find_lat_substring(
                vmr_name, to_float=True)

            if lon % 1 != 0 and not keep_prec:
                assert np.isclose(mod_lon, round(lon))
                assert np.isclose(vmr_lon, round(lon))
            else:
                assert np.isclose(mod_lon, lon)
                assert np.isclose(vmr_lon, lon)

            if lat % 1 != 0 and not keep_prec:
                assert np.isclose(mod_lat, round(lat))
                assert np.isclose(vmr_lat, round(lat))
            else:
                assert np.isclose(mod_lat, lat)
                assert np.isclose(vmr_lat, lat)


def test_potential_temperature():
    # potential temperatures calculated using
    # http://www.eumetrain.org/data/2/28/Content/ptcalc.htm
    temp_C = (15.0,   15.0,   15.0,    -5.0,   -
              5.0,   -5.0,    -25.0, -25.0,   -25.0)
    pres_hpa = (1000.0, 100.0,  10.0,    1000.0,
                100.0,  10.0,    1000.0, 100.0,  10.0)
    theta_K = (288.15, 556.70, 1075.52, 268.15,
               518.06, 1000.87, 248.15, 479.42, 926.22)

    for t, p, theta in zip(temp_C, pres_hpa, theta_K):
        t = t + 273.15
        th_chk = mod_utils.calculate_potential_temperature(p, t)
        assert abs(theta - th_chk) < 0.01


def test_lat_lon_interp(lat_lon_file, test_date):
    sites = tccon_sites.tccon_site_info_for_date(test_date)
    failed_sites = []
    for sid, info in sites.items():
        lat = info['lat']
        lon = info['lon_180']
        new_lat, new_lon = _lat_lon_interp_internal(lat_lon_file, lat, lon)
        if not np.isclose(lat, new_lat) and np.isclose(lon, new_lon):
            failed_sites.append(sid)

    msg = "{nfail}/{tot} sites' interpolated lat/lon do not match their original: {sites}".format(
        nfail=len(failed_sites), tot=len(sites), sites=', '.join(failed_sites))
    assert len(failed_sites) == 0, msg


def _lat_lon_interp_internal(geos_file, site_lat, site_lon):
    with Dataset(geos_file) as ds:
        ids = mod_maker.querry_indices(ds, site_lat, site_lon, None, None)
        lat = ds['lat'][:].filled(np.nan)
        lon = ds['lon'][:].filled(np.nan)
        shape = ds['PS'][:].squeeze().shape
        lat_array = np.broadcast_to(lat.reshape(-1, 1), shape)
        lon_array = np.broadcast_to(lon.reshape(1, -1), shape)

        new_lat = mod_maker.lat_lon_interp(
            lat_array, lat, lon, [site_lat], [site_lon], [ids])[0]
        new_lon = mod_maker.lat_lon_interp(
            lon_array, lat, lon, [site_lat], [site_lon], [ids])[0]
        return new_lat.item(), new_lon.item()
