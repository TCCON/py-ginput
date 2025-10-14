from datetime import datetime as dtime
from itertools import product
import netCDF4 as ncdf
import numpy as np
import os
import unittest

from ..common_utils import mod_utils
from ..mod_maker import mod_maker, tccon_sites

from . import test_utils


class TestGinputUtils(unittest.TestCase):
    def test_find_date_substring(self):
        # Test ideal cases of the three formats of datestring, then also test on an example .mod and .vmr filename
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
            with self.subTest(filename=fname, datestring=dstr, date=date):
                rdstr = mod_utils.find_datetime_substring(fname)
                rdate = mod_utils.find_datetime_substring(
                    fname, out_type=dtime)
                self.assertEqual(dstr, rdstr)
                self.assertEqual(date, rdate)

    def test_find_latlon_substring(self):
        test_lats = [-45., -5.,  5., 45, -30.75, -30.25, 30.25, 30.75]
        test_lons = [-150., -50., -5, 5., 50.,
                     150., -30.75, -30.25, 30.25, 30.75]

        for lon, lat in product(test_lons, test_lats):
            for keep_prec in (True, False):
                mod_name = mod_utils.mod_file_name_for_priors(
                    dtime(2018, 1, 1), lat, lon, round_latlon=not keep_prec)
                vmr_name = mod_utils.vmr_file_name(
                    dtime(2018, 1, 1), lon, lat, keep_latlon_prec=keep_prec)
                with self.subTest(lon=lon, lat=lat, mod_name=mod_name, vmr_name=vmr_name):
                    mod_lon = mod_utils.find_lon_substring(
                        mod_name, to_float=True)
                    mod_lat = mod_utils.find_lat_substring(
                        mod_name, to_float=True)
                    vmr_lon = mod_utils.find_lon_substring(
                        vmr_name, to_float=True)
                    vmr_lat = mod_utils.find_lat_substring(
                        vmr_name, to_float=True)

                    if lon % 1 != 0 and not keep_prec:
                        self.assertAlmostEqual(mod_lon, round(lon))
                        self.assertAlmostEqual(vmr_lon, round(lon))
                    else:
                        self.assertAlmostEqual(mod_lon, lon)
                        self.assertAlmostEqual(vmr_lon, lon)

                    if lat % 1 != 0 and not keep_prec:
                        self.assertAlmostEqual(mod_lat, round(lat))
                        self.assertAlmostEqual(vmr_lat, round(lat))
                    else:
                        self.assertAlmostEqual(mod_lat, lat)
                        self.assertAlmostEqual(vmr_lat, lat)

    def test_potential_temperature(self):
        # potential temperatures calculated using http://www.eumetrain.org/data/2/28/Content/ptcalc.htm
        temp_C = (15.0,   15.0,   15.0,    -5.0,   -
                  5.0,   -5.0,    -25.0, -25.0,   -25.0)
        pres_hpa = (1000.0, 100.0,  10.0,    1000.0,
                    100.0,  10.0,    1000.0, 100.0,  10.0)
        theta_K = (288.15, 556.70, 1075.52, 268.15,
                   518.06, 1000.87, 248.15, 479.42, 926.22)

        for t, p, theta in zip(temp_C, pres_hpa, theta_K):
            with self.subTest(t=t, p=p, theta=theta):
                t = t + 273.15
                th_chk = mod_utils.calculate_potential_temperature(p, t)
                self.assertLess(abs(theta - th_chk), 0.01)


class TestModMakerUtils(unittest.TestCase):
    @staticmethod
    def _test_lat_lon_interp_internal(site_lat, site_lon):
        geos_file = test_utils.lat_lon_file
        with ncdf.Dataset(geos_file) as ds:
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

    def test_lat_lon_interp(self):
        sites = tccon_sites.tccon_site_info_for_date(test_utils.test_date)
        failed_sites = []
        for sid, info in sites.items():
            lat = info['lat']
            lon = info['lon_180']
            new_lat, new_lon = self._test_lat_lon_interp_internal(lat, lon)
            if not np.isclose(lat, new_lat) and np.isclose(lon, new_lon):
                failed_sites.append(sid)

        msg = "{nfail}/{tot} sites' interpolated lat/lon do not match their original: {sites}".format(
            nfail=len(failed_sites), tot=len(sites), sites=', '.join(failed_sites))
        self.assertTrue(len(failed_sites) == 0, msg=msg)


if __name__ == '__main__':
    unittest.main()
