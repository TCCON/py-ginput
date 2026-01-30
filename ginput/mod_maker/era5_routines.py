import numpy as np 
import xarray as xr

import sys 
import glob 
import os 
from datetime import datetime, timedelta 


from ..common_utils import mod_utils

from ..common_utils.gph2alt import convert_gph_to_alt, auto_broadcast
from ..common_utils.tropopause_routines import computetropopauses


##Q    I   specific humidity g kg-1
def qv2mixr(Q):
	return Q/(1.-Q/1000.)


###T    I   kelvins
###returns the saturation vapor pressure in hPa
def esat(T):

    T0 = 0.0       
    e1 = 1013.250
    TK = 273.15

    Tp = T + T0 

    esat = e1 * 10 ** (
        10.79586 * (1.0 - TK / Tp)
        - 5.02808 * np.log10(Tp / TK)
        + 1.50474e-4 * (1.0 - 10 ** (-8.29692 * (Tp / TK - 1.0)))
        + 0.42873e-3 * (10 ** (4.76955 * (1.0 - TK / Tp)) - 1.0)
        - 2.2195983
    )

    return esat

def eice(T):

	T0 = 0.0        # Kelvin
	A = -2663.5
	B = 12.537

	logp = A / (T + T0) + B

	# 10^logp / 100  -> conversion to hPa
	p = (10 ** logp) / 100.0

	return p




## mixr   I    g h2o per kg dry air 
## p      I    pressure hPa 
## T      I    temperature K 
## ice    I    set ice to True to return RH over ice 
## rh     O    relative humidity in %
def mixr2rh(mixr, p, T, ice = False):
	es=esat(T)
	if ice: es = eice(T)

	Mw=18.0160 ## molecular weight of water
	Md=28.9660 ## molecular weight of dry air
	fact=mixr/1000.*Md/Mw

	rh = p/es*fact/(1+fact)*100.
	return rh




def getera5pre(file2d):
    
    a = np.asarray([0.00000, 2.00037, 3.10224, 4.66608, 6.82798, 9.74697, 13.6054, 
               18.6089, 24.9857, 32.9857, 42.8792, 54.9555, 69.5206, 86.8959, 
               107.416, 131.426, 159.279, 191.339, 227.969, 269.540, 316.421, 
               368.982, 427.592, 492.616, 564.413, 643.340, 729.744, 823.968, 
               926.345, 1037.20, 1156.85, 1285.61, 1423.77, 1571.62, 1729.45, 
               1897.52, 2076.10, 2265.43, 2465.77, 2677.35, 2900.39, 3135.12, 
               3381.74, 3640.47, 3911.49, 4194.93, 4490.82, 4799.15, 5119.90, 
               5452.99, 5798.34, 6156.07, 6526.95, 6911.87, 7311.87, 7727.41, 
               8159.35, 8608.53, 9076.40, 9562.68, 10066.0, 10584.6, 11116.7, 
               11660.1, 12211.5, 12766.9, 13324.7, 13881.3, 14432.1, 14975.6, 
               15508.3, 16026.1, 16527.3, 17008.8, 17467.6, 17901.6, 18308.4, 
               18685.7, 19031.3, 19343.5, 19620.0, 19859.4, 20059.9, 20219.7, 
               20337.9, 20412.3, 20442.1, 20425.7, 20361.8, 20249.5, 20087.1, 
               19874.0, 19608.6, 19290.2, 18917.5, 18489.7, 18006.9, 17471.8, 
               16888.7, 16262.0, 15596.7, 14898.5, 14173.3, 13427.8, 12668.3, 
               11901.3, 11133.3, 10370.2, 9617.52, 8880.45, 8163.38, 7470.34, 
               6804.42, 6168.53, 5564.38, 4993.80, 4457.38, 3955.96, 3489.23, 
               3057.27, 2659.14, 2294.24, 1961.50, 1659.48, 1387.55, 1143.25, 
               926.508, 734.992, 568.062, 424.414, 302.477, 202.484, 122.102, 
               62.7812, 22.8359, 3.75781, 0.00000, 0.00000])
    
    
    
    b = np.asarray([0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
               0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
               0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
               0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
               0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
               0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
               0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 
               0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 7.00000e-06, 
               2.40000e-05, 5.90000e-05, 0.000112000, 0.000199000, 0.000340000, 
               0.000562000, 0.000890000, 0.00135300, 0.00199200, 0.00285700, 
               0.00397100, 0.00537800, 0.00713300, 0.00926100, 0.0118060, 
               0.0148160, 0.0183180, 0.0223550, 0.0269640, 0.0321760, 0.0380260, 
               0.0445480, 0.0517730, 0.0597280, 0.0684480, 0.0779580, 0.0882860, 
               0.0994620, 0.111505, 0.124448, 0.138313, 0.153125, 0.168910, 
               0.185689, 0.203491, 0.222333, 0.242244, 0.263242, 0.285354, 
               0.308598, 0.332939, 0.358254, 0.384363, 0.411125, 0.438391, 
               0.466003, 0.493800, 0.521619, 0.549301, 0.576692, 0.603648, 
               0.630036, 0.655736, 0.680643, 0.704669, 0.727739, 0.749797, 
               0.770798, 0.790717, 0.809536, 0.827256, 0.843881, 0.859432, 
               0.873929, 0.887408, 0.899900, 0.911448, 0.922096, 0.931881, 
               0.940860, 0.949064, 0.956550, 0.963352, 0.969513, 0.975078, 
               0.980072, 0.984542, 0.988500, 0.991984, 0.995003, 0.997630, 1.00000])
    
    
    with xr.open_dataset(file2d) as ds:
        lnsp = np.squeeze(ds['lnsp'].to_numpy())
        z    = np.squeeze(ds['z'].to_numpy())
        time = np.asarray(ds['time'])
        lat  = np.asarray(ds['latitude'])
        lon  = np.asarray(ds['longitude'])


    
    sp = np.exp(lnsp)  # surface pressure [Pa]

    # Half-level pressure: p_half = A + B * sp
    # Broadcast to (half, lat, lon)
    p_half = a[:, None, None] + b[:, None, None] * sp[ None, :, :]
    
    # # Full-level pressure = mean of adjacent half levels; convert to hPa
    p_full_hpa = 0.5 * (p_half[ 1:, :, :] + p_half[ :-1, :, :]) / 100.0


    oo = {'time': time, 'lat': lat, 'lon': lon, 'pre':p_full_hpa, 'sp':sp, 'z':z, 'phalf':p_half }

    return oo





def getera5data(file3d, file2d, quickPV = True):

    
    pp = getera5pre(file2d)

    ps = pp['sp']    #Pa
    
    pre = pp['pre'] #hPa
        
    with xr.open_dataset(file3d) as ds:
        lat = ds['latitude'].to_numpy()
        lon = ds['longitude'].to_numpy()
        tem = np.squeeze(ds['t'].to_numpy())   #Kelvins
        vor = np.squeeze(ds['vo'].to_numpy())   #relative vorticity s^-1
        q   = np.squeeze(ds['q'].to_numpy())  ##kg kg-1 specific humidity 
        o3  = np.squeeze(ds['o3'].to_numpy())  ##kg kg-1 ozone mass mixing ratio
        
    lat[np.abs(lat) < 0.001] = 0.0


    
    #### constants
    g = 9.81
    omega = 1.454441e-4  ##angular velocity of the Earth s-1
    
    rGas = 8.314472          # J/mol/K
    MMair = 28.964           # g/mol, molar mass of air
    MMh2o = 18.02            # g/mol, molar mass of water vapor

    
    # Gas constants
    ##rDry = rGas / MMh2o * 1000.   ## this gives Rvapor=461.402
    rDry = rGas / MMair * 1000.     ## this gives Rdry=287.062
    e = MMh2o / MMair

    theta = mod_utils.calculate_potential_temperature(pre, tem)
    
    # theta = tem * (1000.0 / pre) ** (2.0 / 7.0)
    theta[pre==0] = np.nan
    
    nlev = pre.shape[0]
    nlat = pre.shape[1]
    nlon = pre.shape[2]
    
    ###PV from relative vorticity
    pvort = omega * np.sin(np.deg2rad(lat))                       # (lat,)
    pvort3 = np.broadcast_to(pvort.reshape(1, nlat, 1), (nlev, nlat, nlon)).copy()
    
    pre = pre*100. #pressure in pascals
    
    
    if quickPV == False:
        pv    = np.zeros((nlev, nlat, nlon))
        pv[:] = np.nan
        for ilon in np.arange(nlon):
            for ilat in np.arange(nlat):
              pv[:,ilat,ilon] = -g * (vor[:,ilat,ilon] + pvort3[:,ilat,ilon]) * np.gradient(theta[:,ilat,ilon], pre[:,ilat,ilon], edge_order=2)   #dTheta/dp 
        
    
    if quickPV == True:
        #The code below vectorize the loops but it provides slightly different values
        #This method is around ten times faster
        dtheta_dp = np.full_like(theta, np.nan, dtype=np.double)
        
        # First compute forward/backward differences (level-wise, per column)
        dp_fwd     = pre[1:]   - pre[:-1]      # shape (nlev-1, nlat, nlon)
        dθ_fwd     = theta[1:] - theta[:-1]
        
        # Interior points (second-order for nonuniform grids; matches np.gradient with x)
        # Weighted average of adjacent slopes:
        # f'(i) = ( s_plus * (x[i]-x[i-1]) + s_minus * (x[i+1]-x[i]) ) / (x[i+1]-x[i-1])
        # where s_plus  = (f[i+1]-f[i])/(x[i+1]-x[i])
        #       s_minus = (f[i]-f[i-1])/(x[i]-x[i-1])
        num_left   = dθ_fwd[:-1] / dp_fwd[:-1] * (pre[1:-1] - pre[:-2])
        num_right  = dθ_fwd[1:]  / dp_fwd[1:]  * (pre[2:]   - pre[1:-1])
        den_mid    = (pre[2:] - pre[:-2])
        dtheta_dp[1:-1] = (num_left + num_right) / den_mid
        
        # Leading edge (second-order one-sided, nonuniform 3-point forward)
        h0 = pre[1] - pre[0]            # Dp_0
        h1 = pre[2] - pre[1]            # Dp_1
        den0 = h0 * (h0 + h1)
        dtheta_dp[0] = ( -(2*h0 + h1) / den0 * theta[0]
                         + (h0 + h1) / (h0*h1) * theta[1]
                         -  h0       / (h1*(h0 + h1)) * theta[2] )
        
        # Trailing edge (second-order one-sided, nonuniform 3-point backward)
        hm1 = pre[-1] - pre[-2]         # p_{N-2}
        hm2 = pre[-2] - pre[-3]         # p_{N-3}
        denN = hm1 * (hm1 + hm2)
        dtheta_dp[-1] = (  (2*hm1 + hm2) / denN * theta[-1]
                           - (hm1 + hm2) / (hm1*hm2) * theta[-2]
                           +  hm1        / (hm2*(hm1 + hm2)) * theta[-3] )
        
        pv = -g * (vor + pvort3) * dtheta_dp

    
    lat_res = np.abs(np.nanmean(np.gradient(lat)))
    lon_res = np.abs(np.nanmean(np.gradient(lon)))


    
    # --------  GPH
    
    # Virtual temperature
    tVirt = tem * (q + e) / (e * (1.0 + q))


    pHalf = pp['phalf']  ##in pascals
    z     = pp['z']

    # Natural log of pHalf
    lpHalf = np.log(pHalf)

    # Vertical differences of log(pHalf)
    diff = lpHalf[1:, :, :] - lpHalf[:-1, :]

    
    f = np.empty_like(lpHalf, dtype=float)

    # IDL: f[*, *, 0:-2] = rDry * tVirt * diff

    f[:-1, :, :] = rDry * tVirt * diff

    f[-1, :, :] = z

    # phiHalf2 = Reverse(Total(Reverse(f, 3), 3, /CUMULATIVE, /NAN), 3)
    # Reverse along vertical (last axis), cumulative sum, then reverse back
    f_rev = np.flip(f, axis=0)
    dmy = np.nancumsum(f_rev, axis=0)
    phiHalf2 = np.flip(dmy, axis=0)


    p_up = pHalf[1:, :,:]
    p_dn = pHalf[:-1, :,:]

    gph = phiHalf2[1:,:,:] - rDry * tVirt * np.log((p_up + p_dn) / (2.0 * p_up))

    gph = gph/ g
    gph = gph / 1000. ##changing to km
    
    ##we need alt to compute the tropopause
    latB = np.broadcast_to(auto_broadcast(lat, gph), gph.shape)
    alt = convert_gph_to_alt(gph, latB)
    
        
    pre = pre/100 ##hPa
    pv = pv * 1e6 ##to match units on MERRA2



    
    ###varlist = ['T','QV','RH','H','EPV','O3','PHIS', 'lev']

    ##shifting lon to match merra2
    lon_shifted = ((lon + 180) % 360) - 180
    ixlon = np.argsort(lon_shifted)
    
    lon = lon[ixlon]
    pre = pre[:,:,ixlon]
    tem = tem[:,:,ixlon]
    q   = q[:,:,ixlon]
    o3  = o3[:,:,ixlon]
    theta = theta[:,:,ixlon]
    pv = pv[:,:,ixlon]
    vor = vor[:,:,ixlon]
    gph = gph[:,:,ixlon]
    alt = alt[:,:,ixlon]
   

    ##shifting lat to match merra2
    # The ERA5 fields are ordered space-to-surface. The equivalent latitude calculation *may* be okay
    # with that, but I felt it was safer to just go ahead and flip them, as performed for the MERRA2 field
    lat   = np.flip(lat)
    pre   = np.flip(pre,   axis = (0,1))
    tem   = np.flip(tem,   axis = (0,1))
    q     = np.flip(q,     axis = (0,1))
    o3    = np.flip(o3,    axis = (0,1))
    theta = np.flip(theta, axis = (0,1))
    pv    = np.flip(pv,    axis = (0,1))
    vor   = np.flip(vor,   axis = (0,1))
    gph   = np.flip(gph,   axis = (0,1))
    alt   = np.flip(alt,   axis = (0,1))



    sgph = gph[0,:,:]
    phis = sgph*1000. * 9.80665  ##m2 s-2

    
    area = mod_utils.calculate_area(lat, lon, lat_res, lon_res, muted=False)
   
    # eql = mod_utils.calculate_eq_lat(np.flip(pv, axis = 0), np.flip(theta, axis = 0), area) 

    eql = mod_utils.calculate_eq_lat(pv, theta, area) 

    
    t2m    = np.copy(ps)
    qv2m   = np.copy(ps)
    slp    = np.copy(ps)
    
    troppb = np.copy(ps)
    troppv = np.copy(ps)
    troppt = np.copy(ps)
    tropt  = np.copy(ps)

    # tt = computetropopauses(tem, alt, pre, pv, 0)

    # troppb = tt['blnpre']
    # troppv = tt['dynpre']
    # troppt = tt['wmopre']
    # tropt  = tt['blntem']


    alt = alt*1000. ##changing to meters
    gph = gph*1000. ##changing to meters    


    
    mixr = qv2mixr(q*1000.)
    
    dmy = mixr2rh(mixr, pre, tem)
    rhf = dmy / 100.
        
    surf = {'T2M':t2m,'QV2M':qv2m, 'SLP':slp, 'PS':ps, 'TROPPB':troppb, 'TROPPV':troppv, 'TROPPT':troppt, 'TROPT':tropt}

    #the first variable needs to be T so that interp_geos_data_to_sites finds the correct "default_geos_var"
    ee = {'T':tem, 'QV':q, 'O3':o3, 'theta':theta, 'EPV':pv, 'gph':gph, 'H':alt, 'PHIS':phis,'RH':rhf,
          'vor':vor, 'eql':eql, 'area':area, 
          'lat':lat, 'lon':lon, 'lev':pre, 'lat_res':lat_res, 'lon_res':lon_res,'surf':surf}


    return ee


def getera5files_perdate(date2use, path = 'era5/'):

    yyyymmdd = date2use.strftime('%Y%m%d')
    dates = []
    files3d = []
    files2d = []
   
    syntimes = [0,3,6,9,12,15,18,21]  #ERA5 synoptic times to use (only every 3 hours)

    for ss in syntimes:
        fls3d = sorted(glob.glob(os.path.join(path, 'era5*'+yyyymmdd+'_'+str(ss).zfill(2)+'00_3d.nc4'))) 
        fls2d = sorted(glob.glob(os.path.join(path, 'era5*'+yyyymmdd+'_'+str(ss).zfill(2)+'00_2d.nc4'))) 


            
        if len(fls3d) ==1 & len(fls2d) ==1:
            dates.append(date2use+ timedelta(hours=ss))
            files3d.append(fls3d[0])
            files2d.append(fls2d[0])
        else:
            breakpoint()
            print('ERA5 files are missing')
            sys.exit(0)
            
            
    dates = np.asarray(dates)
    files3d = np.asarray(files3d)
    files2d = np.asarray(files2d)
    oo = {'dates':dates, 'files3d':files3d, 'files2d':files2d}

    return oo

def getera5files(start_date, end_date, path = 'era5/'):

    delta = end_date - start_date  

    dates = [start_date + timedelta(int(d)) for d in np.arange(delta.days)]

    oo = []
    for d in dates:
        oo.append(getera5files_perdate(d, path = path))


    dat  = np.concatenate([o['dates']   for o in oo])
    f3d  = np.concatenate([o['files3d'] for o in oo])
    f2d  = np.concatenate([o['files2d'] for o in oo])
    oo = {'dates':dat, 'files3d':f3d, 'files2d':f2d}

    return oo
    


def equivalent_latitude_functions_native_era5(GEOS_path = 'era5/', start_date=None, end_date=None, muted = False):
    

    e5files = getera5files(start_date, end_date, path = GEOS_path)

    func_dict = dict()
    for d, f2, f3 in zip(e5files['dates'], e5files['files2d'], e5files['files3d']):
        print(d, f2, f3)
        aa = getera5data(f3, f2)

        func_dict[d] = aa['eql']

    return func_dict
