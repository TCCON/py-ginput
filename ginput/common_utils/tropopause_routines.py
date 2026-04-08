
# import sys 
import numpy as np
# import matplotlib.pyplot as plt

# from tqdm import tqdm
# import pickle

# from scipy.io import readsav
# from scipy.interpolate import interp1d

# from scipy.interpolate import UnivariateSpline

from tqdm import tqdm 


def static_stability(dtdz, tprf):

    H = 6800.0 ## scale height in meters
    R0 = 287.0 ## gas constant for dry air
    kappa = 2./7. ## cp/cv, ratio of heap capacities.

    staticStability = R0/H* ( dtdz  + (kappa* tprf/H) ) 

    return staticStability




def wmo_tropopauses(tprof, zprof, pprof, pvprof = None, thprof = None,
    plow=500.0, phigh=10.0, zlow=3.5, dttest=-2.0, verbose = False):


    # ---- Basic argument checks ----
    if tprof is None or zprof is None or pprof is None:
        raise ValueError("One of tprof, zprof or pprof is missing.")



    tprof = np.copy(np.asarray(tprof, dtype=float))
    zprof = np.copy(np.asarray(zprof, dtype=float))
    pprof = np.copy(np.asarray(pprof, dtype=float))

    nlevs = pprof.size
    surfprof = np.arange(nlevs, dtype=float)

    have_pv = pvprof is not None
    if have_pv:
        pvprof = np.copy(np.asarray(pvprof, dtype=float))



    baddata = np.nan


    # ---- Potential temperature (if needed) ----
    if thprof is None or np.size(thprof) != nlevs:
        thprof_arr = tprof * (1000.0 / pprof) ** (2.0 / 7.0)
        bad = np.where( np.isnan(tprof) | np.isnan(pprof) )[0]
        if bad.size > 0:
            thprof_arr[bad] = baddata
        thprof = thprof_arr
    else:
        thprof = np.copy(np.asarray(thprof, dtype=float))



	#the profiles need to be from the surface to space
    if zprof[0] > zprof[-1]:
        tprof  = np.flip(tprof)
        zprof  = np.flip(zprof)
        pprof  = np.flip(pprof)
        pvprof = np.flip(pvprof)
        thprof = np.flip(thprof)




    # Allocate output arrays (max 4 WMO tropopauses)
    wmosurf  = np.full(4, baddata, dtype=float)
    wmoalt   = np.full(4, baddata, dtype=float)
    wmotemp  = np.full(4, baddata, dtype=float)
    wmopres  = np.full(4, baddata, dtype=float)
    wmopv    = np.full(4, baddata, dtype=float)
    wmotheta = np.full(4, baddata, dtype=float)
    wmoss    = np.full(4, baddata, dtype=float)
    numwmo   = 0

    out = {'wmosurf':wmosurf, 'wmoalt':wmoalt, 'wmotemp':wmotemp, 
        'wmopres':wmopres, 'wmopv':wmopv, 'wmotheta':wmotheta, 'wmoss':wmoss, 
        'numwmo':numwmo, 'plow':plow, 'phigh':phigh, 'zlow':zlow, 'dttest':dttest, 'baddata':baddata}

    # ---- Mask out bad data ----

    # good_mask = ((np.abs(tprof - baddata) > 1.0) &
    #              (np.abs(zprof - baddata) > 1.0) &
    #              (np.abs(pprof - baddata) > 1.0))

    good_mask = np.isfinite(tprof) & np.isfinite(zprof) & np.isfinite(pprof) & np.isfinite(thprof)
    if have_pv:
    	good_mask = np.isfinite(tprof) & np.isfinite(zprof) & np.isfinite(pprof) & np.isfinite(thprof) & np.isfinite(pvprof)


    gprof = np.where(good_mask)[0]
    ngprof = gprof.size

    if ngprof <= 1:
        if verbose:
            print("-----ERROR -- NOT ENOUGH GOOD INPUT POINTS FOR WMO")
        return out

    # Subset to good levels
    tgprof    = tprof[gprof]
    zgprof    = zprof[gprof]
    pgprof    = pprof[gprof]
    thgprof   = thprof[gprof]
    surfgprof = surfprof[gprof]
    if have_pv:
        pvgprof   = pvprof[gprof]

    # ---- WMO tropopause section ----

    dgprof = np.gradient(tgprof, zgprof, edge_order = 1)  


    ssgprof = static_stability(dgprof * 1e-3, tgprof)

    if np.size(ssgprof) == 1 and not np.isfinite(ssgprof):
        ssgprof = np.full(tgprof.size, baddata, dtype=float)
        if verbose:
            print("Calculation of Static Stability failed!")


    # Apply  limits
    profu_mask = ((zgprof > zlow) & (pgprof > phigh) & (pgprof < plow))
    
    profu = np.where(profu_mask)[0]
    nprofu = profu.size

    if nprofu <= 1:
        if verbose:
            print("-----ERROR, NOT ENOUGH POINTS WITHIN ALT, PRE LIMITS")
        return out

    # Restrict profiles to WMO window
    tgprof    = tgprof[profu]
    zgprof    = zgprof[profu]
    pgprof    = pgprof[profu]
    thgprof   = thgprof[profu]
    surfgprof = surfgprof[profu]
    dgprof    = dgprof[profu]
    ssgprof   = ssgprof[profu]
    if have_pv:
        pvgprof   = pvgprof[profu]

    # ---- 2-point interpolation across 2 points
    def interp_two_point(x_target, x_pair, y_pair):
        """Simple 2-point linear interpolation."""
        x0, x1 = float(x_pair[0]), float(x_pair[1])
        y0, y1 = float(y_pair[0]), float(y_pair[1])
        if x1 == x0:  # degenerate, avoid division by zero
            return y0
        return y0 + (y1 - y0) * (x_target - x0) / (x1 - x0)

    # Find all levels where dT/dz > dttest (>-2 K/km)
    dtlt2 = np.where(dgprof > dttest)[0]
    ndtl2 = dtlt2.size

    dtlt2_first = dtlt2[0] if ndtl2 > 0 else -1   

    # if lowest good level already has dT/dz GT -2K/km -- define tropopause
    # if lowest good level is less than 18 km and there are values of dT/dz LT -2K/km
    # within around 2 km above it (currently within 2.5 km).

    if dtlt2_first == 0:
        dtgt2 = np.where(dgprof <= dttest)[0]
        ndtg2 = dtgt2.size
        if (ndtg2 > 0 and
            zgprof[dtgt2[0]] < 18.0 and
            (zgprof[dtgt2[0]] - zgprof[dtlt2_first]) <= 2.5):

            start = dtgt2[0]
            tmp = np.where(dgprof[start:] > dttest)[0]
            ndtl2 = tmp.size
            if ndtl2 > 0:
                dtlt2 = tmp + start

    # ---- Main WMO tropopause search ----
    if ndtl2 >= 2 and (dtlt2[0] > 0):
        diff = dtlt2[1:] - dtlt2[:-1]
        breaks = np.where(diff > 1)[0]
        nbreaks = breaks.size

        if nbreaks > 0:
            # Multiple contiguous segments where dT/dz > dttest
            for nb in range(nbreaks + 1):
                if numwmo <= 3:  # max 4 WMO tropopauses
                    if nb > 0:
                        index = dtlt2[breaks[nb - 1] + 1] - 1
                    else:
                        index = dtlt2[0] - 1

                    if nb < nbreaks:
                        ind2kt = dtlt2[breaks[nb]]
                    else:
                        ind2kt = dtlt2[ndtl2 - 1]

		            ##  ;;Actual WMO criteria -- if dT/dz GT -2K/km AND the average between that level and 
		            ##  ;;any higher level within 2 km remains GT -2K/km
                    if ((zgprof[ind2kt] - zgprof[index]) > 2.0 or
                        0.5 * (dgprof[index] + dgprof[ind2kt]) > dttest):

                        xint  = dgprof[index:index + 2]
                        yint  = zgprof[index:index + 2]
                        ysrf  = surfgprof[index:index + 2]
                        tint  = tgprof[index:index + 2]
                        pint  = pgprof[index:index + 2]
                        thint = thgprof[index:index + 2]
                        ssint = ssgprof[index:index + 2]

                        wmosurf[numwmo]  = interp_two_point(dttest, xint, ysrf)
                        wmotemp[numwmo]  = interp_two_point(dttest, xint, tint)
                        wmopres[numwmo]  = interp_two_point(dttest, xint, pint)
                        wmoalt[numwmo]   = interp_two_point(dttest, xint, yint)
                        wmotheta[numwmo] = interp_two_point(dttest, xint, thint)
                        wmoss[numwmo]    = interp_two_point(dttest, xint, ssint)

                        if have_pv:
                            qint = pvgprof[index:index + 2]
                            wmopv[numwmo] = interp_two_point(dttest, xint, qint)

                        numwmo += 1
        else:
            # Single contiguous block with dT/dz > dttest
            index = dtlt2[0] - 1
            xint  = dgprof[index:index + 2]
            yint  = zgprof[index:index + 2]
            ysrf  = surfgprof[index:index + 2]
            tint  = tgprof[index:index + 2]
            pint  = pgprof[index:index + 2]
            thint = thgprof[index:index + 2]
            ssint = ssgprof[index:index + 2]

            wmoalt[0]   = interp_two_point(dttest, xint, yint)
            wmosurf[0]  = interp_two_point(dttest, xint, ysrf)
            wmotemp[0]  = interp_two_point(dttest, xint, tint)
            wmopres[0]  = interp_two_point(dttest, xint, pint)
            wmotheta[0] = interp_two_point(dttest, xint, thint)
            wmoss[0]    = interp_two_point(dttest, xint, ssint)

            if have_pv:
                qint = pvgprof[index:index + 2]
                wmopv[0] = interp_two_point(dttest, xint, qint)

            numwmo = 1

        out = {'wmosurf':wmosurf, 'wmoalt':wmoalt, 'wmotemp':wmotemp, 
        'wmopres':wmopres, 'wmopv':wmopv, 'wmotheta':wmotheta, 'wmoss':wmoss, 
        'numwmo':numwmo, 'plow':plow, 'phigh':phigh, 'zlow':zlow, 'dttest':dttest, 'baddata':baddata}

    else:
        if verbose:
            print("****ERROR, FAILED INITIAL WMO TROPOPAUSE TEST****")
        numwmo = 0
        # Arrays already set to baddata

    return out







def interp_prfs(index, xprof, xtarget, yprofs):

    xprof  = np.asarray(xprof)
    yprofs = np.asarray(yprofs)

    # number of profiles is size of 2nd dimension
    numprofs = yprofs.shape[1]

    yvals = np.full(numprofs, np.nan, dtype=float)

    # two-point segment in x
    sl = slice(index, index + 2)
    xint = xprof[sl]

    # sanity check on length
    if xint.size != 2:
        print("---error: xprof does not have two points at given index")
        return yvals

    x0 = xint[0]
    x1 = xint[1]

    # check that xtarget is in range of xint, and two elements of xint have same sign when needed
    in_range_increasing = (xtarget >= x0) and (xtarget <= x1) 
    in_range_decreasing = (xtarget <= x0) and (xtarget >= x1) and (x0 * x1 > 0.0)

    if in_range_increasing or in_range_decreasing:
        # do linear interpolation for each profile
        dx = x1 - x0
        if dx == 0:
            print("%----eror: xint[0] and xint[1] are identical", xint)
            return yvals

        for yy in range(numprofs):
            yint = yprofs[sl, yy]  # two y-values
            y0, y1 = yint[0], yint[1]
            # manual linear interpolation
            yvals[yy] = y0 + (xtarget - x0) * (y1 - y0) / dx

    else:
        if x0 * x1 < 0.0:
            print("-----error: two elements of xint have opposite signs*", xint)
        else:
            print("-----error: xtarget not in range of xprof[index:index+1]", xtarget, xint)

    return yvals



def dyn_tropopauses(tprf, zprf, pprf, pvprf,
                   plow=850., phigh=10., zlow=0.0,
                   tropopv=2.0, thtest=380., 
                   verbose=False):

    tprf  = np.copy(np.asarray(tprf, float))
    pvprf = np.copy(np.asarray(pvprf, float))
    zprf  = np.copy(np.asarray(zprf, float))
    pprf  = np.copy(np.asarray(pprf, float))


    #the profiles need to be from the surface to space
    if zprf[0] > zprf[-1]:
        tprf  = np.flip(tprf)
        zprf  = np.flip(zprf)
        pprf  = np.flip(pprf)
        pvprf = np.flip(pvprf)




    numLevs = pprf.size

    surfPrf = np.arange(numLevs, dtype=float)

    baddata = np.nan
    # Initialize output arrays (max 4 dyn TPs, 3 folds)
    dynalt   = np.full(4, baddata, float)
    dyntemp  = np.full(4, baddata, float)
    dynpres  = np.full(4, baddata, float)
    dynsurf  = np.full(4, baddata, float)
    dynflag  = np.full(4, baddata, float)
    dynTheta = np.full(4, baddata, float)
    dyndTdz  = np.full(4, baddata, float)
    dynSS    = np.full(4, baddata, float)


    tpfalt   = np.full(3, baddata, float)
    tpftem   = np.full(3, baddata, float)
    tpfpre   = np.full(3, baddata, float)
    tpfsurf  = np.full(3, baddata, float)
    tpfTheta = np.full(3, baddata, float)
    tpfdTdz  = np.full(3, baddata, float)
    tpfSS    = np.full(3, baddata, float)

    numdyn = 0
    numtpf = 0



    mask = np.isfinite(tprf) & np.isfinite(zprf) & np.isfinite(pprf) & np.isfinite(pvprf)

    ngprf = len(mask)

    if ngprf == 0:
        print("DynTropopauses: no good input data")
    else:
        tprf    = tprf[mask]
        zprf    = zprf[mask]
        pprf    = pprf[mask]
        pvprf   = pvprf[mask]
        surfprf = surfPrf[mask]

        thprf = tprf * (1000.0 / pprf) ** (2.0 / 7.0)


        dtdzprf = np.gradient(tprf, zprf, edge_order=1)  # dT/dz in K/km

        ssprf = static_stability(dtdzprf/1000., tprf) # convert to K/m


        yprfs = np.empty((ngprf, 7), float)
        yprfs[:, 0] = zprf
        yprfs[:, 1] = tprf
        yprfs[:, 2] = pprf
        yprfs[:, 3] = surfprf
        yprfs[:, 4] = dtdzprf
        yprfs[:, 5] = ssprf
        yprfs[:, 6] = thprf


        # index of highest level where theta < thtest
        thltv = np.where(thprf < thtest)[0]
        if thltv.size > 1:
            difft = thltv[1:] - thltv[:-1]
            thind_idx = np.where(difft > 1)[0]
            thind = thltv[thind_idx[0]] if thind_idx.size > 0 else thltv[-1]
        elif thltv.size == 1:
            thind = thltv[0]
        else:
            thind = None

        pvtest = tropopv * 1e-6

        # points satisfying PV, pressure, and z constraints
        qltv = np.where((np.abs(pvprf) < pvtest) &
                        (pprf < plow) &
                        (zprf > zlow))[0]
        nqltv = qltv.size

        if nqltv == 0:
            if verbose:
                print("DynTropopauses: failed initial PV/pressure test")
        else:
            if nqltv > 1:
                diff = qltv[1:] - qltv[:-1]
                inddyn = np.where(diff > 1)[0]
                numdin = inddyn.size
                if numdin > 0:
                    indup = np.concatenate((qltv[inddyn], [qltv[-1]]))
                else:
                    indup = np.array([qltv[-1]])
            else:
                if verbose:
                    print("Only one level below tropopause")
                diff = np.array([], int)
                inddyn = np.array([], int)
                numdin = 0
                indup = np.array([qltv[-1]])

            nmult = min(numdin + 1, 4)
            nbelow = nmult - 1
            if nbelow > 0 and numdin > 0:
                inddn_full = indup[:numdin] + diff[inddyn] - 1
                inddn = inddn_full[:nbelow]
            else:
                inddn = np.array([], int)

            tmptype = np.zeros(4, float)
            nm = 0
            while nm <= nmult - 1:
                if thind is not None and thprf[indup[nm]] > thtest:
                    nmult = min(numdin + 1, nm + 1)
                    tmptype[nm] = 1.0
                elif (nbelow > nm and thind is not None and
                      thprf[inddn[nm]] > thtest):
                    nmult = min(numdin + 1, nm + 1)
                    nbelow = nmult - 1
                    tmptype[nm] = 0.0
                else:
                    tmptype[nm] = 0.0
                nm += 1

            # tropopauses
            numdyn = 0
            for nm in range(nmult):
                if tmptype[nm] > 0.01 and thind is not None and thind < ngprf - 1:
                    dynflag[nm] = 1.0
                    tpvals = interp_prfs(thind, thprf, thtest,yprfs[:, :6])
                    (dynalt[nm], dyntemp[nm], dynpres[nm], dynsurf[nm],
                    dyndTdz[nm], dynSS[nm]) = tpvals
                    dynTheta[nm] = thtest
                    if np.isfinite(dynalt[nm]):
                        numdyn += 1
                else:
                    dynflag[nm] = 0.0
                    tpvals = interp_prfs(indup[nm], np.abs(pvprf), pvtest,yprfs)
                    (dynalt[nm], dyntemp[nm], dynpres[nm], dynsurf[nm],
                    dyndTdz[nm], dynSS[nm], dynTheta[nm]) = tpvals

                    if dynTheta[nm] > thtest and thind is not None and thind < ngprf - 1:
                        dynflag[nm] = 1.0
                        tpvals = interp_prfs(thind, thprf, thtest,yprfs[:, :6])
                        (dynalt[nm], dyntemp[nm], dynpres[nm], dynsurf[nm], dyndTdz[nm], dynSS[nm]) = tpvals
                        dynTheta[nm] = thtest
                        if np.isfinite(dynalt[nm]):
                            numdyn += 1

            # folds (tropopause folds below)
            if nbelow > 0:
                for nb in range(nbelow):
                    tpvals = interp_prfs(inddn[nb], np.abs(pvprf), pvtest,yprfs)
                    (tpfalt[nb], tpftem[nb], tpfpre[nb], tpfsurf[nb],
                    tpfdTdz[nb], tpfSS[nb], tpfTheta[nb]) = tpvals
                    if tpfTheta[nb] > thtest:
                        tpfalt[nb] = tpftem[nb] = tpfpre[nb] = tpfsurf[nb] = \
                            tpfdTdz[nb] = tpfSS[nb] = tpfTheta[nb] = baddata
                        break
                    numtpf += 1

    tpf = dict(
    plow=plow, phigh=phigh, zlow=zlow,
    tropopv=tropopv, baddata=baddata,
    numtpf=numtpf, tpfsurf=tpfsurf, tpfalt=tpfalt,
    tpftemp=tpftem, tpfpres=tpfpre, tpfTheta=tpfTheta,
    tpfdTdz=tpfdTdz, tpfSS=tpfSS
    )

    trop = dict(
    plow=plow, phigh=phigh, zlow=zlow,
    tropopv=tropopv, baddata=baddata,
    numdyn=numdyn, dynsurf=dynsurf, dynalt=dynalt,
    dyntemp=dyntemp, dynpres=dynpres, dynflag=dynflag,
    dynTheta=dynTheta, dyndTdz=dyndTdz, dynSS=dynSS, folds =tpf
    )

    return trop


###This blended tropopause is the lower of the height of a potential vorticity based tropopause 
###and height of a thermally defined tropopause. 

def blendedtropopause(tprf, zprf, pprf, pvprf, tropopv = 4.5):

    wmo = wmo_tropopauses(tprf, zprf, pprf, pvprof = pvprf)
    dyn = dyn_tropopauses(tprf, zprf, pprf, pvprf, tropopv = tropopv)


    flg = 'wmo'

    if dyn['dynalt'][0] < wmo['wmoalt'][0]:
        flg = 'dyn'


    if flg == 'wmo':
        trop = dict(alt=wmo['wmoalt'][0], pre=wmo['wmopres'][0], tem=wmo['wmotemp'][0], pv=wmo['wmopv'][0])

    if flg == 'dyn':
        trop = dict(alt=dyn['dynalt'][0], pre=dyn['dynpres'][0], tem=dyn['dyntemp'][0], pv=tropopv*1e-6)


    return trop

# move chosen axis to front, flatten the rest
def reshape_and_flatten(arr, axis):
    dmy = np.moveaxis(arr, axis, 0)
    return dmy.reshape(dmy.shape[0], -1)


def computetropopauses(tem, alt, pre, pv, vertaxis, tropopv = 4.5):

    print('---Computing tropopauses')
    tem = np.copy(tem)
    alt = np.copy(alt)
    pre = np.copy(pre)
    pv  = np.copy(pv)

    dims = tem.shape
    dimsvert = dims[vertaxis]
    dims2flatten = dims[:vertaxis] + dims[vertaxis+1:]

    dimflat = np.prod(dims2flatten)

    tem = reshape_and_flatten(tem, vertaxis)
    alt = reshape_and_flatten(alt, vertaxis)
    pre = reshape_and_flatten(pre, vertaxis)
    pv  = reshape_and_flatten(pv, vertaxis)

    blnpre = np.full(dimflat, np.nan, dtype = float)
    dynpre = np.full(dimflat, np.nan, dtype = float)
    wmopre = np.full(dimflat, np.nan, dtype = float)
    blntem = np.full(dimflat, np.nan, dtype = float)

    for ip in tqdm(np.arange(dimflat)):
        
        tprf = tem[:,ip]
        zprf = alt[:,ip]
        pprf = pre[:,ip]
        pvprf = pv[:,ip]
        
        
        bb = blendedtropopause(tprf, zprf, pprf, pvprf, tropopv = tropopv)
        ww = wmo_tropopauses(tprf, zprf, pprf, pvprof = pvprf)
        dd = dyn_tropopauses(tprf, zprf, pprf, pvprf, tropopv = tropopv)

        blnpre[ip] = bb['pre']
        dynpre[ip] = dd['dynpres'][0]
        wmopre[ip] = ww['wmopres'][0]
        blntem[ip] = bb['tem']



    blnpre = np.reshape(blnpre, dims2flatten)
    dynpre = np.reshape(dynpre, dims2flatten)
    wmopre = np.reshape(wmopre, dims2flatten)
    blntem = np.reshape(blntem, dims2flatten)


    oo = dict(blnpre = blnpre, dynpre=dynpre, wmopre=wmopre, blntem=blntem)
    return oo







