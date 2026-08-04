import numpy as np
# from netCDF4 import Dataset
# import matplotlib.pyplot as plt
# import time
# import xarray as xr



def _initpsllm(mgrd, nlons, nlats, stlat, glats=None):
    a0 = 6.3662e6

    nGrd = mgrd * 64 + 1
    pole = int(nGrd / 2.0)

    r0 = float(mgrd) * 31.204359052

    nlatsu = nlats // 2 + 1

    if glats is None or len(glats) == 0:
        if (nlats % 2 != 0) or (abs(stlat - 90.0) > 1.0e-4):
            dlat = 2.0 * stlat / (nlats - 1)
        else:
            dlat = 180.0 / nlats
        lats = stlat - np.arange(nlatsu) * dlat
    else:
        lats = np.asarray(glats)[::-1][:nlatsu]


    lons = np.arange(nlons) * 360.0 / nlons

    rsc = r0 * np.tan(np.pi / 4.0 - np.deg2rad(lats) / 2.0)

    csLon = np.tile(np.cos(np.deg2rad(lons))[:, None], (1, nlatsu))
    snLon = np.tile(np.sin(np.deg2rad(lons))[:, None], (1, nlatsu))

    rsc2 = np.tile(rsc[None, :], (nlons, 1))
    x = pole + rsc2 * csLon
    y = pole + rsc2 * snLon

    ii, jj = np.meshgrid(np.arange(nGrd), np.arange(nGrd), indexing='ij')
    dist2 = (ii - pole) ** 2.0 + (jj - pole) ** 2.0
    dist = np.sqrt(dist2)

    r = (1.0 + dist2 / r0 ** 2.0) * r0 / 2.0 / a0
    phase = np.pi / 2.0 - 2.0 * np.arctan(dist / r0)
    f = np.sin(phase)

    rVal = {
        'mgrd': mgrd,
        'x': x,
        'y': y,
        'r': r,
        'f': f,
        'csLon': csLon,
        'snLon': snLon,
        'r0': r0,
    }

    return rVal


def _relvorm(v, u, vx, uy, r, r0, dud=None):
    a0 = 6.3662e6

    ngrd = vx.shape[0]
    rpole = float(ngrd // 2)


    ii, jj = np.meshgrid(np.arange(ngrd), np.arange(ngrd), indexing='ij')


    term = r * (vx - uy) - ((ii - rpole) * v - (jj - rpole) * u) / r0 / a0

    if dud is not None and np.isfinite(dud):
        vort = np.full(vx.shape, dud, dtype=float)

        good = (
            (np.abs(uy - dud) > 1) &
            (np.abs(vx - dud) > 1) &
            (np.abs(u - dud) > 1) &
            (np.abs(v - dud) > 1)
        )

        vort[good] = term[good]
        return vort

    return term


def _pstollm(inarray, x, y, dud=None):

    merids, lats = x.shape

    finite_baddata = dud is not None and np.isfinite(dud)

    fill = dud if finite_baddata else np.nan
    outarray = np.full((merids, lats), fill, dtype=float)


    for lon in range(merids):
        for lat in range(lats):
            xx = x[lon, lat]
            yy = y[lon, lat]

            m = int(xx)
            n = int(yy)
            xd = xx - m
            yd = yy - n

            # Skip if bad-data checking is enabled and any corner is bad
            if finite_baddata:
                block = inarray[m:m+2, n:n+2]
                if np.any(block == dud):
                    continue

            outarray[lon, lat] = (
                yd * (xd * inarray[m+1, n+1] + (1 - xd) * inarray[m, n+1])
                + (1 - yd) * (xd * inarray[m+1, n] + (1 - xd) * inarray[m, n])
            )

    if finite_baddata:
        outarray[~np.isfinite(outarray)] = dud

    return outarray


def _zderivm(z, dud=None):

    z = np.asarray(z, dtype=float)
    ngrd = z.shape[0]

    zz = z.copy()

    use_dud = dud is not None and np.isfinite(dud)
    if use_dud:
        zz[np.abs(z - dud) < 1] = np.nan

    # Interior with centered differences
    zx = (np.roll(zz, -1, axis=0) - np.roll(zz, 1, axis=0)) / 2.0
    zy = (np.roll(zz, -1, axis=1) - np.roll(zz, 1, axis=1)) / 2.0

    # Boundary derivatives for j = 0 and j = ngrd-1
    zx[:, 0] = (np.roll(zz[:, 0], -1) - np.roll(zz[:, 0], 1)) / 2.0
    zy[:, 0] = zz[:, 1] - zz[:, 0]

    zx[:, ngrd - 1] = (np.roll(zz[:, ngrd - 1], -1) - np.roll(zz[:, ngrd - 1], 1)) / 2.0
    zy[:, ngrd - 1] = zz[:, ngrd - 1] - zz[:, ngrd - 2]

    # Boundary derivatives for i = 0 and i = ngrd-1
    zy[0, :] = (np.roll(zz[0, :], -1) - np.roll(zz[0, :], 1)) / 2.0
    zx[0, :] = zz[1, :] - zz[0, :]

    zy[ngrd - 1, :] = (np.roll(zz[ngrd - 1, :], -1) - np.roll(zz[ngrd - 1, :], 1)) / 2.0
    zx[ngrd - 1, :] = zz[ngrd - 1, :] - zz[ngrd - 2, :]

    if use_dud:
        zx[~np.isfinite(zx)] = dud
        zy[~np.isfinite(zy)] = dud

    return zx, zy





_state = {}

def _lltopsvec(mgrd, arll, arlp, nhem, stlat, lam0, r0,
              test=False, dud=None, reset=False):
    """
    Convert lat/lon vector components to polar stereographic grid.
    
    Parameters
    ----------
    mgrd   : int, grid multiplier
    arll   : 2D array, lat/lon component 1 (e.g. u-wind)
    arlp   : 2D array, lat/lon component 2 (e.g. v-wind)
    nhem   : int, hemisphere (+1 = NH, -1 = SH)
    stlat  : float, standard latitude
    lam0   : float, reference longitude
    r0     : float, map scale factor
    test   : bool, save debug output
    dud    : float, bad/missing data value (None -> NaN)
    reset  : bool, clear cached state
    
    Returns
    -------
    arpsx, arpsy : 2D arrays on PS grid
    """
    global _state

    def is_valid(*vals):
        return np.all(np.isfinite(vals))

    ngrd = mgrd * 64 + 1
    npole = ngrd // 2  # center of PS grid

    if reset:
        _state.clear()

    if dud is None:
        dud = np.nan
    finite_baddata = np.isfinite(dud)

    pi = np.pi
    pi = 3.1415927
    deg2rad = pi / 180.0

    dims = arll.shape
    merids, lats = dims[0], dims[1]

    # Wrap in longitude (add one column)
    arllp = np.empty((merids + 1, lats), dtype=arll.dtype)
    arlpp = np.empty((merids + 1, lats), dtype=arlp.dtype)

    arllp[:merids, :] = arll
    arllp[merids, :] = arll[0, :]
    arlpp[:merids, :] = arlp
    arlpp[merids, :] = arlp[0, :]

    # print(arllp.shape)
    if finite_baddata:
        arllp = np.where(np.abs(arllp - dud) < 0, np.nan, arllp)
        arlpp = np.where(np.abs(arlpp - dud) < 0, np.nan, arlpp)
    # print(arllp.shape)


    # Checking if we need to recompute geometry
    needs_recompute = (
        _state.get('nhem')   != nhem   or
        _state.get('lats')   != lats   or
        _state.get('merids') != merids or
        _state.get('mgrd')   != mgrd   or
        'phi' not in _state
    )

    if needs_recompute:
        # print(f"First call to _lltopsvec for nhem={nhem}, lats={lats}, merids={merids}, mgrd={mgrd}")

        # Grid spacing
        if (lats % 2 != 0) or (abs(stlat - 90.0) > 1e-4):
            delp = 2 * abs(stlat) / (lats - 1)
        else:
            delp = 180.0 / lats
        dell = 360.0 / merids


        fpole = float(npole)
        ii, jj = np.meshgrid(np.arange(ngrd), np.arange(ngrd), indexing='ij')

        # Latitude index on PS grid
        dist = np.sqrt((ii - fpole)**2 + (jj - fpole)**2, dtype = np.float32)

        phi = pi / 2.0 - 2.0 * np.arctan(dist / r0)

        phi = (stlat * deg2rad - phi) / deg2rad / delp

        if nhem < 0:
            phi = float(lats - 1) - phi

        # Longitude angle
        theta = np.abs(np.arctan2(jj - fpole, ii - fpole)) / deg2rad
        lam = np.full((ngrd, ngrd), 180.0)
        lam[jj < npole] = 360.0 - theta[jj < npole]
        lam[jj > npole] = theta[jj > npole]
        lam[(ii >= npole) & (jj == npole)] = 0.0

        snlam = np.sin(lam * deg2rad)
        cslam = np.cos(lam * deg2rad)


        # Convert longitude to grid index
        lam = (lam - lam0) / dell + 1
        lam[lam >= merids] -= merids
        lam[lam < 0] += merids
        if nhem < 0:
            lam = float(merids) - lam



        _state.update(dict(
            phi=phi, lam=lam, snlam=snlam, cslam=cslam,
            dell=dell, delp=delp,
            nhem=nhem, lats=lats, merids=merids, mgrd=mgrd
        ))

    phi   = _state['phi']
    lam   = _state['lam']
    snlam = _state['snlam']
    cslam = _state['cslam']
    delp  = _state['delp']

    # Pole values (average over all meridians, excluding wrapped point)
    llpole = 0 if nhem > 0 else lats - 1
    arpole1 = np.nanmean(arllp[:merids, llpole], dtype = np.float32)
    arpole2 = np.nanmean(arlpp[:merids, llpole], dtype = np.float32)


    # Output arrays
    arpsx = np.full((ngrd, ngrd), dud if finite_baddata else np.nan)
    arpsy = np.full((ngrd, ngrd), dud if finite_baddata else np.nan)


    ilam = np.fix(lam).astype(np.int64)
    iphi = np.fix(phi).astype(np.int64)


    xd = lam - ilam
    yd = phi - iphi

    # We'll build pswrk1, pswrk2 as NaN arrays and fill per case
    pswrk1 = np.full((ngrd, ngrd), np.nan)
    pswrk2 = np.full((ngrd, ngrd), np.nan)

    # --- Case 1: pole point ---
    if np.isfinite(arpole1) and np.isfinite(arpole2):
        pswrk1[npole, npole] = arpole1
        pswrk2[npole, npole] = arpole2

    # Create masks for each case (excluding the pole point)
    not_pole = np.ones((ngrd, ngrd), dtype=bool)
    not_pole[npole, npole] = False

    off_grid  = not_pole & ((iphi >= lats) | (iphi < -1))
    near_south = not_pole & ~off_grid & (iphi == -1)
    near_north = not_pole & ~off_grid & (iphi == lats - 1)
    interior   = not_pole & ~off_grid & ~near_south & ~near_north



    if np.any(off_grid):
        raise ValueError(f"Index off grid at positions: {np.argwhere(off_grid)}")

    # Helper: safe array lookup (returns NaN if out of bounds)
    def v1(r, c):  # arllp[c, r] with bounds check via clipping — only valid where mask is True
        return arllp[np.clip(c, 0, merids), np.clip(r, 0, lats - 1)]
    def v2(r, c):
        return arlpp[np.clip(c, 0, merids), np.clip(r, 0, lats - 1)]

    # --- Case 2: near south pole (iphi == -1) ---
    m = near_south
    if np.any(m):
        ydp = 0.5 - yd[m]
        il, ip = ilam[m], iphi[m]
        x = xd[m]
        a = arllp[il + 1, ip + 1]
        b = arllp[il,     ip + 1]
        ok1 = np.isfinite(a) & np.isfinite(b) & np.isfinite(arpole1)
        tmp = np.where(ok1,
                       (1 - ydp) * (x * a + (1 - x) * b) + ydp * arpole1,
                       np.nan)
        pswrk1[m] = tmp

        a = arlpp[il + 1, ip + 1]
        b = arlpp[il,     ip + 1]
        ok2 = np.isfinite(a) & np.isfinite(b) & np.isfinite(arpole2)
        tmp = np.where(ok2,
                       (1 - ydp) * (x * a + (1 - x) * b) + ydp * arpole2,
                       np.nan)
        pswrk2[m] = tmp

    # --- Case 3: near north pole (iphi == lats-1) ---
    m = near_north
    if np.any(m):
        ydp = 0.5 - yd[m]
        il, ip = ilam[m], iphi[m]
        x = xd[m]
        a = arllp[il + 1, ip]
        b = arllp[il,     ip]
        ok1 = np.isfinite(a) & np.isfinite(b) & np.isfinite(arpole1)
        tmp = np.where(ok1,
                       ydp * (x * a + (1 - x) * b) + (1 - ydp) * arpole1,
                       np.nan)
        pswrk1[m] = tmp

        a = arlpp[il + 1, ip]
        b = arlpp[il,     ip]
        ok2 = np.isfinite(a) & np.isfinite(b) & np.isfinite(arpole2)
        tmp = np.where(ok2,
                       ydp * (x * a + (1 - x) * b) + (1 - ydp) * arpole2,
                       np.nan)
        pswrk2[m] = tmp

    # --- Case 4: interior ---
    m = interior
    if np.any(m):
        il, ip = ilam[m], iphi[m]
        x, y = xd[m], yd[m]
        a00 = arllp[il,     ip    ]
        a10 = arllp[il + 1, ip    ]
        a01 = arllp[il,     ip + 1]
        a11 = arllp[il + 1, ip + 1]
        ok1 = np.isfinite(a00) & np.isfinite(a10) & np.isfinite(a01) & np.isfinite(a11)
        tmp = np.where(ok1,
                       y * (x * a11 + (1-x) * a01) + (1-y) * (x * a10 + (1-x) * a00),
                       np.nan)
        pswrk1[m] = tmp

        b00 = arlpp[il,     ip    ]
        b10 = arlpp[il + 1, ip    ]
        b01 = arlpp[il,     ip + 1]
        b11 = arlpp[il + 1, ip + 1]
        ok2 = np.isfinite(b00) & np.isfinite(b10) & np.isfinite(b01) & np.isfinite(b11)
        tmp = np.where(ok2,
                       y * (x * b11 + (1-x) * b01) + (1-y) * (x * b10 + (1-x) * b00),
                       np.nan)
        pswrk2[m] = tmp


    # --- Rotate to PS x/y components ---
    both_valid = np.isfinite(pswrk1) & np.isfinite(pswrk2)
    arpsx = np.where(both_valid,
                     nhem * (-pswrk1 * snlam - pswrk2 * cslam), np.nan)
    arpsy = np.where(both_valid,
                     nhem * ( pswrk1 * cslam - pswrk2 * snlam), np.nan)

    # --- Pole point: replace with 4-point average ---
    # Determine incp (spacing to nearest valid neighbor)
    incp = 1  # default
    if stlat != 90.0:
        phitest = (float(lats) - phi[npole-1, npole]) if nhem == -1 else phi[npole-1, npole]
        if phitest < 0.0 and mgrd > 1:
            r0max = 2.0 * 31.204359052
            phi2_val = pi/2.0 - 2.0*np.arctan(1.0/r0max)
            phi2_val = (stlat * deg2rad - phi2_val) / deg2rad / delp
            r0max = 31.204359052
            phi1_val = pi/2.0 - 2.0*np.arctan(1.0/r0max)
            phi1_val = (stlat * deg2rad - phi1_val) / deg2rad / delp
            if phi2_val > 0.0:
                lowgrd = 2
            elif phi1_val > 0.0:
                lowgrd = 1
            else:
                print("****warning - lat/lon grid too coarse for ps grid****")
                lowgrd = mgrd
                incp = 1
            ninc = mgrd // lowgrd - 1
            incp = ninc + 1
        else:
            lowgrd = mgrd
            ninc = 0
            incp = 1
    else:
        lowgrd = mgrd
        ninc = 0
        incp = 1

    def pole_avg(arr, inc):
        pts = [arr[npole, npole + inc],
               arr[npole, npole - inc],
               arr[npole - inc, npole],
               arr[npole + inc, npole]]
        if all(np.isfinite(p) for p in pts):
            return np.mean(pts)
        return np.nan

    arpsx[npole, npole] = pole_avg(arpsx, incp)
    arpsy[npole, npole] = pole_avg(arpsy, incp)

    # --- Reinterpolate near-pole points if needed ---
    if stlat != 90.0 and lowgrd != mgrd:
        for i in range(-ninc, ninc + 1):
            for j in range(-ninc, ninc + 1):
                if i == 0 or j == 0:
                    continue
                ii_idx = npole + i
                jj_idx = npole + j
                indx = npole - ninc - 1 if i < 0 else npole
                indy = npole - ninc - 1 if j < 0 else npole
                dx = float(ii_idx - indx) / float(ninc + 1)
                dy = float(jj_idx - indy) / float(ninc + 1)
                for arr in (arpsx, arpsy):
                    a00 = arr[indx,        indy       ]
                    a10 = arr[indx + incp, indy       ]
                    a01 = arr[indx,        indy + incp]
                    a11 = arr[indx + incp, indy + incp]
                    if np.isfinite(a00) and np.isfinite(a10) \
                       and np.isfinite(a01) and np.isfinite(a11):
                        arr[ii_idx, jj_idx] = (
                            dy * (dx * a11 + (1-dx) * a01) +
                            (1-dy) * (dx * a10 + (1-dx) * a00)
                        )
                    else:
                        arr[ii_idx, jj_idx] = np.nan

    # Convert NaN back to dud if needed
    if finite_baddata:
        arpsx = np.where(~np.isfinite(arpsx), dud, arpsx)
        arpsy = np.where(~np.isfinite(arpsy), dud, arpsy)

    return arpsx, arpsy



def compute_pv(x, y, u, v, t, p, 
    relvor=None, gaussian=False, rvcalc="PS", 
    verbose=False, baddata=np.nan, test=False, mgrd=4): 
    """
    This routines computes pv from lat lon u v winds, temperature pressure 
    x is lon 1D [deg E]
    y is lat 1D [deg N]

    all the following variables are 3D arrays with size, lon, lat lev 
    lev needs to be sorted so that it goes from surface to space
    u east west direction wind [m s-1] 
    v north south wind  [m s-1] 
    t Temperature  [K]
    p pressure [hPa]

    returns 
    PV 3D array [lon, lat, lev]  'K m+2 kg-1 s-1'
    """
    
    # constants
    g = 9.81
    omega = 7.292115922e-5
    deg_to_m = 2.0 * np.pi * 6.378137e6 / 360.0

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)

    nx = x.size
    ny = y.size
    nz = u.shape[2]

    rvcalc = rvcalc.upper()

    # horizontal metric terms on lat/lon grid
    dx = deg_to_m * 0.5 * ((np.roll(x, -1) - np.roll(x, 1) + 360.0) % 360.0)
    dx3d = np.broadcast_to(dx[:, None, None], (nx, ny, nz)).copy()



    y3d = np.broadcast_to(y[None, :, None], (nx, ny, nz)).copy()
    dx3d *= np.sin(np.deg2rad(y3d + 90.0))
    # dx3d *= np.sin(np.deg2rad(y3d))


    dy = deg_to_m * 0.5 * (np.roll(y, -1) - np.roll(y, 1))
    dy3d = np.broadcast_to(dy[None, :, None], (nx, ny, nz)).copy()



    dx_safe = dx3d
    dy_safe = dy3d


    # relative vorticity
    if rvcalc == "LL":
        relvor = (
            (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx_safe)
            - (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy_safe)
        )

    else:

        swlat = y[-1]
        swlon = 0.0

        # Save original Gaussian latitude grid if needed
        if gaussian:
            rr = _initpsllm(4, nx, ny, swlat, glats=y)
            xpso = rr["x"]
            ypso = rr["y"]
            yo = y.copy()

            # interpolate u,v from Gaussian latitudes to an equally spaced latitude grid
            meandlats = np.mean(y[1:] - y[:-1])
            meanlats = y[0] + np.arange(ny) * meandlats

            u_interp = np.empty_like(u)
            v_interp = np.empty_like(v)

            for i in range(nx):
                for k in range(nz):
                    u_interp[i, :, k] = np.interp(meanlats, y, u[i, :, k])
                    v_interp[i, :, k] = np.interp(meanlats, y, v[i, :, k])

            u = u_interp
            v = v_interp
            y = meanlats

        relvor = np.zeros_like(u)

        ngrd = mgrd * 64 + 1
        latsu = ny // 2 + 1

        ni = np.arange(nx)
        si = ni[::-1]
        sj = np.arange(latsu)
        nj = (np.arange(latsu) + latsu - 1)[::-1]

        rr   = _initpsllm(4, nx, ny, swlat)
        xps  = rr["x"]
        yps  = rr["y"]
        r0ps  = rr["r0"]
        rps = rr["r"]


        if not gaussian:
            xpso = xps
            ypso = yps

        if verbose:
            print("r0:", r0ps)
            print("Going into hemispheres loop")

        for ihem in [1, -1]:
            indhem = (ihem + 1) // 2

            if verbose:
                hemi_name = "Northern" if ihem == 1 else "Southern"
                print(f"Working on {hemi_name} hemisphere")

            if indhem == 1:
                ii = ni
                jj = nj
            else:
                ii = si
                jj = sj

            dud = np.nan

            for k in range(nz):
                # reverse latitude so it goes north -> south
                uu = u[:, ::-1, k]
                vv = v[:, ::-1, k]





                # lat/lon vector -> polar stereographic vector
                ups1, vps1 = _lltopsvec(mgrd, uu, vv, ihem, swlat, swlon, r0ps, dud=dud)



                dudy1_x, dudy1 = _zderivm(ups1, dud=dud)
                dvdx1, dvdx1_y = _zderivm(vps1, dud=dud)




                # # PS relative vorticity
                vortps1 = _relvorm(vps1, ups1, dvdx1, dudy1, rps, r0ps, dud=dud)

                # back to lat/lon grid
                vortll1 = _pstollm(vortps1, xpso, ypso, dud=dud)


                for i in range(nx):
                    jst = 1 if ihem == 1 else 0
                    jed = latsu - 1 if ihem == 1 else latsu - 2

                    for j in range(jst, jed + 1):
                        relvor[ii[i], jj[j], k] = vortll1[i, j]
  

        if gaussian:
            y = yo

            # rebuild y3d on original Gaussian latitudes
            y3d = np.broadcast_to(y[None, :, None], (nx, ny, nz)).copy()


    # absolute vorticity
    absvor = relvor + 2.0 * omega * np.sin(np.deg2rad(y3d))

    # potential temperature
    theta = (1000.0 / p) ** (2.0 / 7.0) * t

    dthetadp = np.zeros((nx, ny, nz), dtype=float)
    dudp = np.zeros((nx, ny, nz), dtype=float)
    dvdp = np.zeros((nx, ny, nz), dtype=float)

    # interior levels
    for k in range(1, nz - 1):
        dp1 = p[:, :, k] - p[:, :, k - 1]
        dp2 = p[:, :, k + 1] - p[:, :, k]

        dthetadp[:, :, k] = (
            ((theta[:, :, k + 1] - theta[:, :, k]) * dp1**2
             - (theta[:, :, k - 1] - theta[:, :, k]) * dp2**2)
            / (dp1 * dp2 * (dp1 + dp2))
        )

        dudp[:, :, k] = (
            ((u[:, :, k + 1] - u[:, :, k]) * dp1**2
             - (u[:, :, k - 1] - u[:, :, k]) * dp2**2)
            / (dp1 * dp2 * (dp1 + dp2))
        )

        dvdp[:, :, k] = (
            ((v[:, :, k + 1] - v[:, :, k]) * dp1**2
             - (v[:, :, k - 1] - v[:, :, k]) * dp2**2)
            / (dp1 * dp2 * (dp1 + dp2))
        )

    # bottom level
    dp1 = p[:, :, 1] - p[:, :, 0]
    dp2 = p[:, :, 2] - p[:, :, 0]

    dthetadp[:, :, 0] = (
        ((theta[:, :, 1] - theta[:, :, 0]) * dp2**2
         - (theta[:, :, 2] - theta[:, :, 0]) * dp1**2)
        / (dp1 * dp2 * (dp2 - dp1))
    )

    dudp[:, :, 0] = (
        ((u[:, :, 1] - u[:, :, 0]) * dp2**2
         - (u[:, :, 2] - u[:, :, 0]) * dp1**2)
        / (dp1 * dp2 * (dp2 - dp1))
    )

    dvdp[:, :, 0] = (
        ((v[:, :, 1] - v[:, :, 0]) * dp2**2
         - (v[:, :, 2] - v[:, :, 0]) * dp1**2)
        / (dp1 * dp2 * (dp2 - dp1))
    )

    # top level
    dp1 = p[:, :, nz - 2] - p[:, :, nz - 1]
    dp2 = p[:, :, nz - 3] - p[:, :, nz - 1]

    dthetadp[:, :, nz - 1] = (
        ((theta[:, :, nz - 2] - theta[:, :, nz - 1]) * dp2**2
         - (theta[:, :, nz - 3] - theta[:, :, nz - 1]) * dp1**2)
        / (dp1 * dp2 * (dp2 - dp1))
    )

    dudp[:, :, nz - 1] = (
        ((u[:, :, nz - 2] - u[:, :, nz - 1]) * dp2**2
         - (u[:, :, nz - 3] - u[:, :, nz - 1]) * dp1**2)
        / (dp1 * dp2 * (dp2 - dp1))
    )

    dvdp[:, :, nz - 1] = (
        ((v[:, :, nz - 2] - v[:, :, nz - 1]) * dp2**2
         - (v[:, :, nz - 3] - v[:, :, nz - 1]) * dp1**2)
        / (dp1 * dp2 * (dp2 - dp1))
    )

    # horizontal theta gradients
    dthdx = (np.roll(theta, -1, axis=0) - np.roll(theta, 1, axis=0)) / (2.0 * dx3d)
    dthdy = (np.roll(theta, -1, axis=1) - np.roll(theta, 1, axis=1)) / (2.0 * dy3d)

    # PV
    pv = -g * (absvor * dthetadp - dvdp * dthdx + dudp * dthdy) * 1.0e-2

    # pole fix only for LL relvor branch
    if rvcalc == "LL":
        pv[:, 0, :] = np.nanmean(pv[:, 1, :], axis=0)[None, :]
        pv[:, ny - 1, :] = np.nanmean(pv[:, ny - 2, :], axis=0)[None, :]

    if rvcalc == "PS":
        pv[:, 0, :] = np.nanmean(pv[:, 1, :], axis=0)[None, :]

    
    # replace bad values
    pv = np.asarray(pv, dtype=float)
    bad = ~np.isfinite(pv)
    if np.any(bad):
        pv[bad] = baddata

    return pv