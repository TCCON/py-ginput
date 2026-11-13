# -*- coding: utf-8 -*-
"""

The function "slantify" is the main function in this code, it converts a vertical grid at a given location and time into a slant grid along the sun ray.

Running this python code will run a test case (see the bottom of the code)

"""
import numpy as np
from numpy import cos,sin,tan,arctan,arccos,arcsin,arctan2,deg2rad,rad2deg
from datetime import datetime, timedelta
from matplotlib import pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import ephem
import pytz

def ssp(date):
    """
    Get the latitude and longitude of the sub-solar point

    Inputs:
        - date : UTC datetime object
    Outputs:
        - lat : sub-solar point latitude (radians)
        - lon : sub-solar point longitude (radians)
    """

    greenwitch = ephem.Observer()
    greenwitch.lon = "0"
    greenwitch.lat = "0"
    greenwitch.date = date
    greenwitch.pressure = 0

    sun = ephem.Sun(greenwitch)

    sun.compute(greenwitch.date)

    lon = sun.ra-greenwitch.sidereal_time()

    if lon < -180:
        lon = lon + 360
    if lon > 180:
        lon = lon - 360

    lat = sun.dec

    return lat, lon

def sun_earth_distance(date):
    """
    Input:
        - UTC datetime object
    Output:
        - Sun-Earth distance (meters)
    """
    # TODO: Workaround for skyfield not importing on linux in Python 3.15 as of 12 Nov 2026
    # Move back to a top-level import later
    from skyfield.api import load, utc

    planets = load('de421.bsp')

    earth,sun = planets['earth'],planets['sun']

    ts = load.timescale()

    t = ts.utc(date.replace(tzinfo=utc))

    astrometric = earth.at(t).observe(sun)
    ra, dec, distance = astrometric.radec()

    return distance.m

def r_geoid(lat,lon,re,rp):
    """
    Radius of geoid at lat,lon (meters)

    Inputs:
        - lat : geodetic latitude (radians)
        - lon : longitude (radians)
        - re : equatorial radius of Earth (meters)
        - rp : polar radius of Earth (meters)

    Outputs:
        - Radius of geoid at lat,lon (meters)
    """

    return np.sqrt( (cos(lat)**2*re**4+sin(lat)**2*rp**4) / ((cos(lat)*re)**2+(sin(lat)*rp)**2) )

def rv(lat,re,n):
    """
    Meridional radius of curvature at lat

    Inputs:
        - lat : geodetic latitude (radians)
        - re : equatorial radius of Earth (meters)
        - n : oblateness of Earth (meters)

    Outputs:
        - Meridional radius of curvature at lat (meters)
    """

    return re/np.sqrt(cos(lat)**2+(n*sin(lat))**2)

def geoid_position(lat,lon,re,n):
    """
    Cartesian position vector from geoid center to surface at lat,lon

    Inputs:
        - lat : geodetic latitude (radians)
        - lon : longitude (radians)
        - re : equatorial radius of Earth (meters)
        - n : oblateness of Earth (meters)

    Outputs:
        - Cartesian position vector from geoid center to surface at lat,lon
    """

    return rv(lat,re,n)*np.array([cos(lat)*cos(lon),cos(lat)*sin(lon),sin(lat)*n**2])

def vertical_unit_vector(lat,lon):
    """
    Unit vector along the vertical at lat,lon

    Inputs:
        - lat : geodetic latitude (radians)
        - lon : longitude (radians)
    Outputs:
        - Unit vector along the vertical at lat,lon
    """

    return np.array([cos(lat)*cos(lon),cos(lat)*sin(lon),sin(lat)])

def lat_lon_alt_at_position(position,re,rp,n):
    """
    Inputs:
        - position : cartesian position vector
        - re : equatorial radius of Earth (meters)
        - rp : polar radius of Earth (meters)
        - n : oblateness of Earth (meters)
    Outputs:
        - lat : geodetic latitude at position (degrees)
        - lon : longitude at position (degrees)
        - alt : vertical distance from geoid surface at position lat/lon (meters)
    """

    x,y,z = position

    lat = arctan2(z,n**2*np.sqrt(x**2+y**2))

    lon = arctan2(y,x)

    rg = r_geoid(lat,lon,re,rp)

    Pg = geoid_position(lat,lon,re,n)

    alt = distance_between(position,Pg) # vertical distance from geoid surface (meters)

    if np.linalg.norm(position)<rg:
        alt = -alt

    return rad2deg(lat),rad2deg(lon),alt

def distance_between(position1,position2):
    """
    Inputs:
        - position1 : 3d position vector
        - position2 : 3d position vector
    Outputs:
        - d : distance between the two positions
    """

    x1,y1,z1 = position1
    x2,y2,z2 = position2

    d = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

    return d

def sun_angles(date,lat,lon,alt,pres=0,temp=0):
    """
    Inputs:
        - date: datetime object
        - lat: geodetic latitude (radians)
        - lon: longitude (radians)
        - alt: altitude (meters)
        - (optional) pres: surface pressure (mbar)
        - (optional) temp: surface temperature (Celcius)
    Outputs:
        - corrected_sza: solar zenith angle (radians)
        - azim: solar azimuth angle (radians)
    """

    # setup observer at date,lat,lon,alt
    obs = ephem.Observer()
    obs.date = date
    obs.lon = lon
    obs.lat = lat
    obs.elevation = alt
    # if surface pressure and temperature are provided for date,lat,lon then atmospheric refraction will also be taken into account when computing the sun position
    obs.pressure = pres	# when set to 0 it will not be used
    obs.temp = temp

    # setup sun relative to the observer
    sun = ephem.Sun()
    sun.compute(obs)
    # this includes some corrections (e.g. parallax); see: https://rhodesmill.org/pyephem/radec
    corrected_sza= 0.5*np.pi-sun.alt	# solar zenith angle (radians)
    azim = sun.az	# azimuth angle (radians)

    return corrected_sza,azim

def slantify(date,lat,lon,alt,vertical_distances,pres=0,temp=0,plots=False):
    """
    Inputs:
        - lat : geodetic latitude (degrees)
        - lon : geodetic longitude (degrees)
        - alt : surface altitude at lat,lon (meters)
        - date : datetime object
        - vertical_distances : array of vertical levels above lat (meters)
        - pres: surface pressure (mbar); if set to 0, atmospheric refraction won't be included in angle calculations
        - temp: surface temperature (Celcius), only used when pres is not 0 for atmospheric refraction
        - plots: if True, plots will be displayed

    Outputs:
        - data: dictionary containing:
            'site_lat'	input lat 							(degrees)
            'site_lon'	input lon 							(degrees)
            'vertical'	input vertical_distances			(km)
            'slant'		slant_distances						(km)
            'lat'		geodetic latitude of slant points	(degrees)
            'lon'		loongitude of slant points			(degrees)
            'alt'		altitude of slant points			(km)
            'sza'		solar zenith angle					(degrees)
            'azim'		azimuth angle						(degrees)
    """

    # if the date in naive, make it aware as UTC
    # TODO: Workaround for skyfield not importing on linux in Python 3.15 as of 12 Nov 2026
    # Move back to a top-level import later
    from skyfield.api import utc
    if date.tzinfo is None:
        date = pytz.utc.localize(date)

    if lon>180:
        lon = lon-360

    lat = deg2rad(lat)
    lon = deg2rad(lon)

    re = 6378137 # equatorial radius of Earth (meters)
    rp = 6356752.3142 # polar radius of Earth (meters)
    n = rp/re # oblateness of Earth

    # radius of geoid at lat (meters)
    rg = r_geoid(lat,lon,re,rp)

    corrected_sza, azim = sun_angles(date,lat,lon,alt,pres,temp) # get solar zenith and azimuth angles

    ssp_lat,ssp_lon = ssp(date)		# latitude and longitude of sub-solar point

    vs = vertical_unit_vector(ssp_lat,ssp_lon)	# vertical unit vector at sub-solar point
    v = vertical_unit_vector(lat,lon)			# vertical unit vector at lat,lon

    d = sun_earth_distance(date) # meters

    B = rg/d
    uncorrected_sza = arccos(np.dot(v,vs))	# radians
    vsp = (vs-B*v)*sin(corrected_sza)/sin(uncorrected_sza) # vector towards the sun from observer

    Pg = geoid_position(lat,lon,re,n) # site position on geoid

    Po = Pg + alt*v 	# position vector up to site altitude

    t_tp = -(vsp[0]*Po[0]+vsp[1]*Po[1]+vsp[2]*Po[2]/n**2) / (vsp[0]**2+vsp[1]**2+(vsp[2]/n)**2)	# distance from site to tangent point along sun ray (meters)

    P_tp = Po + t_tp*vsp	# position of tangent point

    tp_lat,tp_lon,tp_alt = lat_lon_alt_at_position(P_tp,re,rp,n) # degrees, degrees, meters

    fixed_slant_distances = np.arange(0,5000001,1000) # fixed 1 km slant spacing up to 5000 km
    fixed_slant_positions = [Po+elem*vsp for elem in fixed_slant_distances]
    fixed_slant_coords = [lat_lon_alt_at_position(position,re,rp,n) for position in fixed_slant_positions]

    P_slant = np.array([elem[2] for elem in fixed_slant_coords]) # vertical distance from geoid surface for each fixed slant point
    P_vertical = vertical_distances

    slant_distances = np.interp(P_vertical,P_slant,fixed_slant_distances) # slant distances along sun ray corresponding to the vertical distances
    slant_positions = [Po+elem*vsp for elem in slant_distances] # position vectors corresponding to the slant distances along the sun ray
    slant_coords = [lat_lon_alt_at_position(position,re,rp,n) for position in slant_positions]

    data = {}
    data['site_lat'] = rad2deg(lat)										# degrees
    data['site_lon'] = rad2deg(lon)										# degrees
    data['vertical'] = vertical_distances/1000.0 						# km
    data['slant'] = slant_distances/1000.0								# km
    data['lat'] = np.array([elem[0] for elem in slant_coords])			# degrees
    data['lon'] = np.array([elem[1] for elem in slant_coords])			# degrees
    data['alt'] = np.array([elem[2]/1000.0 for elem in slant_coords])	# km
    data['sza'] = rad2deg(corrected_sza)								# degrees
    data['azim'] = rad2deg(azim)										# degrees

    if plots:
        title = datetime.strftime(date,'%Y %b %d at %H:%M')+'; lat={:.3f}; lon={:.3f}; sza={:.3f}'.format(rad2deg(lat),rad2deg(lon),rad2deg(corrected_sza))
        show_distances(re,rp,n,Pg,Po,P_tp,t_tp,tp_alt,v,vsp,vertical_distances,title=title)
        #show_positions(slant_positions)
        show_sunray(data,title=title)

    return data

### Some plotting functions

def show_positions(slant_positions):
    """
    Plots a collection of 3d vectors

    slant_positions: list of 3d vectors
    """
    fig = pl.figure()
    ax = fig.add_subplot(111,projection='3d')
    for position in slant_positions:
        ax.quiver(0,0,0,*position)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    pl.show()

def show_sunray(data,title=''):
    """
    3d plot showing the vertical levels and the slant levels along the sun ray

    data: this is the output of the "slantify" function
    """

    fig = pl.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(data['lat'],data['lon'],data['alt'],label='Along Sun ray')
    ax.scatter([data['site_lat'] for i in data['vertical']],[data['site_lon'] for i in data['vertical']],data['vertical'],label='Along vertical')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Altitude (km)')
    pl.legend()
    pl.title(title)
    pl.show()

def show_sunrays_over_time(date_list,lat,lon,alt,h):
    """
    date_list: list of UTC datetime objects

    plot points along Sun ray for each date in the list
    """

    fig = pl.figure()
    ax = fig.add_subplot(111,projection='3d')

    first = True
    for i,date in enumerate(date_list):

        data = slantify(date,lat,lon,alt,h)

        if first:
            ax.scatter([data['site_lat'] for i in data['vertical']],[data['site_lon'] for i in data['vertical']],data['vertical'],label='Along vertical',color='black')
            first = False

        if data['sza']>90:
            continue

        ax.scatter(data['lat'],data['lon'],data['alt'],label=datetime.strftime(date+timedelta(hours=data['site_lon']/15.0),'%b %d %H:%M')+'; sza={:.1f}'.format(data['sza']))

    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Altitude (km)')
    pl.legend()
    pl.title('From '+datetime.strftime(date_list[0]+timedelta(hours=data['site_lon']/15.0),'%Y-%m-%d %H:%M')+' to '+datetime.strftime(date_list[-1]+timedelta(hours=data['site_lon']/15.0),'%Y-%m-%d %H:%M'))
    pl.show()


def show_distances(re,rp,n,Pg,Po,P_tp,t_tp,tp_alt,v,vsp,h,title=''):
    """
    First plot: distance from Earth center vs distance along the vertical and the sun ray, also shows the tangent point distance along the sun ray

    Second plot: same but the distances along the sun ray were interpolated to that the distances from Earth center match the vertical ones
    """

    t = t = np.arange(-10000000,10000001,1000) # slant distances
    h = h # vertical distance

    fixed_slant_positions = [Po+elem*vsp for elem in t]
    fixed_slant_coords = [lat_lon_alt_at_position(position,re,rp,n) for position in fixed_slant_positions]

    Pt = np.array([elem[2] for elem in fixed_slant_coords]) # vertical distance from geoid surface for each slant point
    Ph = h

    IDs = np.where(Pt<(np.max(Ph)+2000))
    min_ID = IDs[0][0]
    max_ID = IDs[0][-1]

    Pt = Pt[min_ID:max_ID]
    t = t[min_ID:max_ID]

    # put everything in Km for the plots
    t = t / 1000.0
    Pt = Pt / 1000.0
    h = h / 1000.0
    Ph = Ph / 1000.0
    t_tp = t_tp / 1000.0
    P_tp = P_tp / 1000.0
    tp_alt = tp_alt / 1000.0

    plot([(t,Pt,'Along sun ray','blue'),(h,Ph,'Along vertical','green'),([t_tp],[tp_alt],'Tangent point','red')],xlab='Distance (km)',ylab='Vertical distance from geoid surface (km)',title=title)

    t_interp = np.interp(Ph,Pt[t>=0],t[t>=0])
    slant_positions = [Po+elem*vsp for elem in 1000.0*t_interp]
    slant_coords = [lat_lon_alt_at_position(position,re,rp,n) for position in slant_positions]

    Pt_interp = np.array([elem[2] for elem in slant_coords]) / 1000.0 # vertical distance from geoid surface for each slant point

    plot([(t_interp,Pt_interp,'Along sun ray','blue'),(h,Ph,'Along vertical','green')],xlab='Distance (km)',ylab='Vertical distance from geoid surface (km)',title=title)

def plot(arg,line=False,xlab='x',ylab='y',title=''):
    """
    convenience function to plot several lines

    arg: list of (x,y,label,color) tuples
    """
    pl.clf()
    for elem in arg:
        x,y,lab,c = elem
        if line:
            pl.plot(x,y,label=lab,color=c)
        else:
            pl.scatter(x,y,label=lab,color=c)
    pl.grid()
    pl.legend()
    pl.xlabel(xlab)
    pl.ylabel(ylab)
    pl.title(title)
    pl.show()

if __name__=="__main__": # this only executes when the program is run directly (and not when imported from another code)

    # Set a test case for Lamont
    alt = 320 # meters
    lat = 36.604 # degrees
    lon = 262.514 # degrees
    date = datetime(2012,12,10,20)

    # vertical grid from 0 to 100 km with 1 km
    h = np.arange(0,100001,1000) # (meters)

    r = slantify(date,lat,lon,alt,h,plots=True)

    # If you have surface pressure and temperature data, the calculation will include atmospheric refraction:
    #p_surf = 1050 # millibar
    #t_surf = 15 # celcius
    #r = slantify(date,lat,lon,alt,h,pres=p_surf,temp=t_surf,plots=True)
