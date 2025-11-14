from __future__ import print_function # allows the use of Python 3.x print function in python 2.x code so that print('a','b') prints 'a b' and not ('a','b')

"""
Setup your .netrc and .urs_cookies files by following the steps at:  https://disc.gsfc.nasa.gov/data-access

Here are the relevant instructions for Mac/Linux (from the link above)
##################################################################################
Create a .netrc file in your home directory.
    cd ~ or cd $HOME
    touch .netrc
    echo "machine urs.earthdata.nasa.gov login <uid> password <password>" >> .netrc (where <uid> is your user name and <password> is your Earthdata Login password without the brackets)
    chmod 0600 .netrc (so only you can access it)
Create a cookie file. This file will be used to persist sessions across calls to wget or curl.
    cd ~ or cd $HOME
    touch .urs_cookies.
Note: you may need to re-create .urs_cookies in case you have already executed wget without valid authentication.
##################################################################################

After you have set up the two files above run this code with:

python get_MERRA2.py collection=c dates=d1,d2 lon=lon1,lon2 lat=lat1,lat2 var=var1,var2,var3 path=p

Mandatory:
    collection=c
    c is the MERRA2 collection name e.g. M2I3NPASM for 3d assimilated meteorological fields on 42 levels
    (see https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf to see all collection names)

    dates=d1,d2
    d1,d2 : first and last dates in YYYYMMDD format

    path=p
    p: path to the directory where the files will be downloaded

Optional:
    lon=lon1,lon2
    lon1,lon2 : min and max longitude in [-180,180] range (if not specified, defaults to -180,180)

    lat=lat1,lat2
    lat1,lat2: min and max latitude in [-90,90] range (if not specified, defaults to -90,90)

    var=var1,var2,var3
    var1,var2, etc.: variables to read (need the variables names like they are in the dataset e.g. EPV for potential vorticity), latitude,longitude,time, and pressure are always read, you do not need to specify them
    (if var= is not specified all variables will be used)

e.g.
To get global files with all variables between 2016-01-15 and 2016-01-16 for the Assimilated Meteorological Fields on 42 levels:
python get_MERRA2.py collection=M2I3NPASM dates=20160115,20160116

The same on 72 levels (the data collection changes):
python get_MERRA2.py collection=M2I3NVASM dates=20160115,20160116

To get global files for potential vorticity, temperature, and relative humidity:
python get_MERRA2.py collection=M2I3NPASM dates=20160115,20160116 var=EPV,T,RH

To get all variables in a specific lat,lon box:
python get_MERRA2.py collection=M2I3NPASM dates=20160115,20160116 lon=-98,-95 lat=34,38
"""

import os
import sys
from datetime import datetime,timedelta
from subprocess import Popen,PIPE,CalledProcessError


def execute(cmd,cwd=os.getcwd()):
    '''
    function to execute a unix command and print the output as it is produced
    '''
    popen = Popen(cmd, stdout=PIPE, universal_newlines=True,cwd=cwd)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise CalledProcessError(return_code, cmd)


def collection_info(collection):
    """
    Use a collection shortname to get its code and server number
    """

    code_dict = {'NP':'3d','NV':'3d','NX':'2d','T1':'tavg1','T3':'tavg3','I1':'inst1','I3':'inst3'}

    code = '_'.join([code_dict[i] for i in [collection[2:4],collection[4:6]]]+[collection[6:].lower(),collection[4]+collection[5].lower()])

    server = 5
    if 'NX' in collection:
        server = 4

    return code,server


def URLlist_MERRA2(collection,date_range,timestep=timedelta(days=1),lat_range=[-90,90],lon_range=[-180,180],variables=[],outpath=''):
    """
    MERRA-2 data has one global file every 1 days (from 00:00 to 21:00 UTC each day)
    collection: shortdname of data collection
    date_range: [start,end] datetime objects
    timestep: use the model time resolution to get all files, or a multiple of it to get less files
    lat_range: latitude range of subset (-90 to 90)
    lon_range: longitude range of subset (-180 to 180)
    variables: list of exact variables names (pressure,latitude,longitude, and time will always be read)
    outpath: full path to the file in which the list of URLs will be written
    """

    collection_code,server = collection_info(collection)

    start,end = date_range

    min_lat, max_lat = lat_range
    min_lon, max_lon = lon_range

    if variables == []:
        fmt = "http://goldsmr{}.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2F{}.5.12.4%2F{}%2F{:0>2}%2FMERRA2_{}.{}.{}.nc4&FORMAT=bmM0Yy8&BBOX={}%2C{}%2C{}%2C{}&LABEL=MERRA2_{}.{}.{}.SUB.nc&SHORTNAME={}&SERVICE=SUBSET_MERRA2&VERSION=1.02&DATASET_VERSION=5.12.4\n"
    else:
        fmt = "http://goldsmr{}.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2F{}.5.12.4%2F{}%2F{:0>2}%2FMERRA2_{}.{}.{}.nc4&FORMAT=bmM0Yy8&BBOX={}%2C{}%2C{}%2C{}&LABEL=MERRA2_{}.{}.{}.SUB.nc&SHORTNAME={}&SERVICE=SUBSET_MERRA2&VERSION=1.02&DATASET_VERSION=5.12.4&VARIABLES="+'%2C'.join(variables)+'\n'

    if outpath=='': # if no specified full path to make the file, just write a file in the current directory
        outpath = 'getMERRA2.dat'

    print('Writting URL list in:',outpath)

    curdate = start
    with open(outpath,'w') as f:
        while curdate<end:
            if curdate<datetime(1992,1,1):
                data_stream = '100'
            elif curdate>=datetime(1992,1,1) and curdate<datetime(2001,1,1):
                data_stream = '200'
            elif curdate>=datetime(2001,1,1) and curdate<datetime(2011,1,1):
                data_stream = '300'
            else:
                data_stream = '400'

            YYYYMMDD = datetime.strftime(curdate,'%Y%m%d')
            f.write(fmt.format(server,collection,curdate.year,curdate.month,data_stream,collection_code,YYYYMMDD,min_lat,min_lon,max_lat,max_lon,data_stream,collection_code,YYYYMMDD,collection))
            curdate += timestep

    return outpath


if __name__=="__main__":
    argu = sys.argv[1:]

    start = None
    end = None
    lat_range = [-90,90]
    lon_range = [-180,180]
    variables = []
    path = os.getcwd()

    for arg in argu:
        if 'dates=' in arg:
            var = arg.split('=')[1]
            date_range = [datetime.strptime(elem,'%Y%m%d') for elem in var.split(',')]
        if 'lon=' in arg:
            var = arg.split('=')[1]
            lon_range = [int(elem) for elem in var.split(',')]
        if 'lat=' in arg:
            var = arg.split('=')[1]
            lat_range = [int(elem) for elem in var.split(',')]
        if 'var=' in arg:
            var = arg.split('=')[1]
            variables = [elem for elem in var.split(',')]
        if 'collection=' in arg:
            collection = arg.split('=')[1]
        if 'path=' in arg:
            path = arg.split('=')[1]

    if not os.path.exists(path):
        print('Creating',path)
        os.makedirs(path)

    URLlist_MERRA2(collection,date_range=date_range,lat_range=lat_range,lon_range=lon_range,variables=variables,outpath=os.path.join(path,'getMERRA2.dat'))

    for line in execute('wget --load-cookies /home/sroche/.urs_cookies --save-cookies /home/sroche/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i getMERRA2.dat'.split(),cwd=path):
        print(line, end="")