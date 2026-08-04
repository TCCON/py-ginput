import sys, os, datetime
import cdsapi
import argparse
from pathlib import Path

###to be able to install see: 
###https://cds.climate.copernicus.eu/how-to-api
### this uses cfgrib>=0.9,<1.0


##example to run python getera5.py 2018 1 1
##or  python getera5.py 2018 1 1 -d <output directory path>

## the parameters are defined at https://codes.ecmwf.int/grib/param-db/
## the param is value.table    to get the tables you have to click on each parameter 

def downloadDay2d(year, month, day, time):
        if time not in range(0, 24):
                raise Exception("Time must be in {0 - 23}")

        c = cdsapi.Client()

        r= c.retrieve('reanalysis-era5-complete',{
                        'class': 'ea',
                        'date': '%02d-%02d-%02d' % (year, month, day),
                        'expver': '1',
                        'grid': '0.3/0.3',
                        'levelist': '1',
                        'levtype': 'ml',
                        'levelist': '1/to/137/by/1',
                        #'param': '129/152',
                        'param': '129.128/152.128',
                        'stream': 'oper',
                        'time': '%02d:00' % (hour),
                        'type': 'an'
        })

        return r

def downloadDay3d(year, month, day, time):
        if time not in range(0, 24):
                raise Exception("Time must be in {0 - 23}")

        c = cdsapi.Client()

        r = c.retrieve(
                'reanalysis-era5-complete',
                {
			'class': 'ea',
                        'date': '%02d-%02d-%02d' % (year, month, day),
                        'expver': '1',
			'levelist':'1/to/137/by/1',
                        # 'param': '130.128/131.128/132.128/133.128/138.128/203.128',
                        # 'param':'130.128/138.128/133.128/203.128/123.210',
                        'param':'130.128/138.128/133.128/203.128',
                        'stream': 'oper',
                        'time': '%02d:00' % (hour),
			'levtype': 'ml',
                        'type': 'an',
			'target': 'output',
			'grid': '0.3/0.3',
                })

        return r

if __name__ == '__main__':


    args = sys.argv


    parser = argparse.ArgumentParser(description="Download ERA5 data for a given date.")

    parser.add_argument("year", type=int, help="Year (e.g., 2005)")
    parser.add_argument("month", type=int, help="Month (1-12)")
    parser.add_argument("day", type=int, help="Day (1-31)")

    
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=str(Path.cwd() / "era5"), 
        help="Output directory (default: <current dir>/era5/)"
    )

    args = parser.parse_args()

    year = args.year
    month = args.month
    day = args.day
    directory = args.directory

    
    #only downloading evey 3 hour 
    #(the fields are actually available every hour but modmaker just uses every 3 hours for now)
    hours = [0,3,6,9,12,15,18,21]
    # hours = [0]
    print('hours to donwload:', hours)
    for hour in hours:
            doy = int(datetime.datetime(year, month, day).strftime('%j'))
            # directory = '/data/ecmwf/era5/%04d/%03d/' % (year, doy)
            # directory = '/users/lmillan/work/testings/era5/'
            #directory = '/home/lmillan/work/tccon/era5/'
            if not os.path.exists(directory):
                    os.makedirs(directory)

    # Download 2D data
            file_path = directory + 'era5_%04d%02d%02d_%02d00_2d.grib' % (year, month, day, hour)

            if os.path.isfile(file_path.split('.')[0] + '.grib'):
                    print(file_path + " is already downloaded.")
            else:
                    print("Submitting job for " + file_path)
                    request = downloadDay2d(year, month, day, hour)
                    request.download(file_path)

                    print("Downloaded file: %s" % str(file_path))

    #Download 3D data
            file_path = directory + 'era5_%04d%02d%02d_%02d00_3d.grib' % (year, month, day, hour)

            if os.path.isfile(file_path.split('.')[0] + '.grib'):
                    print(file_path + " is already downloaded.")
            else:
                    print("Submitting job for " + file_path)
                    request = downloadDay3d(year, month, day, hour)
                    request.download(file_path)

                    print("Downloaded file: %s" % str(file_path))



