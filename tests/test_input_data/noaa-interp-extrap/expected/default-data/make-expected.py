from ginput.priors import tccon_priors
import pandas as pd
from pathlib import Path

mydir = Path(__file__).parent
last_date = pd.Timestamp(2028,1,1)
records = {
    'co2': tccon_priors.CO2TropicsRecord(last_date=last_date),
    'n2o': tccon_priors.N2OTropicsRecord(last_date=last_date),
    'ch4': tccon_priors.CH4TropicsRecord(last_date=last_date),
}

for gas, rec in records.items():
    seas_nc_file = mydir / f'{gas}_seasonal.nc'
    rec.conc_df_to_nc(seas_nc_file, trend=False)
    trend_nc_file = mydir / f'{gas}_trend.nc'
    rec.conc_df_to_nc(trend_nc_file, trend=True)
