# prc_challenge
A model created for the PRC data challenge


required packages

```
pandas
numpy
dask
dask[dataframe]
fastparquet
dask[distributed]
bokeh>=3.1.0
openap
openap-top
pandarallel
fastmeteo
metpy
airportsdata
tabulate
future
requests
h20 (https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html#install-in-python)
```


## Basic ETL

1. Download project files

```
mc cp --recursive dc24/competition-data/ /Volumes/SMB/mark/flight_competition/competition_files_update_Oct11/ 
```

2. Repartition parquet files for faster flight-specific access

```
import dask.dataframe as dd
import glob
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=5, threads_per_worker=1, memory_limit='10GB')
client = Client(cluster)

df = dd.read_parquet("/mnt/SMB_share/mark/flight_competition/competition_files_update_Oct11/", engine='pyarrow')
df["first_five"] = df["flight_id"].astype(str).str[:5]
df.to_parquet(
    "/mnt/SMB_share/mark/flight_competition/repartitioned_from_fcompute/",
    engine='pyarrow',
    partition_on=["first_five", "flight_id"],  
    write_index=False,                
    compression='snappy',             
)
client.close()
```

## Use OpenAP to calculate idealized flight paths for each flight in competition and submission sets

See openap.ipynb

## Create features from flight path data

See get_trajectory_characteristics.ipynb

## Clean up data and finalize stage I features

See clean_up_phase_one.ipynb

| field | description | percent available |
| --- | --- | --- | 
| flight_id | (str) unique identifier | 100% |
| month | (int) month of flight | 100% |
| day_of_week | (int) day of week of flight | 100% |
| hour_in_local | (int) hour of flight in local time zone. This uses the first time zone from country_timezones package| 100% |
| adep | (str) departure airport code | 100% |
| ades | (str) destination airport code | 100% |
| aircraft_type | (str) aircraft type code | 100% |
| replacer | (str) alternative aircraft type to be used for weight regression when original code is not available in openap. This is usually the aircraft type in the challenge data with the closest average MTOW. | 100% |
| airline | (str) unique airline code | 100% |
| flight_duration_sec | (int) duration of flight in seconds | 100% |
| great_circle_km | (int) great circle distance in km, calculated using osmnx | 100% |
| mtow_fill | (int) MTOW from openap if available, if not from FAA data, rounded, kg | 100% |
| oew_fill | (int) OEW from openap if available, if not from FAA data, rounded, kg | 100% |
| total_fuel_fill | (int) estimated fuel weight in kg from openap for adep, ades and aircraft type or replacer. If airport not available, uses linear regression value from great circle distance rounded. | 100% |
| tow | (float) TOW provided by challenge dataset, kg rounded | 100% |
| dataset | (str) "challenge" or "submission" dataset | 100% |
| first_cruise_alt | (float) altitude in km when aircraft first reached cruise classification for four intervals consecutively, rounded | 34% |
| time_to_cruise | (float) seconds from takeoff to first cruise in first_cruise_alt rounded | 34% |
| alt_per_s | (float) first_cruise_alt divided by time_to_cruise, rounded to one decimal point | 34% |
| est_load_lf_adjusted | (float) mtow_fill minus oew_fill minus total_fuel_fill to get the estimated possible max passenger load, multiplied by the average monthly load factor in Europe for takeoff month  | 100% |
| est_tow | (float) est_load_lf_adjusted plus oew_fill plus total_fuel_fill for an estimated total TOW | 100% |

## Train phase one model

See train_stage_one.ipynb



## Clean up data and finalize stage II features

