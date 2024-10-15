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

| field | description | perc_avail |
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


## Clean up data and finalize stage II features

