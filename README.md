# prc_challenge
A model created for the PRC data challenge


required packages

```
dask
dask[dataframe]
fastparquet
dask[distributed]
bokeh>=3.1.0
```


## ETL

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




