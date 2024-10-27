# Mindful Donkey PRC Model

## General model description

This model was created for the [PRC Challenge](https://ansperformance.eu/study/data-challenge/) by Team Mindful Donkey. This work is copyrighted by Mark Fahey (2024) and free to use under the terms of the GNU GPLv3 license ([see gpl-3.0.txt](https://github.com/mtfahey/prc_challenge/blob/main/gpl-3.0.txt)). 

The model has a three stage structure. The first stage is an supervised learning ensemble model trained on the data available for each flight _excluding_ the trajectory data. This is essentially a baseline model that estimates the TOW based on the expected fuel amount and cargo capacity for each flight, as well as some macro data about the general shape of the climb trajectory. The second stage is a collection of models trained on the minute-by-minute cleaned ADS-B trajectory data for the climb section of each flight, with a different ensemble model for each aircraft type. The third stage integrates the first and second stages and also includes some summary data the accuracy of the stage two models to provide a final TOW estimate for each flight.  

||||||
| --- | --- | --- | --- | --- |
| STAGE I  -- Baseline TOW estimates from general flight information and macro flight characteristics | &rarr;  |  |
| STAGE II -- Models for each aircraft type to estimate TOW at each point in flight| &rarr; | STAGE III -- Model to integrate estimates  | &rarr; | Final TOW estimates
Stage II model accuracies | &rarr; |
||||||

### Model accuracy

The final TOW submission estimates for this model have a RMSE of 2683, compared to a test data RMSE of 2398.

STAGE I alone gives us a submission RSME of 2719. Using the median of STAGE II (or STAGE I if trajectories were not available) gives an RSME of 6140. Combining both models as part of STAGE III improves the predictions to 2683. 


## Model creation

Python packages required for replicating this work can be found in [requirements.txt](https://github.com/mtfahey/prc_challenge/blob/main/requirements.txt). All processing work and training of this model was done locally via CPU (Intel Xeon E5-2430). 

### Basic ETL

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

### OpenAP features for to STAGE I training

See [OpenAP notebook](https://github.com/mtfahey/prc_challenge/blob/main/notebooks/openap.ipynb). Here we am using Junzi Sun's [openap](https://github.com/junzis/openap) and [openap trajectory optimizer](https://github.com/junzis/openap-top) to: 
1. Add OEW and MTOW features for each aircraft type. [FAA data](https://github.com/mtfahey/prc_challenge/blob/main/data/aircraft_data_faa.csv) were used to fill in any missing values.
2. Add great circle distance features for each adep/ades combination. If not available in openap, Mike Borsetti's [airportsdata](https://github.com/mborsetti/airportsdata) package was used to find lat/lngs to calculate distances. Some airports were found and added manually.
3. Add an estimated fuel weight for the idealized trajectory between each adep/ades/aircraft type combination. If an aircraft type was not available, the aircraft type with the next closest average TOW in the training data was used as a replacement. If airports were not available, a linear regression of fuel weight on calculated linear distance was used as a replacement value. 

### Create general climb trajectory features and pre-process STAGE II features

See [get_trajectory_characteristics notebook](https://github.com/mtfahey/prc_challenge/blob/main/notebooks/get_trajectory_characteristics.ipynb). Here we use Xavier Olive's [traffic library](https://traffic-viz.github.io/) to process the flight trajectory data. Data were filtered and resampled at 30 seconds. Flight phases were added and fuel flow was calculated. The maximum fuel flow from each flight was calculated. 

The start of the cruise phase of each flight was found and the altitude and time to cruise were recorded for use as training features. 

Formulas from Yoshiki Kato's [weather_paramters](https://github.com/Yoshiki443/weather_parameters) package were used to calculate cross wind and tail wind from the provided v and u compoents.

Filtered climb phase 30-second trajectory data were saved for use in Stage II.  

### Clean up data and finalize STAGE I features

See [clean_up_phase_one notebook](https://github.com/mtfahey/prc_challenge/blob/main/notebooks/clean_up_stage_one.ipynb). This notebook restructures the data and adds some additional features, including the day of the week, the hour in local time, and an adjusted cargo load weight value based on the average monthly [aircraft load factors in Europe](https://github.com/mtfahey/prc_challenge/blob/main/data/passenger_load_factors.csv) extracted from monthly IATA publications to adjust for daily, weekly and monthly variation. I have also included some code here for replacing adep/ades values in the submission data that did not appear in the training data, but this does not appear to have improved the output and was not included in the final model.

The final STAGE I training features and descriptions are listed below:

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
| first_cruise_alt | (float) altitude in km when aircraft first reached cruise classification for four intervals consecutively, rounded | 84% |
| time_to_cruise | (float) seconds from takeoff to first cruise in first_cruise_alt rounded | 84% |
| alt_per_s | (float) first_cruise_alt divided by time_to_cruise, rounded to one decimal point | 84% |
| est_load_lf_adjusted | (float) mtow_fill minus oew_fill minus total_fuel_fill to get the estimated possible max passenger load, multiplied by the average monthly load factor in Europe for takeoff month  | 100% |
| est_tow | (float) est_load_lf_adjusted plus oew_fill plus total_fuel_fill for an estimated total TOW | 100% |

### Train STAGE I model

See [train_stage_one](https://github.com/mtfahey/prc_challenge/blob/main/notebooks/train_stage_one.ipynb) notebook. We used [H2O AutoML](https://pages.github.com/](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)) to test 30 possible models and assemble 16 into an ensemble model. The model models were tested using 5-fold cross validation and a GLM metalearner algorithm.  Ultimately, 8/10 GM models, 6/10 XGBoost models, 1/2 DRF models and 1/7 DeepLearning models were incorporated into the ensemble. One GLM model was tested but not included. 

The training RMSE for this model was 2036, the cross-validation RMSE was 2753 and the test RMSE was 2749. The final STAGE I model used for submission [can be found here](https://drive.google.com/drive/folders/1epyKt1HCyRLLWGmdF4-Q0Zjyqk_vF64P?usp=drive_link) (too large for github). 

### Clean up data and finalize STAGE II features

The [clean_up_stage_two](https://github.com/mtfahey/prc_challenge/blob/main/notebooks/clean_up_stage_two.ipynb) notebook makes csv training sets from parquet files for each aircraft type. Flight ids were divided into 10 fold groups to prevent cross-training within each flight. The features used to train a separate model for each aircraft type are listed below:

| field | description | percent available |
| --- | --- | --- | 
| sec_since_takeoff | (int) seconds since takeoff, calculated from each trajectory timestamp | 100% |
| altitude | (float) altitude in ft, provided in original data | 100% |
| groundspeed | (int) groundspeed in kt, provided in original data| 100% |
| vertical_rate | (int) vertical rate in ft/min, provided in original data | 100% |
| temperature | (float) temperature in K, provided in original data | 100% |
| specific_humidity | (float) specific humidity in g/kg, provided in original data | 100% |
| tail_wind | (float) calculated tail wind in m/s  | 100% |
| cross_wind | (float) calculated cross wind in m/s  | 100% |
| fold_group | (int) fold group to avoid training across flight ids | 100% |
| tow |(float) TOW provided by challenge dataset, kg rounded| 100% |

### Train STAGE II models

See [train_stage_two](https://github.com/mtfahey/prc_challenge/blob/main/notebooks/train_stage_two.ipynb) notebook. A separate model was trained for each of 27 aircraft types. Training was aimed at maximizing RMSE across fold groups without training within a flight id. Forty models were tested using H2OAutoML for each aircraft type, with the best model (usually an ensemble) retained. Final models [can be found here](https://drive.google.com/drive/folders/1CDqsbL7leA_y6fLWr3qTBH7L7kKm8tGx?usp=drive_link) (too large for github). A test group 10 percent of training data was used to calculate an RMSE and error as a percentage of the average training TOW to be used as a feature in STAGE III.


### Predict STAGE I, STAGE II, clean up data and finalize STAGE III features

See [predict_stage_two](https://github.com/mtfahey/prc_challenge/blob/main/notebooks/predict_stage_two.ipynb) notebook. Stage II predictions were binned by median value at 100 second intervals and saved as features for STAGE III. STAGE II errors and STAGE I predictions were also incldued as features for STAGE III. 

| field | description | percent available train |
| --- | --- | --- | 
| stage_one |(float) predicted TOW from STAGE I model| 100% |
| stage_two_100 |(float) median predicted TOW from seconds 0-100 of STAGE II model| 100% |
| stage_two_200 |(float) median predicted TOW from seconds 100-200 of STAGE II model| 94% |
| stage_two_300 |(float) median predicted TOW from seconds 200-300 of STAGE II model| 94% |
| stage_two_400 |(float) median predicted TOW from seconds 300-400 of STAGE II model| 94% |
| stage_two_500 |(float) median predicted TOW from seconds 400-500 of STAGE II model| 93% |
| stage_two_600 |(float) median predicted TOW from seconds 500-600 of STAGE II model| 92% |
| stage_two_700 |(float) median predicted TOW from seconds 600-700 of STAGE II model| 90% |
| stage_two_800 |(float) median predicted TOW from seconds 700-800 of STAGE II model| 86% |
| stage_two_900 |(float) median predicted TOW from seconds 800-900 of STAGE II model| 84% |
| stage_two_1000 |(float) median predicted TOW from seconds 900-1000 of STAGE II model| 78% |
| aircraft_type |(str) aircraft type provided in challenge dataset | 100% |
| percent_error |(float) RMSE test dataset error as a percentage of the median TOW for aircraft type | 100% |
| tow |(float) TOW provided by challenge dataset, kg rounded| 100% |

### Train STAGE III model

See [train_stage_three](https://github.com/mtfahey/prc_challenge/blob/main/notebooks/train_stage_three.ipynb) notebook. STAGE III model was trained via H2OAutoML over 50 RMSE-optimizing models from results of STAGE I and STAGE II as described above. Final model [can be found here](https://drive.google.com/drive/folders/1r7X6Si00P0CiFhd-S9tiw7JPt38lJ3qO?usp=drive_link) (too large for github). This file also contains prediction code for the submission file. 

