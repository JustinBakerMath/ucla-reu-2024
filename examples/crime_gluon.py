import argparse
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator, SimpleFeedForwardEstimator, WaveNetEstimator
from gluonts.transform import Identity

sys.path.append('./transforms')
from acfit import ACFITransform

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams["font.size"] = 30
plt.rcParams["font.family"] = 'serif'
plt.rcParams['mathtext.default'] = 'default'
# plt.rcParams["font.weight"] = 'bold'
plt.rcParams["xtick.color"] = 'black'
plt.rcParams["ytick.color"] = 'black'
plt.rcParams["axes.edgecolor"] = 'black'
plt.rcParams["axes.linewidth"] = 1


# Arguments
#==============================================================================
parser = argparse.ArgumentParser(description="")
parser.add_argument('--nbhd', type=str, help="Neighborhood", default="Five Points", choices=
['Gateway Green Valley Ranch','Hampden', 'Mar Lee', 'Jefferson Park',
'Auraria', 'Hampden South', 'Westwood', 'Capitol Hill', 'Fort Logan',
'Union Station', 'Civic Center', 'Bear Valley', 'Highland', 'Cole', 'Overland',
'Lincoln Park', 'Montbello', 'Goldsmith', 'Five Points', 'CBD',
'Northeast Park Hill', 'Athmar Park', 'Platt Park', 'Whittier',
'City Park West', 'Berkeley', 'Harvey Park South', 'West Colfax',
'South Park Hill', 'City Park', 'Speer', 'Sun Valley', 'Baker Valverde',
'Cherry Creek', 'Clayton', 'East Colfax', 'Globeville', 'West Highland',
'University Hills', 'Central Park', 'College View South Platte', 'Sunnyside',
'DIA', 'Lowry Field', 'Cory Merrill', 'Southmoor Park', 'Harvey Park',
'Country Club', 'North Park Hill', 'University Park', 'Belcaro',
'Virginia Village', 'Cheesman Park', 'Sloan Lake', 'Marston', 'Hale'
'Washington Park West', 'Rosedale', 'North Capitol Hill', 'Kennedy',
'Villa Park', 'Windsor', 'Regis', 'Ruby Hill', 'University', 'Skyland',
'Congress Park', 'Washington Virginia Vale', 'Barnum', 'Hilltop',
'Elyria Swansea', 'Washington Park', 'Wellshire', 'Barnum West', 'Montclair',
'Chaffee Park', 'Indian Creek'])
parser.add_argument('--freq', type=str, help="Frequency", default="d", choices=['h', 'd', 'M'])
parser.add_argument('--method', type=str, help="Method", default="deepar", choices=['feed_forward', 'deepar', 'wavenet'])
parser.add_argument('--epochs', type=int, help="Number of epochs", default=10)
parser.add_argument('--batch_size', type=int, help="Number of epochs", default=16)
parser.add_argument('--transform', type=str, help="Transform to use", default='none', choices=['none', 'acfit'])
args = parser.parse_args()

# Data Processing
#==============================================================================

# Clean Data
#-----------
df = pd.read_csv('./data/crime_2012_2024.csv', encoding='latin-1', low_memory=False)
LON = [-104.5, -105.2]
LAT = [39.6, 39.95]
df_cln1 = df[(df['geo_lon'] >= LON[1]) & (df['geo_lon'] <= LON[0]) &
                 (df['geo_lat'] >= LAT[0]) & (df['geo_lat'] <= LAT[1])]

# Temporal Grouping
#------------------
df_tmprl = df_cln1.copy()
df_tmprl['first_occurrence_date'] = pd.to_datetime(df_tmprl['first_occurrence_date'])

df_tmprl['hour'] = df_tmprl['first_occurrence_date'].dt.hour
df_tmprl['day'] = df_tmprl['first_occurrence_date'].dt.day
df_tmprl['year'] = df_tmprl['first_occurrence_date'].dt.year

df_hourly = df_tmprl.copy()
df_daily = df_tmprl.copy()
df_monthly = df_tmprl.copy()

df_hourly['y-m-d-h'] = df_hourly['first_occurrence_date'].dt.strftime('%Y-%m-%d %H:00:00')  # Combines day and hour
df_hourly['y-m-d-h'] = pd.to_datetime(df_hourly['y-m-d-h'])
df_hourly = df_hourly.groupby(['y-m-d-h', 'neighborhood_id']).size().reset_index(name='count')

df_daily['y-m-d'] = df_daily['first_occurrence_date'].dt.strftime('%Y-%m-%d')  # Combines day and hour
df_daily['y-m-d'] = pd.to_datetime(df_daily['y-m-d'])
df_daily = df_daily.groupby(['y-m-d', 'neighborhood_id']).size().reset_index(name='count')

df_monthly['y-m'] = df_monthly['first_occurrence_date'].dt.strftime('%Y-%m')  # Combines day and hour
df_monthly['y-m'] = pd.to_datetime(df_monthly['y-m'])
df_monthly = df_monthly.groupby(['y-m', 'neighborhood_id']).size().reset_index(name='count')

# Fill Missing Data
start = pd.to_datetime('2012-01-01')
end = pd.to_datetime('2024-07-31')
neighborhood = args.nbhd

freq = 'h'
full_date_range = pd.date_range(start=start, end=end, freq=freq)
df_hourly_fill = df_hourly[df_hourly['neighborhood_id']==neighborhood].drop(columns=['neighborhood_id']).set_index('y-m-d-h').reindex(full_date_range, fill_value=0)

freq = 'd'
full_date_range = pd.date_range(start=start, end=end, freq=freq)
df_daily_fill = df_daily[df_daily['neighborhood_id']==neighborhood].drop(columns=['neighborhood_id']).set_index('y-m-d').reindex(full_date_range, fill_value=0)

freq = 'M'
full_date_range = pd.date_range(start=start, end=end, freq='MS')
df_monthly_fill = df_monthly[df_monthly['neighborhood_id']==neighborhood].drop(columns=['neighborhood_id']).set_index('y-m').reindex(full_date_range, fill_value=0)


# Forecasting
#==============================================================================
freq = args.freq
dpm = [29, 31, 30, 31, 30, 31, 31]
# Local Train
if freq == 'h':
  df_data = df_hourly_fill#[df_hourly['y-m-d-h']<pd.to_datetime('2023-12-31')]
  start = df_data.index[0]
  offset = -24*7 #offset by 2024 data
  prediction_length = 24 #predict for one day
  windows = 7
elif freq == 'd':
  df_data = df_daily_fill#[df_daily['y-m-d']<pd.to_datetime('2023-12-31')]
  start = df_data.index[0]
  offset = -sum(dpm) #offset by 2024 data
  prediction_length = 21 #predict for three weeks
  windows = 10
elif freq == 'M':
  df_data = df_monthly_fill#[df_monthly['y-m']<pd.to_datetime('2023-12-31')]
  start = df_data.index[0]
  offset = -6 #offset by 2024 data
  prediction_length = 6 #predict for one year
  windows = 1
else:
  raise ValueError('Invalid frequency')

# Apply transformation
#==============================================================================

if args.transform == 'acfit':
  transform = ACFITransform(nlags=prediction_length, freq=args.freq, name=neighborhood)
else:
  transform = Identity()
data = ListDataset(
      [
          {
              "start": start,  # Start of the time series
              "target": df_data['count'].values  # Time series data
          }
      ],
  freq=freq
)

transf_data = transform.transform(data)
inv_data = transform.invert(transf_data)

plt.figure()
plt.plot(df_data.index, df_data['count'], label="Actual Data", color='k')
plt.plot(df_data.index, inv_data[0]['target'], label="Inverted Data", color='cyan', alpha=0.6, dashes=[2, 6])
plt.savefig(f'./out/{neighborhood}_{args.transform}_{freq}_transform.png')

data = transform.transform(data)

# Train the model
#==============================================================================
train_ds, test_gen = split(data, offset=offset) #offset by 2024 data
#model = DeepAREstimator(
    #prediction_length=prediction_length, freq=freq, trainer_kwargs={"max_epochs": args.epochs}
#)

if (args.method == "feed_forward"):
	estimator = SimpleFeedForwardEstimator(freq=freq,
										   prediction_length=prediction_length,
                                        trainer_kwargs={"max_epochs": args.epochs})
elif(args.method == "deepar"):
	estimator = DeepAREstimator(freq=freq,
								prediction_length=prediction_length,
                             trainer_kwargs={"max_epochs": args.epochs})
elif (args.method == "wavenet"):
	estimator = WaveNetEstimator(freq=freq,
								 prediction_length=prediction_length,
                              trainer_kwargs={"max_epochs": args.epochs})

model = estimator.train(train_ds)



# Predict the model
#==============================================================================
test_data = test_gen.generate_instances(prediction_length=prediction_length, windows=windows)
forecasts = list(model.predict(test_data.input))

# Visualize the forecast
#==============================================================================
def scale_xlim(freq):
    if freq == 'h':
      plt.xlim(pd.to_datetime('2024-07-24'),pd.to_datetime('2024-07-31'))
    elif freq == 'd':
      plt.xlim(pd.to_datetime('2024-01-01'),pd.to_datetime('2024-07-31'))
    elif freq == 'M':
      plt.xlim(pd.to_datetime('2022-01-01'),pd.to_datetime('2024-07-31'))
    else:
      raise ValueError('Invalid frequency')

plt.figure(figsize=(28, 9))
if freq == 'h':
    plt.plot(df_data.index[prediction_length:], data[0][FieldName.TARGET], label="Actual Data")
elif freq == 'd':
    plt.plot(df_data.index[prediction_length:], data[0][FieldName.TARGET], label="Actual Data")
elif freq == 'M':
    plt.plot(df_data.index[prediction_length:], data[0][FieldName.TARGET], label="Actual Data")
else:
  raise ValueError('Invalid frequency')

for i,forecast in enumerate(forecasts):
  forecast.plot(color='b')

scale_xlim(freq)
# Set plot labels and legend
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.savefig(f'./out/{neighborhood}_{freq}_{args.method}_{args.transform}_forecast.png')


# Forecast Inverted
#==============================================================================
test_data = test_gen.generate_instances(prediction_length=prediction_length, windows=1)
forecasts = list(model.predict(test_data.input))

final_forecasts = []
for f in forecasts:
  final_forecasts.append(f.median)

data = np.concatenate((data[0][FieldName.TARGET], np.hstack(final_forecasts)))
data = [{'target': data}]
inv_data = transform.invert(data)

inv_data[0]['target'] = np.round(inv_data[0]['target'])

plt.figure()
plt.plot(df_data.index, df_data['count'], label="Actual Data", color='k')
plt.plot(df_data.index[-prediction_length:], inv_data[0]['target'][-prediction_length:], label="Inverted Data", color='cyan', alpha=1)
scale_xlim(freq)
plt.savefig(f'./out/{neighborhood}_{freq}_{args.method}_{args.transform}_forecast_inverted.png')
