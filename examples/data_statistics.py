import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import folium
from folium.plugins import HeatMap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator

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


# Sparsity Information
#---------------------
neighborhoods = df['neighborhood_id'].unique()
start = pd.to_datetime('2012-01-01')
end = pd.to_datetime('2024-07-31')
for neighborhood in neighborhoods:
  print('='*5+f' {neighborhood} '+'='*5)
  # Daily Sparsity
  freq = 'H'
  full_date_range = pd.date_range(start=start, end=end, freq=freq)
  missing_dates = full_date_range.difference(df_hourly['y-m-d-h'][df_hourly['neighborhood_id']==neighborhood])
  sparsity_percentage = ((len(missing_dates) / len(full_date_range))) * 100
  print(f"Hourly sparsity: {sparsity_percentage:.2f}%")

  # Daily Sparsity
  freq = 'D'
  full_date_range = pd.date_range(start=start, end=end, freq=freq)
  missing_dates = full_date_range.difference(df_daily['y-m-d'][df_daily['neighborhood_id']==neighborhood])
  sparsity_percentage = ((len(missing_dates) / len(full_date_range))) * 100
  print(f"Daily sparsity: {sparsity_percentage:.2f}%")

  # Monthly
  freq = 'MS'
  full_date_range = pd.date_range(start=start, end=end, freq=freq)
  missing_dates = full_date_range.difference(df_monthly['y-m'][df_monthly['neighborhood_id']==neighborhood])
  sparsity_percentage = ((len(missing_dates) / len(full_date_range))) * 100
  print(f"Monthly sparsity: {sparsity_percentage:.2f}%")
