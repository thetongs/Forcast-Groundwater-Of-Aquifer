## Problem statement
# Predict the depth of groundwater of aquifer located in Petrignano

## Load libraries
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Load dataset
#
dataset = pd.read_csv('dataset/Aquifer_Petrignano.csv')
print("Total dataset lines {}".format(len(dataset)))
dataset.head()

## Drop unwanted columns and columns
# we need to remove first lines till 2009 because there is no data available
dataset = dataset.drop(['Depth_to_Groundwater_P24', 'Temperature_Petrignano'], axis = 1)
dataset = dataset[1024:]
print("Total dataset lines {}".format(len(dataset)))
dataset.head()

## Rename column names
names = ['Date', 'Rainfall', 'Depth_to_Groundwater','Temperature', 'Drainage_Volume', 'River_Hydrometry']
dataset.columns = names
dataset.head()

## Divide into independent and dependent variables
independent_var = ['Date', 'Rainfall', 'Temperature', 'Drainage_Volume', 'River_Hydrometry']
dependent_var = ['Depth_to_Groundwater']

target = dataset[dependent_var]
features = dataset[independent_var]
features.head()

## Check datatype
#
features.dtypes

## Convert Date into date type
#
from datetime import date, datetime
features.Date = pd.to_datetime(features.Date, format = '%d/%m/%Y')
features.dtypes

## Check missing values
#
features.isna().sum()

## Missing values
#
features.Drainage_Volume = features.Drainage_Volume.fillna(features.Drainage_Volume.mean())
# features.Drainage_Volume = features.Drainage_Volume.fillna(features.Drainage_Volume.mode()[0])
features.isna().sum()

## Visualize the features and target
# River hydrometry vs date
f, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 25))
plt.figure(figsize = (20, 8))

ax[0].plot(features.Date, features.Rainfall, color = 'r')
ax[0].set(xlabel = 'Date', ylabel = 'Rainfall')
ax[0].set(title = 'Date vs Rainfall')

ax[1].plot(features.Date, features.Temperature, color = 'g')
ax[1].set(xlabel = 'Date', ylabel = 'Temperature')
ax[1].set(title = 'Date vs Temperature')

ax[2].plot(features.Date, features.Drainage_Volume, color = 'b')
ax[2].set(xlabel = 'Date', ylabel = 'Drainage_Volume')
ax[2].set(title = 'Date vs Drainage_Volume')

ax[3].plot(features.Date, features.River_Hydrometry, color = 'y')
ax[3].set(xlabel = 'Date', ylabel = 'River_Hydrometry')
ax[3].set(title = 'Date vs River_Hydrometry')

plt.show()
plt.close()


ts = pd.concat([features.Date, target], axis = 1)
ts.head()

ts.Depth_to_Groundwater = ts.Depth_to_Groundwater.fillna(ts.Depth_to_Groundwater.mean())
ts.isna().sum()

ts.Depth_to_Groundwater = ts.Depth_to_Groundwater.abs()
ts

ts.dtypes

con = ts.Date
ts['Date'] = pd.to_datetime(ts.Date)
ts.set_index('Date', inplace = True)
ts

## Visualize
# Data vs Depth_to_Groundwater
plt.plot(ts, color = 'g')
plt.xlabel('Years')
plt.ylabel('Depth to groundwater')
plt.title('Years vs Deapth to groundwater')
plt.show()
plt.close()


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries.any()).rolling(window=12, center = False).mean()
    rolstd = pd.Series(timeseries.any()).rolling(window=12, center = False).std()
#Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(ts1)

ts_log = np.log(ts)
plt.plot(ts_log)
plt.show()
plt.close()

ts_log_diff = ts_log - ts_log.shift()
ts_lof_diff

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order = (2, 1, 0))
result_AR = model.fit(disp =-1)
plt.plot(ts_log_diff)
plt.plot(result_AR.fittedvalues , color = 'red')
plt.show()
plt.close()

## End
#

