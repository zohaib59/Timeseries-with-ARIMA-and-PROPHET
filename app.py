#Arima
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
import os
import matplotlib.pyplot as plt

#set the new path
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv('awl.csv')# put the path and filename
data.head()

data["Price"] = np.log10(data["Price"])

data = data.drop("Date", axis=1)
data.head()

data.plot(figsize=(15,8))#plot the data, a large one

len(data)


#divide into train and validation set
train = data.iloc[:593]
len(train)
valid = data.iloc[593:]
len(valid)

model = auto_arima(train, trace=True,start_p=0, start_q=0,
start_P=0, start_Q=0,max_p=30, max_q=30, max_P=30, max_Q=30, seasonal=True,
stepwise=False, suppress_warnings=True, D=1, max_D=30,
error_action='ignore',approximation = True)


model.fit(train)
model.summary()


#predicting values and evaluating model
y_pred = model.predict(n_periods=len(valid))

y_pred = pd.DataFrame(y_pred)

import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(valid, y_pred)#mape

train.plot()
y_pred.plot()


rock = model.predict(n_periods=30)
rock = pd.DataFrame(rock)

rock = np.power(10,rock)#reverse of log10
rock

rock.to_csv('bhulbhal.csv',index=False)











#Prophet
from prophet import Prophet
import pandas as pd
import holidays
import os

# Set path and load data
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")
data = pd.read_csv("awl.csv")

# Convert and rename columns for Prophet
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
data = data.rename(columns={"Date": "ds", "Price": "y"})

# Add Indian holidays
years = pd.DatetimeIndex(data["ds"]).year.unique()
ind_holidays = holidays.India(years=years)
holiday_df = pd.DataFrame({"ds": list(ind_holidays.keys()), "holiday": "india_national"})

# Fit Prophet model
model = Prophet(holidays=holiday_df)
model.fit(data)

# Forecast 30 days ahead
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Save last 30 days of forecast
forecast[["ds", "yhat"]].tail(30).to_csv("forecast_prophet.csv", index=False)
