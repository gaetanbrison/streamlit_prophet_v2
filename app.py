## New Project 


import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric


st.title("Automated time series forecast")


#### Step 1 Import Data


df = st.file_uploader('Importing any time series dataset',type='csv')
st.info(
    f""" Upload a csv.here is a sample to start:[test.csv](https://raw.githubusercontent.com/gaetanbrison/streamlit_prophet_v2/master/example_data/test.csv)"""
)


if df is not None:
    data=pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce')
    st.write(data)

### Step 2 Select Forecast Horizon 

periods_input = st.number_input('How many periods would you like to forecast into the future?',min_value=1,max_value=365)


### Step 3: Visualize Forecast Data

if df is not None:
    m = Prophet()
    m.fit(data)
    future = m.make_future_dataframe(periods=periods_input)
    forecast = m.predict(future)
    fcst = forecast[['ds','yhat','yhat_lower','yhat_upper']]
    max_date = data['ds'].max()
    fcst_filtered = fcst[fcst['ds'] > max_date]
    st.write(fcst_filtered)


    fig1 = m.plot(forecast)
    st.write(fig1)


    fig2 = m.plot_components(forecast)
    st.write(fig2)

