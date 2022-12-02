## New Project 


import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64


st.title("Automated time series forecast")


#### Step 1 Import Data

st.markdown("### Step 1 Import Data")

df = st.file_uploader('Importing any time series dataset',type='csv')
st.info(
    f""" Upload a csv.here is a sample to start:[test.csv](https://raw.githubusercontent.com/gaetanbrison/streamlit_prophet_v2/master/example_data/test.csv)"""
)


if df is not None:
    data=pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce')
    st.write(data)

### Step 2 Select Forecast Horizon 
st.markdown("### Step 2 Select Forecast Horizon ")

periods_input = st.number_input('How many periods would you like to forecast into the future?',min_value=1,max_value=365)


### Step 3: Visualize Forecast Data
st.markdown("### Step 3 Select Forecast Horizon ")

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


### Step 4 Output and download output
st.markdown("### Step 4 Output and download output")

st.dataframe(fcst_filtered)

if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    bs64 = base64.b64encode(csv_exp.encode()).decode()
    href = f'<a href="data:file/csv;base64,{bs64}">Download CSV File</a> (right-click and save as**&lt;forecast_name&gt;.csv**)'
    st.markdown(href,unsafe_allow_html=True)