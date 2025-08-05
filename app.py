# AI-Powered Sales Forecasting Dashboard using Streamlit + Prophet
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page setup
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📈 AI-Powered Sales Forecasting Dashboard")

# Sidebar
st.sidebar.header("📊 Settings")
periods_input = st.sidebar.slider("Forecast Period (Days)", min_value=30, max_value=180, value=90)

# Generate Sample Sales Data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=730)
    sales = np.random.poisson(lam=200, size=730) + np.sin(np.linspace(0, 20, 730)) * 30
    df = pd.DataFrame({'Date': dates, 'Sales': sales.astype(int)})
    return df

df = generate_sample_data()
st.subheader("🗃️ Historical Sales Data")
st.line_chart(df.set_index('Date')['Sales'])

# Prophet requires 'ds' and 'y' columns
df_prophet = df.rename(columns={'Date': 'ds', 'Sales': 'y'})
model = Prophet()
model.fit(df_prophet)

# Future prediction
future = model.make_future_dataframe(periods=periods_input)
forecast = model.predict(future)

# Show forecast table
st.subheader("🔮 Forecast Table")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_input))

# Plot forecast chart
st.subheader("📉 Forecast Chart")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Show components (trend, weekly seasonality, etc.)
st.subheader("🧩 Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# Download forecast
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button("📥 Download Forecast CSV", data=csv, file_name='forecast.csv', mime='text/csv')

st.success("✅ Forecasting Completed Successfully!")
