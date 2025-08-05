# ===== SALES FORECASTING DASHBOARD USING PROPHET =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# --- Step 1: Generate Sample Sales Data ---
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", periods=730)
sales = np.random.poisson(lam=200, size=730) + np.sin(np.linspace(0, 20, 730)) * 30
df = pd.DataFrame({'Date': dates, 'Sales': sales.astype(int)})

# --- Step 2: Visualize the Original Sales Data ---
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Sales'], color='blue', label='Actual Sales')
plt.title("Retail Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("raw_sales.png")
plt.close()

# --- Step 3: Prepare Data for Prophet ---
df_prophet = df.rename(columns={'Date': 'ds', 'Sales': 'y'})
df_prophet['y'] = df_prophet['y'].astype(float)

# --- Step 4: Create & Train the Forecasting Model ---
model = Prophet()
model.fit(df_prophet)

# --- Step 5: Make Forecast for Next 90 Days ---
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# --- Step 6: Plot Forecast Results ---
fig1 = model.plot(forecast)
plt.title("Sales Forecast (Next 90 Days)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("forecast_plot.png")
plt.close()

# --- Step 7: Plot Trend, Seasonality, etc. ---
fig2 = model.plot_components(forecast)
fig2.savefig("forecast_components.png")
plt.close()

# --- Step 8: Save Forecast Results to CSV ---
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast.csv", index=False)

# --- Step 9: Final Plot - Actual vs Predicted ---
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Sales'], label='Actual Sales', color='skyblue')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Sales', color='orange')
plt.axvline(x=df['Date'].max(), color='red', linestyle='--', label='Forecast Start')
plt.title("Actual vs Forecasted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final_forecast.png")
plt.close()

print("✅ Forecasting completed. Check the output files in your folder.")
