
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value
from google.colab import files

# --- 1. Upload and Prepare the Demand Data ---
print('Please upload your demand data file (Excel format).')
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_excel(file_name)

# Reshape to long format
month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df_long = df.melt(id_vars='Year', value_vars=month_cols,
                  var_name='Month', value_name='Demand')
df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'], format='%Y-%b')
df_long = df_long.sort_values('Date').reset_index(drop=True)
df_long.set_index('Date', inplace=True)

# --- 5. Forecast 2025 ---

# SARIMA Forecast
sarima_model = SARIMAX(df_long['Demand'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
sarima_future = sarima_model.get_forecast(steps=12).predicted_mean
future_index = pd.date_range(start=df_long.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
sarima_future.index = future_index

# Prophet Forecast (Enhanced)
df_prophet = df_long.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
df_prophet['cap'] = df_prophet['y'].max() * 3
df_prophet['floor'] = df_prophet['y'].min() * 0.5
df_prophet['company_growth'] = df_prophet['ds'].dt.year - 2017
df_prophet = add_promotion_factors(df_prophet)

model_prophet = Prophet(growth='logistic', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model_prophet.add_regressor('company_growth')
model_prophet.add_regressor('Promotion')
model_prophet.fit(df_prophet[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])

future = model_prophet.make_future_dataframe(periods=12, freq='MS')
future['cap'] = df_prophet['cap'].iloc[0]
future['floor'] = df_prophet['floor'].iloc[0]
future['company_growth'] = future['ds'].dt.year - 2017
future = add_promotion_factors(future)

prophet_future = model_prophet.predict(future)['yhat'].values[-12:]

# Holt-Winters Forecast
hw_model_full = ExponentialSmoothing(df_long['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
hw_future = hw_model_full.forecast(12).values

# Combined Forecast
w1, w2, w3 = best_weights
combined_forecast = w1 * sarima_future.values + w2 * prophet_future + w3 * hw_future

# --- 6. Workforce Scheduling using PuLP ---
M = 12  # Months
S = 3   # Shift types
Productivity = 23
Cost = 8.5
Days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
Hours = [6, 6, 6]

model = LpProblem("Workforce_Scheduling", LpMinimize)
X = {(i, j): LpVariable(f"X_{i}_{j}", lowBound=0, cat='Integer') for i in range(M) for j in range(S)}
model += lpSum(Cost * X[i, j] * Hours[j] * Days[i] for i in range(M) for j in range(S))

for i in range(M):
    model += lpSum(Productivity * X[i, j] * Hours[j] * Days[i] for j in range(S)) >= combined_forecast[i]

model.solve()

# --- 7. Output Results ---
print(f"\nBest Initial Window: {best_initial_window} months | Optimal Weights: SARIMA={w1:.2f}, Prophet={w2:.2f}, HW={w3:.2f} | Min MAE: {best_mae_global:.2f}")
print(f"{'Month':<10} {'Forecasted Demand':>20} {'Workers Required':>20}")
print("=" * 55)
for i in range(M):
    month_name = future_index[i].strftime('%b')
    demand = combined_forecast[i]
    workers = sum(value(X[i, j]) for j in range(S))
    print(f"{month_name:<10} {demand:>20.2f} {workers:>20.0f}")

    # --- 8. Plots ---
plt.figure(figsize=(14, 5))
plt.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
plt.plot(future_index, sarima_future, label='SARIMA Forecast', marker='o')
plt.title("SARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
plt.plot(future_index, prophet_future, label='Prophet Forecast', marker='o')
plt.title("Prophet Forecast (Enhanced)")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
plt.plot(future_index, hw_future, label='Holt-Winters Forecast', marker='o')
plt.title("Holt-Winters Forecast")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
plt.plot(future_index, combined_forecast, label='Forecasted Demand (Weighted)', marker='o', linestyle='--')
plt.title("Historical Demand + Next Year Weighted Forecast")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 9. In-Sample Fitted Forecast ---
sarima_fitted = sarima_model.fittedvalues

hw_model_full = ExponentialSmoothing(df_long['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
hw_fitted = hw_model_full.fittedvalues

df_prophet_fit = df_long.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
df_prophet_fit['cap'] = df_prophet_fit['y'].max() * 3
df_prophet_fit['floor'] = df_prophet_fit['y'].min() * 0.5
df_prophet_fit['company_growth'] = df_prophet_fit['ds'].dt.year - 2017
df_prophet_fit = add_promotion_factors(df_prophet_fit)

model_prophet_fit = Prophet(growth='logistic', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model_prophet_fit.add_regressor('company_growth')
model_prophet_fit.add_regressor('Promotion')
model_prophet_fit.fit(df_prophet_fit[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])

future_fit = df_prophet_fit[['ds', 'cap', 'floor', 'company_growth', 'Promotion']]
prophet_fitted = model_prophet_fit.predict(future_fit)['yhat'].values

combined_fitted = w1 * sarima_fitted.values + w2 * prophet_fitted + w3 * hw_fitted.values
combined_fitted_series = pd.Series(combined_fitted, index=df_long.index[:len(combined_fitted)])

# --- 10. Evaluation ---
print(f"Optimal Weights: SARIMA={w1:.2f}, Prophet={w2:.2f}, HW={w3:.2f} | In-Sample MAE: {mean_absolute_error(df_long['Demand'], combined_fitted_series):.2f}")
print(f"In-Sample MAPE: {mean_absolute_percentage_error(df_long['Demand'], combined_fitted_series) * 100:.2f}%")

# --- Final Plots ---
plt.figure(figsize=(14, 5))
plt.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
plt.plot(df_long.index, sarima_fitted, label='SARIMA Fitted', marker='o')
plt.title("SARIMA In-Sample Fit")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
plt.plot(df_long.index, prophet_fitted, label='Prophet Fitted', marker='o')
plt.title("Prophet In-Sample Fit")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
plt.plot(df_long.index, hw_fitted, label='Holt-Winters Fitted', marker='o')
plt.title("Holt-Winters In-Sample Fit")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df_long.index, df_long['Demand'], label='Historical Demand', marker='o')
plt.plot(combined_fitted_series.index, combined_fitted_series, label='Combined Weighted Fit', linestyle='--', marker='x')
plt.title("Historical Demand + In-Sample Fitted (Weighted)")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
