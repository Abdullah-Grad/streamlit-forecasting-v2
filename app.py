import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value
from scipy.optimize import minimize

# --- Streamlit setup ---
st.set_page_config(layout='wide')
st.markdown(
    "<div style='text-align: center;'><img src='https://raw.githubusercontent.com/Abdullah-Grad/streamlit-forecasting-v2/main/logo.png' width='200'></div>",
    unsafe_allow_html=True
)
st.title("Salasa Demand Forecasting & Workforce Requirements 📈")

# --- Upload demand file ---
uploaded_file = st.file_uploader("📤 Upload your Monthly Demand Excel File", type=["xlsx"])
if uploaded_file:
    with st.spinner("⏳ Processing data and building model..."):
        # Load and reshape data
        df = pd.read_excel(uploaded_file)
        month_cols = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        df_long = df.melt(id_vars='Year', value_vars=month_cols,
                          var_name='Month', value_name='Demand')
        df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'],
                                         format='%Y-%b')
        df_long = df_long.sort_values('Date').reset_index(drop=True)
        df_long.set_index('Date', inplace=True)

        # --- Vectorized promotion factors assignment ---
        def add_promotion_factors(df):
            # Assumes column 'ds' holds dates
            month = df['ds'].dt.month
            year = df['ds'].dt.year
            condition = (
                ((month == 4) & (year.isin([2023, 2024]))) |
                ((month == 5) & (year.isin([2020, 2021, 2022]))) |
                ((month == 6) & (year == 2019)) |
                (month == 9) |
                ((month == 2) & (year >= 2022)) |
                (month == 11) |
                (month == 12)
            )
            df['Promotion'] = condition.astype(int)
            return df

        # --- Robust cross-validation using optimizer for weight selection ---
        def cv_for_window(initial_window):
            # Use all available folds with training expanding until the end
            n_folds = len(df_long) - initial_window
            actuals = []
            sarima_forecasts = []
            prophet_forecasts = []
            hw_forecasts = []
            
            # Loop through each fold (each month in the hold-out period)
            for i in range(n_folds):
                # Expanding training window: from start until (initial_window + i)
                train = df_long.iloc[:initial_window + i]
                test = df_long.iloc[initial_window + i: initial_window + i + 1]
                if test.empty:
                    break

                # --- SARIMAX forecast ---
                try:
                    sarima_model = SARIMAX(train['Demand'], order=(1, 1, 1),
                                           seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                    sarima_pred = sarima_model.get_forecast(steps=1).predicted_mean.values[0]
                except Exception:
                    sarima_pred = 0

                # --- Prophet forecast ---
                df_prophet_train = train.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
                df_prophet_train['cap'] = df_prophet_train['y'].max() * 3
                df_prophet_train['floor'] = df_prophet_train['y'].min() * 0.5
                df_prophet_train['company_growth'] = df_prophet_train['ds'].dt.year - 2017
                df_prophet_train = add_promotion_factors(df_prophet_train)

                model_prophet = Prophet(growth='logistic',
                                        yearly_seasonality=True,
                                        weekly_seasonality=False,
                                        daily_seasonality=False)
                model_prophet.add_regressor('company_growth')
                model_prophet.add_regressor('Promotion')
                model_prophet.fit(df_prophet_train[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])
                future = model_prophet.make_future_dataframe(periods=1, freq='MS')
                future['cap'] = df_prophet_train['cap'].iloc[0]
                future['floor'] = df_prophet_train['floor'].iloc[0]
                future['company_growth'] = future['ds'].dt.year - 2017
                future = add_promotion_factors(future)
                prophet_pred = model_prophet.predict(future)['yhat'].values[-1]

                # --- Exponential Smoothing forecast ---
                try:
                    hw_model = ExponentialSmoothing(train['Demand'], trend='add',
                                                    seasonal='add', seasonal_periods=12).fit()
                    hw_pred = hw_model.forecast(1).values[0]
                except Exception:
                    hw_pred = train['Demand'].mean()

                # Append results from this fold
                actuals.append(test['Demand'].values[0])
                sarima_forecasts.append(sarima_pred)
                prophet_forecasts.append(prophet_pred)
                hw_forecasts.append(hw_pred)

            # --- Optimize blend weights via minimization ---
            forecasts = np.column_stack([sarima_forecasts, prophet_forecasts, hw_forecasts])
            
            def objective(weights):
                blended = np.dot(forecasts, weights)
                return mean_absolute_error(actuals, blended)
            
            # Constraint: weights sum to 1, and each is between 0 and 1
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1), (0, 1), (0, 1)]
            result = minimize(objective, x0=[1/3, 1/3, 1/3], bounds=bounds, constraints=constraints)
            best_weights = result.x
            best_mae = result.fun
            
            return best_mae, best_weights, initial_window, n_folds

        # --- Iterate over candidate initial windows to select the best ---
        candidate_windows = range(30, 49, 3)
        best_overall_mae = float('inf')
        best_initial_window = None
        best_weights_global = None
        best_n_folds = None
        for window in candidate_windows:
            mae, weights, init_win, n_folds = cv_for_window(window)
            if mae < best_overall_mae:
                best_overall_mae = mae
                best_initial_window = init_win
                best_weights_global = weights
                best_n_folds = n_folds

        st.info(f"📊 CV folds used (for best window): {best_n_folds}")
        w1, w2, w3 = best_weights_global
        st.success(f"✅ Optimal Weights: SARIMA={w1:.2f}, Prophet={w2:.2f}, HW={w3:.2f}")
        st.info(f"📊 Cross-Validation MAE: {best_overall_mae:.2f} (Initial window = {best_initial_window} months)")

        # --- Refit models on the full series and forecast the next 12 months ---
        sarima_model = SARIMAX(df_long['Demand'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
        sarima_future = sarima_model.get_forecast(steps=12).predicted_mean
        future_index = pd.date_range(start=df_long.index[-1] + pd.DateOffset(months=1),
                                     periods=12, freq='MS')
        sarima_future.index = future_index

        df_prophet = df_long.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
        df_prophet['cap'] = df_prophet['y'].max() * 3
        df_prophet['floor'] = df_prophet['y'].min() * 0.5
        df_prophet['company_growth'] = df_prophet['ds'].dt.year - 2017
        df_prophet = add_promotion_factors(df_prophet)

        model_prophet = Prophet(growth='logistic', yearly_seasonality=True,
                                weekly_seasonality=False, daily_seasonality=False)
        model_prophet.add_regressor('company_growth')
        model_prophet.add_regressor('Promotion')
        model_prophet.fit(df_prophet[['ds','y','cap','floor','company_growth','Promotion']])
        future = model_prophet.make_future_dataframe(periods=12, freq='MS')
        future['cap'] = df_prophet['cap'].iloc[0]
        future['floor'] = df_prophet['floor'].iloc[0]
        future['company_growth'] = future['ds'].dt.year - 2017
        future = add_promotion_factors(future)
        prophet_future = model_prophet.predict(future)['yhat'].values[-12:]

        hw_model_full = ExponentialSmoothing(df_long['Demand'], trend='add',
                                              seasonal='add', seasonal_periods=12).fit()
        hw_future = hw_model_full.forecast(12).values

        combined_forecast = w1 * sarima_future.values + w2 * prophet_future + w3 * hw_future

        # --- Workforce optimization ---
        M, S = 12, 3
        Productivity = 23
        Cost = 8.5
        Days = [31,28,31,30,31,30,31,31,30,31,30,31]
        Hours = [6,6,6]

        model = LpProblem("Workforce", LpMinimize)
        X = {(i, j): LpVariable(f"x_{i}_{j}", lowBound=0, cat="Integer")
             for i in range(M) for j in range(S)}
        model += lpSum(Cost * X[i, j] * Hours[j] * Days[i] for i in range(M) for j in range(S))
        for i in range(M):
            model += lpSum(Productivity * X[i, j] * Hours[j] * Days[i] for j in range(S)) >= combined_forecast[i]
        model.solve()
        st.info(f"💰 Total Workforce Cost: {value(model.objective):,.2f} SAR")

        df_results = pd.DataFrame({
            "Month": [d.strftime("%b %Y") for d in future_index],
            "📈 Forecasted Demand": combined_forecast,
            "👷 Workers Required": [sum(value(X[i, j]) for j in range(S)) for i in range(M)]
        })
        st.dataframe(df_results)

        # --- Plot Historical + Forecasted Demand ---
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df_long.index, df_long["Demand"], label="Historical", marker="o")
        ax.plot(future_index, combined_forecast, label="Forecast (Weighted)", marker="o")
        ax.set_title("Historical + Forecasted Demand")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # --- In-sample fit evaluation ---
        sarima_fitted = sarima_model.fittedvalues
        hw_fitted = hw_model_full.fittedvalues
        prophet_fit = model_prophet.predict(df_prophet[["ds","cap","floor","company_growth","Promotion"]])["yhat"].values
        combined_fit = w1 * sarima_fitted.values + w2 * prophet_fit + w3 * hw_fitted.values
        fit_series = pd.Series(combined_fit, index=df_long.index)

        mae_fit = mean_absolute_error(df_long["Demand"], fit_series)
        mape_fit = mean_absolute_percentage_error(df_long["Demand"], fit_series) * 100
        st.info(f"📎 In-Sample Fitted MAE: {mae_fit:.2f} | MAPE: {mape_fit:.2f}%")

        fig2, ax2 = plt.subplots(figsize=(12,5))
        ax2.plot(df_long.index, df_long["Demand"], label="Actual", marker="o")
        ax2.plot(df_long.index, fit_series, label="Fitted (Weighted)", marker="x", linestyle="--")
        ax2.set_title("In-Sample Fitted vs Actual")
        ax2.grid()
        ax2.legend()
        st.pyplot(fig2)
