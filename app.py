import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value

st.set_page_config(layout='wide')
st.markdown("<div style='text-align: center;'><img src='logo.png' width='200'/></div>", unsafe_allow_html=True)
st.title("📈 Demand Forecasting & Workforce Scheduling")

def add_promotion_factors(df):
    df['Promotion'] = 0
    for index, row in df.iterrows():
        if (row['ds'].month == 4 and row['ds'].year in [2023, 2024]) or \
           (row['ds'].month == 5 and row['ds'].year in [2020, 2021, 2022]) or \
           (row['ds'].month == 6 and row['ds'].year == 2019) or \
           (row['ds'].month in [2, 9, 11, 12]):
            df.at[index, 'Promotion'] = 1
    return df

st.subheader("Upload Monthly Demand Excel File")
uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])

if uploaded_file:
    with st.spinner("Processing and forecasting..."):
        df = pd.read_excel(uploaded_file)
        month_cols = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        df_long = df.melt(id_vars='Year', value_vars=month_cols, var_name='Month', value_name='Demand')
        df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'], format='%Y-%b')
        df_long = df_long.sort_values('Date').reset_index(drop=True)
        df_long.set_index('Date', inplace=True)

        def run_cv(initial_window):
            actuals, sarima_preds, prophet_preds, hw_preds = [], [], [], []
            n_splits = 12
            for i in range(n_splits):
                train_end = initial_window + i
                train = df_long.iloc[:train_end]
                test = df_long.iloc[train_end:train_end + 1]
                if len(test) == 0:
                    break

                try:
                    sarima = SARIMAX(train['Demand'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
                    sarima_preds.append(sarima.get_forecast(1).predicted_mean.values[0])
                except:
                    sarima_preds.append(0)

                dfp = train.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
                dfp['cap'] = dfp['y'].max() * 3
                dfp['floor'] = dfp['y'].min() * 0.5
                dfp['company_growth'] = dfp['ds'].dt.year - 2017
                dfp = add_promotion_factors(dfp)

                m = Prophet(growth='logistic', yearly_seasonality=True)
                m.add_regressor('company_growth')
                m.add_regressor('Promotion')
                m.fit(dfp[['ds','y','cap','floor','company_growth','Promotion']])
                future = m.make_future_dataframe(1, freq='MS')
                future['cap'] = dfp['cap'].iloc[0]
                future['floor'] = dfp['floor'].iloc[0]
                future['company_growth'] = future['ds'].dt.year - 2017
                future = add_promotion_factors(future)
                prophet_preds.append(m.predict(future)['yhat'].values[-1])

                try:
                    hw = ExponentialSmoothing(train['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
                    hw_preds.append(hw.forecast(1).values[0])
                except:
                    hw_preds.append(train['Demand'].mean())

                actuals.append(test['Demand'].values[0])

            best_mae = float('inf')
            best_weights = (1/3, 1/3, 1/3)
            for w1 in np.linspace(0, 1, 21):
                for w2 in np.linspace(0, 1 - w1, 21):
                    w3 = 1 - w1 - w2
                    blend = w1*np.array(sarima_preds) + w2*np.array(prophet_preds) + w3*np.array(hw_preds)
                    mae = mean_absolute_error(actuals, blend)
                    if mae < best_mae:
                        best_mae = mae
                        best_weights = (w1, w2, w3)

            return best_mae, best_weights

        # CV optimization
        best_mae = float('inf')
        best_initial_window = 36
        for win in range(30, 49, 3):
            mae, _ = run_cv(win)
            if mae < best_mae:
                best_mae = mae
                best_initial_window = win

        _, (w1, w2, w3) = run_cv(best_initial_window)

        # Final forecast
        sarima_model = SARIMAX(df_long['Demand'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
        sarima_forecast = sarima_model.get_forecast(12).predicted_mean
        future_index = pd.date_range(df_long.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
        sarima_forecast.index = future_index

        dfp = df_long.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
        dfp['cap'] = dfp['y'].max() * 3
        dfp['floor'] = dfp['y'].min() * 0.5
        dfp['company_growth'] = dfp['ds'].dt.year - 2017
        dfp = add_promotion_factors(dfp)

        m = Prophet(growth='logistic', yearly_seasonality=True)
        m.add_regressor('company_growth')
        m.add_regressor('Promotion')
        m.fit(dfp[['ds','y','cap','floor','company_growth','Promotion']])
        future = m.make_future_dataframe(12, freq='MS')
        future['cap'] = dfp['cap'].iloc[0]
        future['floor'] = dfp['floor'].iloc[0]
        future['company_growth'] = future['ds'].dt.year - 2017
        future = add_promotion_factors(future)
        prophet_forecast = m.predict(future)['yhat'].values[-12:]

        hw = ExponentialSmoothing(df_long['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
        hw_forecast = hw.forecast(12).values

        combined = w1 * sarima_forecast.values + w2 * prophet_forecast + w3 * hw_forecast

        # Workforce optimization
        M, S = 12, 3
        Productivity, Cost = 23, 8.5
        Days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        Hours = [6, 6, 6]

        model = LpProblem("Workforce", LpMinimize)
        X = {(i,j): LpVariable(f"x_{i}_{j}", lowBound=0, cat='Integer') for i in range(M) for j in range(S)}
        model += lpSum(Cost * X[i,j] * Hours[j] * Days[i] for i in range(M) for j in range(S))
        for i in range(M):
            model += lpSum(Productivity * X[i,j] * Hours[j] * Days[i] for j in range(S)) >= combined[i]
        model.solve()
        total_cost = value(model.objective)

    # ✅ Display Results
    st.success(f"Optimal Weights → SARIMA={w1:.2f}, Prophet={w2:.2f}, Holt-Winters={w3:.2f}")
    st.info(f"💰 Total Workforce Cost: **{total_cost:,.2f} SAR**")

    result = pd.DataFrame({
        'Month': [d.strftime('%b %Y') for d in future_index],
        'Forecasted Demand': combined,
        'Workers Required': [sum(value(X[i,j]) for j in range(S)) for i in range(M)]
    })
    st.subheader("📊 Monthly Forecasts and Required Workers")
    st.dataframe(result)

    # ✅ Individual Forecast Plots
    def plot_forecast(label, forecast):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_long.index, df_long['Demand'], label='Historical', marker='o')
        ax.plot(future_index, forecast, label=label, marker='o')
        ax.set_title(f"{label} Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    st.subheader("📉 Forecast Plots")
    plot_forecast("SARIMA", sarima_forecast)
    plot_forecast("Prophet", prophet_forecast)
    plot_forecast("Holt-Winters", hw_forecast)
    plot_forecast("Combined Weighted", combined)
