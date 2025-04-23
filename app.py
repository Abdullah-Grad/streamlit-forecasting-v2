import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, value

# --- Streamlit setup ---
st.set_page_config(layout='wide')
st.title("Salasa Demand Forecasting & Workforce Requirements 📈")

# --- Upload demand file ---
uploaded_file = st.file_uploader("📤 Upload your Monthly Demand Excel File", type=["xlsx"])
if uploaded_file:
    with st.spinner("⏳ Preparing data..."):
        df = pd.read_excel(uploaded_file)
        month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df_long = df.melt(id_vars='Year', value_vars=month_cols,
                          var_name='Month', value_name='Demand')
        df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month'], format='%Y-%b')
        df_long = df_long.sort_values('Date').reset_index(drop=True)
        df_long.set_index('Date', inplace=True)

    def add_promotion_factors(df):
        df['Promotion'] = 0
        for index, row in df.iterrows():
            if (row['ds'].month == 4 and row['ds'].year in [2023, 2024]) or \
               (row['ds'].month == 5 and row['ds'].year in [2020, 2021, 2022]) or \
               (row['ds'].month == 6 and row['ds'].year == 2019):
                df.at[index, 'Promotion'] = 1
            elif row['ds'].month in [2, 9, 11, 12]:
                df.at[index, 'Promotion'] = 1
        return df

    with st.spinner("🔍 Running cross-validation to find optimal model weights..."):
        train = df_long.iloc[:-12]
        test = df_long.iloc[-12:]

        sarima = SARIMAX(train['Demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
        sarima_pred = sarima.get_forecast(steps=12).predicted_mean

        dfp = train.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
        dfp['cap'] = dfp['y'].max() * 3
        dfp['floor'] = dfp['y'].min() * 0.5
        dfp['company_growth'] = dfp['ds'].dt.year - 2017
        dfp = add_promotion_factors(dfp)

        m = Prophet(growth='logistic', yearly_seasonality=True)
        m.add_regressor('company_growth')
        m.add_regressor('Promotion')
        m.fit(dfp[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])

        future = m.make_future_dataframe(periods=12, freq='MS')
        future['cap'] = dfp['cap'].iloc[0]
        future['floor'] = dfp['floor'].iloc[0]
        future['company_growth'] = future['ds'].dt.year - 2017
        future = add_promotion_factors(future)
        prophet_pred = m.predict(future)['yhat'].values[-12:]

        hw = ExponentialSmoothing(train['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
        hw_pred = hw.forecast(12).values

        best_mae = float('inf')
        best_weights = (1/3, 1/3, 1/3)
        for w1 in np.linspace(0, 1, 21):
            for w2 in np.linspace(0, 1 - w1, 21):
                w3 = 1 - w1 - w2
                blended = w1 * sarima_pred.values + w2 * prophet_pred + w3 * hw_pred
                mae = mean_absolute_error(test['Demand'].values, blended)
                if mae < best_mae:
                    best_mae = mae
                    best_weights = (w1, w2, w3)

        w1, w2, w3 = best_weights

    st.success(f"✅ Optimal Weights: SARIMA={w1:.2f}, Prophet={w2:.2f}, HW={w3:.2f}")
    st.info(f"📊 Cross-Validation MAE: {best_mae:.2f}")

    with st.spinner("📈 Forecasting demand for next 12 months..."):
        full_sarima = SARIMAX(df_long['Demand'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
        sarima_fc = full_sarima.get_forecast(steps=12).predicted_mean
        future_index = pd.date_range(df_long.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
        sarima_fc.index = future_index

        dfp_full = df_long.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
        dfp_full['cap'] = dfp_full['y'].max() * 3
        dfp_full['floor'] = dfp_full['y'].min() * 0.5
        dfp_full['company_growth'] = dfp_full['ds'].dt.year - 2017
        dfp_full = add_promotion_factors(dfp_full)

        m_full = Prophet(growth='logistic', yearly_seasonality=True)
        m_full.add_regressor('company_growth')
        m_full.add_regressor('Promotion')
        m_full.fit(dfp_full[['ds', 'y', 'cap', 'floor', 'company_growth', 'Promotion']])

        future_full = m_full.make_future_dataframe(periods=12, freq='MS')
        future_full['cap'] = dfp_full['cap'].iloc[0]
        future_full['floor'] = dfp_full['floor'].iloc[0]
        future_full['company_growth'] = future_full['ds'].dt.year - 2017
        future_full = add_promotion_factors(future_full)

        prophet_fc = m_full.predict(future_full)['yhat'].values[-12:]

        hw_full = ExponentialSmoothing(df_long['Demand'], trend='add', seasonal='add', seasonal_periods=12).fit()
        hw_fc = hw_full.forecast(12).values

        combined = w1 * sarima_fc.values + w2 * prophet_fc + w3 * hw_fc

    st.line_chart(pd.DataFrame({
        "SARIMA Forecast": sarima_fc,
        "Prophet Forecast": prophet_fc,
        "Holt-Winters Forecast": hw_fc,
        "Combined Forecast": combined
    }, index=future_index))
