# cv_helpers.py
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

def add_promotion_factors(df):
    # Assumes the date column is 'ds'
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

def cv_fold(i, initial_window, df_long):
    train_end = initial_window + i
    train = df_long.iloc[:train_end]
    test = df_long.iloc[train_end:train_end + 1]

    # SARIMAX forecast
    try:
        sarima_model = SARIMAX(train['Demand'],
                               order=(1, 1, 1),
                               seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        sarima_forecast = sarima_model.get_forecast(steps=1).predicted_mean.values[0]
    except Exception:
        sarima_forecast = 0

    # Prepare Prophet training data
    df_prophet_train = train.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
    df_prophet_train['cap'] = df_prophet_train['y'].max() * 3
    df_prophet_train['floor'] = df_prophet_train['y'].min() * 0.5
    df_prophet_train['company_growth'] = df_prophet_train['ds'].dt.year - 2017
    df_prophet_train = add_promotion_factors(df_prophet_train)

    # Prophet forecast
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
    prophet_forecast = model_prophet.predict(future)['yhat'].values[-1]

    # Exponential Smoothing forecast
    try:
        hw_model = ExponentialSmoothing(train['Demand'],
                                        trend='add',
                                        seasonal='add',
                                        seasonal_periods=12).fit()
        hw_forecast = hw_model.forecast(1).values[0]
    except Exception:
        hw_forecast = train['Demand'].mean()

    actual = test['Demand'].values[0]
    return (actual, sarima_forecast, prophet_forecast, hw_forecast)
