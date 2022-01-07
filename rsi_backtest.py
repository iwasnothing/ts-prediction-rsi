# First, run `pip install ray`.
import yfinance as yf
from darts.metrics import mase
from All_Functions_Master_File import rsi,adder,ma
import Primal_Functions_Performance_Evaluation
from darts.models import NBEATSModel
from darts import TimeSeries
from prophet import Prophet
import numpy as np
import pandas as pd
from darts.metrics import mase
from sklearn import metrics
import ray
ray.init()

def OnePrediction(hist,model,future):
    hist_series = TimeSeries.from_values(np.float32(hist))
    pred = model.predict(n=future,series=hist_series)
    return pred.mean()[0]


def simple_rsi_signal(Data, rsi_col, threshold, buy, sell):
    high = 100 - threshold
    low = threshold
    Data[:, buy] = 0
    Data[:, sell] = 0
    for i in range(len(Data)):

        if Data[i, rsi_col] < low and i > 14:

            Data[i, buy] = 1

        elif Data[i, rsi_col] > high:

            Data[i, sell] = -1
    return Data


def predicted_rsi_signal(Data, rsi_col, threshold,model, input_chunk_length, buy, sell):
    high = 100 - threshold
    low = threshold
    Data[:, buy] = 0
    Data[:, sell] = 0
    for i in range(len(Data)):
        if Data[i, rsi_col] < low and i > input_chunk_length:
            u = OnePrediction(Data[i - input_chunk_length:i + 1, rsi_col], model, 5)
            if u > Data[i, rsi_col] * 1.1:
                Data[i, buy] = 1

        elif Data[i, rsi_col] > 60 and i > input_chunk_length:
            u = OnePrediction(Data[i-input_chunk_length:i+1, rsi_col] , model, 5)
            if u < Data[i, rsi_col]:
                Data[i, sell] = -1
    return Data

def prophet_rsi_signal(Data, rsi_col, threshold,forecast, buy, sell):
    low = threshold
    Data[:, buy] = 0
    Data[:, sell] = 0
    for i in range(len(Data)):
        #print("rsi prediction",Data[i,rsi_col],forecast[i],forecast[i+5])
        if Data[i, rsi_col] < low and forecast[i+5] > forecast[i] and forecast[i+5] > low:
            Data[i, buy] = 1
        elif Data[i, rsi_col] > 65 and forecast[i + 5] < forecast[i] and forecast[i + 5] < 65:
            Data[i, sell] = -1
    return Data

def getStockFor(tic,period):
    ticker = yf.Ticker(tic)
    df = ticker.history(period=period)
    return df.values[:,:4],df.index
def getStock(tic,start_period):
    ticker = yf.Ticker(tic)
    df = ticker.history(start=start_period)
    return df.values[:,:4],df.index


@ray.remote
def f(stock):
    lookback = 14
    future_range = 5
    data, ts = getStock(stock, '2017-12-01')
    if len(data) < 100:
        return {"asset": stock, "baseline": 0, "ML": 0, "performance": 0}
    my_data = adder(data, 20)
    my_data = rsi(my_data, 14, 3, 4)
    my_data = ma(my_data, 20, 4, 5)
    my_data[:, 6:10] = 0
    test_data = my_data[-100:,:]
    test_data = simple_rsi_signal(test_data, 5, 40, 6, 7)
    expected_cost = 0.02
    lot = 1
    investment = 100
    test_data_ret = Primal_Functions_Performance_Evaluation.holding(test_data, 6, 7, 8, 9)
    test_data_eq = Primal_Functions_Performance_Evaluation.equity_curve(test_data_ret, 8, expected_cost, lot, investment)
    profit_pct0 = Primal_Functions_Performance_Evaluation.performance(test_data_eq, 8, my_data, stock, expected_cost, lot, investment)
    #series = TimeSeries.from_values(np.float32(my_data[:, 5]))
    #train, val = series[:-100], series[-100:]
    #model = NBEATSModel(input_chunk_length=lookback, output_chunk_length=future_range)
    ts = ts[-1 * len(my_data):]
    df = pd.DataFrame({'y': my_data[:, 5], 'ds': ts})
    train, val = df.iloc[:-100], df.iloc[-100:]
    m = Prophet(yearly_seasonality=True)
    m.add_seasonality(name='half_year', period=182, fourier_order=5)
    m.add_country_holidays(country_name='US')
    m.fit(train)
    future = m.make_future_dataframe(periods=105)
    forecast = m.predict(future)
    #print(forecast)
    err = metrics.mean_absolute_error(val['y'].values, forecast.iloc[-105:-5]['yhat'].values)
    print("validation mase: ",err)
    my_data[:, 9:13] = 0
    test_data1 = my_data[-100:,:]
    test_data1 = prophet_rsi_signal(test_data1, 4, 35,forecast.iloc[-105:]['yhat'].values, 9, 10)
    test_data_ret1 = Primal_Functions_Performance_Evaluation.holding(test_data1, 9, 10, 11, 12)
    test_data_eq1 = Primal_Functions_Performance_Evaluation.equity_curve(test_data_ret1, 11, expected_cost, lot, investment)
    profit_pct1 = Primal_Functions_Performance_Evaluation.performance(test_data_eq1, 11, my_data, stock, expected_cost, lot, investment)
    return {"asset": stock, "baseline": profit_pct0, "ML": profit_pct1, "performance": profit_pct1-profit_pct0}

my_file = open("vgt.txt", "r")
content = my_file.read()
list = content.split('\n')
print(list)
futures = [f.remote(i) for i in list]
df = pd.DataFrame(ray.get(futures)).sort_values(by=['ML'])
print(df)
df.to_csv("backtest_rsi_signals.csv",index=False)