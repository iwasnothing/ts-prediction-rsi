# First, run `pip install ray`.
import yfinance as yf
from darts.metrics import mase
from All_Functions_Master_File import rsi,adder
import Primal_Functions_Performance_Evaluation
from darts.models import NBEATSModel
from darts import TimeSeries
import numpy as np
import pandas as pd
from darts.metrics import mase
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

        elif Data[i, rsi_col] > 70 and i > input_chunk_length:
            u = OnePrediction(Data[i-input_chunk_length:i+1, rsi_col] , model, 5)
            if u < Data[i, rsi_col]:
                Data[i, sell] = -1
    return Data

def getStock(tic,period):
    ticker = yf.Ticker(tic)
    df = ticker.history(period=period)
    return df.values[:,:4],df.index



@ray.remote
def f(stock):
    lookback = 14
    future_range = 5
    data, ts = getStock(stock, '1y')
    if len(data) < 100:
        return {"asset": stock, "baseline": 0, "ML": 0, "performance": 0}
    my_data = adder(data, 20)
    my_data = rsi(my_data, 14, 3, 4)
    my_data[:, 5:10] = 0
    my_data = simple_rsi_signal(my_data, 4, 30, 5, 6)
    expected_cost = 0.02
    lot = 1
    investment = 100
    my_data_ret = Primal_Functions_Performance_Evaluation.holding(my_data, 5, 6, 7, 8)
    my_data_eq = Primal_Functions_Performance_Evaluation.equity_curve(my_data_ret, 7, expected_cost, lot, investment)
    profit_pct0 = Primal_Functions_Performance_Evaluation.performance(my_data_eq, 7, my_data, stock, expected_cost, lot, investment)
    series = TimeSeries.from_values(np.float32(my_data[:, 4]))
    train, val = series[:100], series[100:]
    model = NBEATSModel(input_chunk_length=lookback, output_chunk_length=future_range)
    model.fit(train)
    pred = model.predict(n=len(val), series=train)
    err = mase(series, pred, train)
    print("validation mase: ",err)
    my_data[:, 9:13] = 0
    my_data = predicted_rsi_signal(my_data, 4, 40,model, 14, 9, 10)
    my_data_ret1 = Primal_Functions_Performance_Evaluation.holding(my_data, 9, 10, 11, 12)
    my_data_eq1 = Primal_Functions_Performance_Evaluation.equity_curve(my_data_ret1, 11, expected_cost, lot, investment)
    profit_pct1 = Primal_Functions_Performance_Evaluation.performance(my_data_eq1, 11, my_data, stock, expected_cost, lot, investment)
    return {"asset": stock, "baseline": profit_pct0, "ML": profit_pct1, "performance": profit_pct1-profit_pct0}

#list = [ "QQQ", "AAPL"]
my_file = open("naq100.txt", "r")
content = my_file.read()
list = content.split('\n')
print(list)
futures = [f.remote(i) for i in list]
df = pd.DataFrame(ray.get(futures)).sort_values(by=['ML'])
print(df)
df.to_csv("backtest_rsi_signals.csv",index=False)