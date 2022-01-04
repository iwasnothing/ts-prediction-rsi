# First, run `pip install ray`.
import yfinance as yf
from All_Functions_Master_File import rsi,adder,ma
from prophet import Prophet
from darts.models import NBEATSModel
from darts import TimeSeries
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import ray
ray.init()

today_str = datetime.today().strftime('%Y-%m-%d')
def getStock(tic,start_period):
    ticker = yf.Ticker(tic)
    df = ticker.history(start=start_period)
    return df.values[:,:4],df.index

def getStockFor(tic,period):
    ticker = yf.Ticker(tic)
    df = ticker.history(period=period)
    return df.values[:,:4],df.index

def take_profit(data,close,lookback):
    i = -1*lookback
    high = np.max(data[i:,close])
    sig = np.std(data[i:,close])
    print(high,sig,data[-1,close])
    if data[-1,close] > high:
        high = data[-1,close] + sig
    low = data[-1, close] - sig
    return high,low

@ray.remote
def f(stock):
    signal = 0
    high,low = 0,0
    lookback = 14
    future_range = 5
    data, ts = getStock(stock, '2017-12-01')
    if len(data) < 100:
        return {"asset": stock, "signal": signal}
    my_data = adder(data, 20)
    my_data = rsi(my_data, 14, 3, 4)
    my_data = ma(my_data, 20, 4, 5)
    my_data[:, 6:10] = 0
    ts = ts[-1 * len(my_data):]
    df = pd.DataFrame({'y': my_data[:, 5], 'ds': ts})
    m = Prophet(yearly_seasonality=True)
    m.add_seasonality(name='half_year', period=182, fourier_order=5)
    m.add_country_holidays(country_name='US')
    m.fit(df)
    pred_period = 5
    future = m.make_future_dataframe(periods=pred_period)
    forecast = m.predict(future)
    #print(forecast)
    pred = forecast.iloc[-1*pred_period:]['yhat'].values
    last_Rsi = my_data[-1,4]
    print("prediction",stock,last_Rsi,pred)
    if last_Rsi < 35 and pred[-1] > last_Rsi  and pred[-1] > pred[0]:
        signal = 1
        high,low = take_profit(data, 3, lookback)
    elif last_Rsi > 65  and pred[-1] < last_Rsi and pred[-1] < pred[0]:
        signal = -1
    return {"DATE": today_str, "Asset": stock, "signal": signal, "take_profit": high, "stop_loss": low}

#list = [ "QQQ", "AAPL"]
my_file = open("vgt.txt", "r")
content = my_file.read()
list = content.split('\n')
print(list)
futures = [f.remote(i) for i in list]
df = pd.DataFrame(ray.get(futures)).sort_values(by=['signal'],ascending=False)
print(df)
#df = df.loc[df.signal != 0,:]
df.to_csv("vgt_rsi_signals.csv",index=False)
df['Action'] = df['signal'].apply(lambda x: "BUY" if x == 1 else "SELL" if x == -1 else "NONE")
df['Target Price'] = df['take_profit'].apply(lambda x: str(round(x, 2)) if x > 0 else "")
df['Stop Loss'] = df['stop_loss'].apply(lambda x: str(round(x, 2)) if x > 0 else "")
df[["DATE","Action","Asset","Target Price","Stop Loss"]].to_html('vgt_rsi_signals.html',index=False)
