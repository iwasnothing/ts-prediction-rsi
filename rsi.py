# First, run `pip install ray`.
import yfinance as yf
from All_Functions_Master_File import rsi,adder
from darts.models import NBEATSModel
from darts import TimeSeries
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import ray
ray.init()

today_str = datetime.today().strftime('%Y-%m-%d')

def getStock(tic,period):
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
    data, ts = getStock(stock, '1y')
    if len(data) < 100:
        return {"asset": stock, "signal": signal}
    my_data = rsi(data, 14, 3, 4)
    train = TimeSeries.from_values(np.float32(my_data[:, 4]))
    model = NBEATSModel(input_chunk_length=lookback, output_chunk_length=future_range)
    model.fit(train)
    pred = model.predict(n=future_range, series=train)
    u = pred.mean()[0]
    last_Rsi = my_data[-1,4]
    print(stock,last_Rsi,u)
    if last_Rsi < 30 and u > last_Rsi * 1.1:
        signal = 1
        high,low = take_profit(data, 3, lookback)

    elif last_Rsi > 70  and u < last_Rsi :
        signal = -1
    return {"DATE": today_str, "Asset": stock, "signal": signal, "take_profit": high, "stop_loss": low}

#list = [ "QQQ", "AAPL"]
#my_file = open("naq100.txt", "r")
#content = my_file.read()
#list = content.split('\n')
df = pd.read_csv('backtest_rsi_signals.csv')
list = df.loc[df.ML>0 ,'asset' ].values
print(list)
futures = [f.remote(i) for i in list]
df = pd.DataFrame(ray.get(futures)).sort_values(by=['signal'],ascending=False)
print(df)
df = df.loc[df.signal != 0,:]
df.to_csv("today_rsi_signals.csv",index=False)
df['Action'] = df['signal'].apply(lambda x: "BUY" if x == 1 else "SELL" if x == -1 else "NONE")
df['Target Price'] = df['take_profit'].apply(lambda x: str(round(x, 2)) if x > 0 else "")
df['Stop Loss'] = df['stop_loss'].apply(lambda x: str(round(x, 2)) if x > 0 else "")
df[["DATE","Action","Asset","Target Price","Stop Loss"]].to_html('today_rsi_signals.html',index=False)
