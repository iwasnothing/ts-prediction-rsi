import pandas as pd
import yfinance as yf
import os
from datetime import datetime,timedelta
today_str = datetime.today().strftime('%Y-%m-%d')

def getBid(tic):
    i = yf.Ticker(tic).info
    return i["bid"]
def getAsk(tic):
    i = yf.Ticker(tic).info
    return i["ask"]

filename="today_rsi_signals.csv"
if os.path.isfile(filename):
    df = pd.read_csv(filename)
    buy_df = df.loc[df.signal == 1, :]
    sell_df = df.loc[df.signal == -1, :]
    buy_df["bought_date"] = buy_df["Asset"].apply(lambda x: today_str)
    buy_df["bought_price"] = buy_df["Asset"].apply(lambda x: getBid(x))
    buy_df["sold_date"] = buy_df["Asset"].apply(lambda x: "")
    buy_df["sold_price"] = buy_df["Asset"].apply(lambda x: 0)
    buy_df["Profit"] = buy_df["Asset"].apply(lambda x: 0)
    buy_df["Profit_Pct"] = buy_df["Asset"].apply(lambda x: 0)
    buy_df["Hold_days"] = buy_df["Asset"].apply(lambda x: 0)
    buy_df["status"] = buy_df["Asset"].apply(lambda x: "open")
    trade_book = "trade_book.csv"
    html = "trade_book.html"
    if os.path.isfile(trade_book):
        book = pd.read_csv(trade_book)
        open_list = book.loc[book.status == "open","Asset"].values
        for s in open_list:
            buy_df = buy_df.loc[buy_df.Asset != s, :]
        book = buy_df.append(book)
        sell_list = sell_df["Asset"].values
        for s in open_list:
            if s in sell_list:
                idx = book.loc[(book.Asset == s) & (book.status == "open"),:].index
                if len(idx) == 1 :
                    book.iloc[idx,book.columns.get_loc("sold_date")] = today_str
                    orig = book.iloc[idx,book.columns.get_loc("bought_price")].values[0]
                    sold = getAsk(s)
                    PL = sold - orig
                    rate = str(round( (sold - orig)/orig,2)) + '%'
                    book.iloc[idx,book.columns.get_loc("sold_price")] = sold
                    book.iloc[idx,book.columns.get_loc("Profit")] = PL
                    book.iloc[idx,book.columns.get_loc("Profit_Pct")] = rate
                    t0 = datetime.strptime(book.iloc[idx,book.columns.get_loc("bought_date")].values[0],'%Y-%m-%d')
                    d = datetime.today() - t0
                    book.iloc[idx,book.columns.get_loc("Hold_days")] = d.days
                    book.iloc[idx,book.columns.get_loc("status")] = "completed"
        book.to_csv(trade_book, index=False)
        book.to_html(html, index=False)
        print(book)
    else:
        buy_df.to_csv(trade_book,index=False)
        buy_df.to_html(html, index=False)

