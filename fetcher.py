import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.index = stock_data.index.tz_localize(None)
    stock_data.reset_index(inplace=True)
    return stock_data
