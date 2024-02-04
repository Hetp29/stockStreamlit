import streamlit as st
import pandas as pd
from fetcher import get_stock_data
from predictor import predict_stock_prices

st.title("Stock Information App!")

ticker = st.text_input("Enter Stock Ticker Symbol (ex. AAPL): ", "AAPL")
start_date = st.date_input("Select Start Date:", pd.to_datetime('2022-01-01'))
end_date = st.date_input("Select End Date:", pd.to_datetime('2023-01-01'))

stock_data = get_stock_data(ticker, start_date, end_date)

st.subheader(f"Stock Data for {ticker}")
st.write(stock_data.head())

predictions = predict_stock_prices(stock_data)

st.subheader("Predicted Stock Prices")
st.line_chart(predictions)


