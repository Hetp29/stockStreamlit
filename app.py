import streamlit as st
import pandas as pd
from fetcher import get_stock_data
from predictor import predict_stock_prices
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt

st.title("Stock Information App!")

ticker = st.text_input("Enter Stock Ticker Symbol (ex. AAPL): ", "AAPL")
start_date = st.date_input("Select Start Date:", pd.to_datetime('2024-02-01'))
end_date = st.date_input("Select End Date:", pd.to_datetime('2024-02-03'))

stock_data = get_stock_data(ticker, start_date, end_date)

st.subheader(f"Stock Data for {ticker}")
st.write(stock_data.head())

model = LinearRegression()
X = stock_data.index.values.reshape(-1, 1)
y = stock_data['Close'].values
model.fit(X, y)
predictions_linear = model.predict(X)

torch_predictions = predict_stock_prices(stock_data)

st.subheader("Linear Regression Predictions")
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'], y, label='Actual Prices')
plt.plot(stock_data['Date'], predictions_linear, label='Linear Regression Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f"Stock Prices and Predictions for {ticker}")
plt.legend()
st.pyplot(plt)

st.subheader("PyTorch Neural Network Predictions")
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'], y, label='Actual Prices')
plt.plot(stock_data['Date'], torch_predictions, label='PyTorch Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f"PyTorch Neural Network Predictions for {ticker}")
plt.legend()
st.pyplot(plt)

predictions_df = pd.DataFrame({'Date': stock_data['Date'], 'Actual': y, 'Linear Regression': predictions_linear})
st.subheader("Past Stock Price Predictions")
st.write(predictions_df)



st.subheader("Predicted Stock Prices")
st.line_chart(predictions)

#

