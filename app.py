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

predictions = model.predict(X)
torch_predictions = predict_stock_prices(stock_data)
combined_predictions = 0.5 * predictions + 0.5 * torch_predictions

future_dates = pd.date_range(start=start_date, periods=len(stock_data)+10, freq='B')
future_X = pd.Series(range(len(stock_data), len(stock_data) + len(future_dates)))
future_predictions = model.predict(future_X.values.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'], y, label='Actual Prices')
plt.plot(stock_data['Date'], predictions, label='Linear Regression Predictions')
plt.plot(stock_data['Date'], torch_predictions, label='PyTorch Predictions')
plt.plot(stock_data['Date'], combined_predictions, label='Combined Predictions', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f"Stock Prices and Predictions for {ticker}")
plt.legend()
st.pyplot(plt)

plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'], y, label='Actual Prices')
plt.plot(stock_data['Date'], predictions, label='Past Predictions')
future_dates_extended = pd.date_range(start=stock_data['Date'].iloc[-1], periods=len(future_dates), freq='B')[1:]
plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f"Stock Prices and Predictions for {ticker}")
plt.legend()
st.pyplot(plt)

st.subheader("Linear Regression Model Coefficients")
st.write(f"Intercept: {model.intercept_}")
st.write(f"Coefficient: {model.coef_[0]}")

predictions_df = pd.DataFrame({'Date': stock_data['Date'], 'Actual': y, 'Predicted': predictions})
st.subheader("Past Stock Price Predictions")
st.write(predictions_df)

future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_predictions})
st.subheader("Future Stock Price Predictions")
st.write(future_predictions_df)


st.subheader("Predicted Stock Prices")
st.line_chart(predictions)

#

