import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Cracking the Market Code")
st.subheader("AI-Driven Stock Price Prediction Using Time Series Analysis")

# Stock input
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY):", value='AAPL')

# Date range
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if st.button("Predict"):
    data = yf.download(stock, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found. Please check the symbol or date range.")
    else:
        # Feature Engineering
        data['Prev Close'] = data['Close'].shift(1)
        data.dropna(inplace=True)

        X = data[['Prev Close']]
        y = data['Close']

        # Split data
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Predict next day
        last_close = float(data['Close'].iloc[-1])
        next_input = np.array([last_close]).reshape(1, -1)  # Ensures 2D
        next_price = model.predict(next_input)[0]


        # Graph
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values, label='Actual', color='green', linewidth=2)
        ax.plot(preds, label='Predicted', color='orange', linestyle='--', linewidth=2)
        ax.set_title(f"{stock} - Actual vs Predicted")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Output
        next_price = float(next_price)  # force to regular float
        rmse = float(mean_squared_error(y_test, preds, squared=False))  # convert RMSE to float

        st.success(f"Predicted Next Closing Price: **${next_price:.2f}**")
        st.metric("Model RMSE", f"${rmse:.2f}")

