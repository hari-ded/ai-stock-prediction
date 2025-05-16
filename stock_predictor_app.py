import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #2E86AB;'>Cracking the Market Code</h1>", unsafe_allow_html=True)
st.subheader("AI-Powered Stock Price Prediction using Time Series Analysis")

# Input Section
stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY):", value='AAPL')
start_date = st.date_input("Select Start Date", value=datetime(2022, 1, 1))
end_date = st.date_input("Select End Date", value=datetime.today())

if st.button("Predict"):
    # Load data
    data = yf.download(stock, start=start_date, end=end_date)

    if data.empty:
        st.warning("No data found for the given stock symbol and date range.")
    else:
        # Simple Feature: Previous Day Close
        data['Prev Close'] = data['Close'].shift(1)
        data.dropna(inplace=True)

        X = data[['Prev Close']]
        y = data['Close']

        # Train-test split (80-20)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Display metrics
        rmse = mean_squared_error(y_test, preds, squared=False)
        st.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")

        # Visualize predictions
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.values, label="Actual", color='green')
        ax.plot(preds, label="Predicted", color='orange')
        ax.set_title(f"{stock} - Predicted vs Actual Closing Prices")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

        # Predict next day
        next_input = np.array([[data['Close'].iloc[-1]]])
        next_pred = model.predict(next_input)[0]
        st.success(f"Predicted Closing Price for Next Day: **${next_pred:.2f}**")
