# ibraries Required
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Data Loading
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-12-31')

# EDA
plt.figure(figsize=(12, 5))
plt.plot(data['Close'], label='Closing Price')
plt.title(f'{ticker} Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.savefig("images/closing_price_trend.png")
plt.close()

# Mapping
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig("images/correlation_heatmap.png")
plt.close()


data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['Lag1'] = data['Close'].shift(1)

data.dropna(inplace=True)

# Processing of Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close', 'MA10', 'MA50', 'Lag1']])

# Sequences
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i+time_step])
        y.append(dataset[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Train-Test Split
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Model Evaluation
predictions = model.predict(X_test)
y_test_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1),
                                                      np.zeros((len(y_test), 3)))))[:, 0]
predictions_rescaled = scaler.inverse_transform(np.hstack((predictions,
                                                           np.zeros((len(predictions), 3)))))[:, 0]

plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled, label='Actual Price')
plt.plot(predictions_rescaled, label='Predicted Price')
plt.title('Predicted vs Actual Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.savefig("images/predicted_vs_actual.png")
plt.close()

# Step 9: Print Metrics
print("MAE:", mean_absolute_error(y_test_rescaled, predictions_rescaled))
print("RMSE:", mean_squared_error(y_test_rescaled, predictions_rescaled, squared=False))
print("R2 Score:", r2_score(y_test_rescaled, predictions_rescaled))
