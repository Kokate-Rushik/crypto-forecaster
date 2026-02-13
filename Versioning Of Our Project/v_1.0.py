import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import streamlit as st

# --------------------------------------------------
# 1. Load Stock Price Data

data = pd.read_csv("stock_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# --------------------------------------------------
# 2. Generate Headlines Proxy from Price Movements

data['headline'] = data.apply(
    lambda row: "Bullish momentum" if row['Close'] > row['Open'] 
                else ("Bearish pressure" if row['Close'] < row['Open'] else "Neutral day"),
    axis=1
)

# --------------------------------------------------
# 3. VADER Sentiment

sia = SentimentIntensityAnalyzer()
data['vader_score'] = data['headline'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# --------------------------------------------------
# 4. FinBERT Sentiment

finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
data['finbert_label'] = data['headline'].apply(lambda x: finbert(str(x))[0]['label'])
label_map = {"positive": 1, "negative": -1, "neutral": 0}
data['finbert_score'] = data['finbert_label'].map(label_map)

# --------------------------------------------------
# 5. Market Psychology Mapping

def map_emotion_to_market(text):
    fear_signals = ["fear","anxiety","anger","pessimism","bearish"]
    greed_signals = ["joy","optimism","excitement","bullish","moon","hodl"]

    text = text.lower()
    if any(sig in text for sig in fear_signals):
        return "Market Fear"
    elif any(sig in text for sig in greed_signals):
        return "Market Greed"
    else:
        return "Neutral"

data['market_psychology'] = data['headline'].apply(map_emotion_to_market)

# --------------------------------------------------
# 6. Final Sentiment Score

data['sentiment_score'] = (data['vader_score'] + data['finbert_score']) / 2

# --------------------------------------------------
# 7. Forecasting Models

# ARIMA
model_arima = ARIMA(data['Close'], order=(5, 1, 0))
result_arima = model_arima.fit()
forecast_arima = result_arima.forecast(steps=30)

# SARIMA
model_sarima = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
result_sarima = model_sarima.fit()
forecast_sarima = result_sarima.forecast(steps=30)

# Prophet
prophet_df = data.reset_index()[['Date', 'Close']]
prophet_df.columns = ['ds', 'y']
model_prophet = Prophet()
model_prophet.fit(prophet_df)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)

# LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Close']])

X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i - 60:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X, y, epochs=10, batch_size=32, verbose=0)

pred_input = data_scaled[-60:].reshape(1, 60, 1)
lstm_predictions = []
for _ in range(30):
    next_pred = model_lstm.predict(pred_input, verbose=0)[0][0]
    lstm_predictions.append(next_pred)
    pred_input = np.append(pred_input[:, 1:, :], [[[next_pred]]], axis=1)
lstm_predictions_actual = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))

last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
lstm_forecast_df = pd.DataFrame({'Date': future_dates, 'LSTM_Predicted_Close': lstm_predictions_actual.flatten()})
lstm_forecast_df.set_index('Date', inplace=True)

# --------------------------------------------------
# 8. Streamlit Dashboard

st.title("📈 Stock Market Forecast + Sentiment Dashboard")
st.subheader("Historical Stock Price")
st.line_chart(data['Close'])

st.subheader("Sentiment Score Trend")
st.line_chart(data['sentiment_score'])

st.subheader("ARIMA Forecast")
st.line_chart(forecast_arima)

st.subheader("SARIMA Forecast")
st.line_chart(forecast_sarima)

st.subheader("Prophet Forecast")
st.line_chart(forecast_prophet[['ds', 'yhat']].set_index('ds').tail(30))

st.subheader("LSTM Forecast")
st.line_chart(lstm_forecast_df)

st.success("Forecasts and sentiment analysis generated successfully!")
