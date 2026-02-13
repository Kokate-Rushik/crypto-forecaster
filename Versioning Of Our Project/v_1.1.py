import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Crypto Forecast AI", layout="wide")

# ------------------ DATASET PATHS ------------------
DATASETS = {
    "BTC": "Data/BTC_historical_INR.csv",
    "ETH": "Data/ETH_historical_INR.csv",
    "SOL": "Data/SOL_historical_INR.csv",
    "USDC": "Data/USDC_historical_INR.csv",
    "USDT": "Data/USDT_historical_INR.csv"
}

# ------------------ SIDEBAR ------------------
st.sidebar.title("Select Cryptocurrency")
coin = st.sidebar.selectbox("Choose Coin", list(DATASETS.keys()))

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

data = load_data(DATASETS[coin])

st.title(f"📈 {coin} Forecast & Sentiment Dashboard")

# ------------------ HISTORICAL GRAPH ------------------
st.subheader("Historical Price Trend")
st.line_chart(data['Close'])

# ------------------ ARIMA ------------------
st.subheader("ARIMA Forecast")
model_arima = ARIMA(data['Close'], order=(5,1,0))
res_arima = model_arima.fit()
forecast_arima = res_arima.forecast(30)
st.line_chart(forecast_arima)

# ------------------ PROPHET ------------------
st.subheader("Prophet Forecast")
prophet_df = data.reset_index()[['Date','Close']]
prophet_df.columns = ['ds','y']
model_prophet = Prophet()
model_prophet.fit(prophet_df)
future = model_prophet.make_future_dataframe(periods=30)
forecast_prophet = model_prophet.predict(future)
st.line_chart(forecast_prophet[['ds','yhat']].set_index('ds').tail(30))

# ------------------ LSTM ------------------
st.subheader("LSTM Forecast")
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[['Close']])

X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(loss='mse', optimizer='adam')
model_lstm.fit(X, y, epochs=5, batch_size=32, verbose=0)

pred_input = scaled[-60:].reshape(1,60,1)
preds = []

for _ in range(30):
    p = model_lstm.predict(pred_input, verbose=0)[0][0]
    preds.append(p)
    pred_input = np.append(pred_input[:,1:,:], [[[p]]], axis=1)

preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
future_dates = pd.date_range(data.index[-1], periods=30)
lstm_df = pd.DataFrame(preds, index=future_dates, columns=['Prediction'])
st.line_chart(lstm_df)

# ------------------ SENTIMENT ANALYSIS ------------------
st.subheader("Market Sentiment Analysis")

sia = SentimentIntensityAnalyzer()
data['headline'] = data.apply(
    lambda r: "Bullish trend" if r['Close'] > r['Open'] else "Bearish trend", axis=1
)
data['sentiment_score'] = data['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

st.line_chart(data['sentiment_score'])

avg_sentiment = data['sentiment_score'].mean()

if avg_sentiment > 0.2:
    st.success("Market Mood: Greed / Bullish 🟢")
elif avg_sentiment < -0.2:
    st.error("Market Mood: Fear / Bearish 🔴")
else:
    st.warning("Market Mood: Neutral 🟡")

st.success("Dashboard Loaded Successfully 🚀")
