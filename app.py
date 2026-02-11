import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from supabase import create_client

st.set_page_config(layout="wide", page_title="Crypto AI Forecast")

# ================= SUPABASE AUTH =================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if "user" not in st.session_state:
    st.session_state.user = None

def login(email, password):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        st.session_state.user = res.user
        st.success("Login successful")
    except Exception as e:
        st.error("Invalid credentials")

def signup(email, password):
    try:
        res = supabase.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True
        })
        st.success("Signup successful! You can login now.")
    except Exception as e:
        st.error(f"Signup failed: {e}")



# LOGIN PAGE
if st.session_state.user is None:
    st.title("🔐 Login to Crypto AI Dashboard")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(email, password)

    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        if st.button("Signup"):
            signup(new_email, new_password)

    st.stop()

# LOGOUT
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# ================= DATA =================
DATASETS = {
    "BTC": "data/BTC_historical_INR.csv",
    "ETH": "data/ETH_historical_INR.csv",
    "SOL": "data/SOL_historical_INR.csv",
    "USDC": "data/USDC_historical_INR.csv",
    "USDT": "data/USDT_historical_INR.csv"
}

@st.cache_data
def load_price(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

@st.cache_data
def load_news():
    url = "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/bitcoin_news.csv"
    return pd.read_csv(url)

@st.cache_resource
def load_finbert():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

coin = st.sidebar.selectbox("Select Coin", list(DATASETS.keys()))
data = load_price(DATASETS[coin])
news = load_news()
finbert = load_finbert()

st.title(f"{coin} AI Forecast + News Sentiment")

# ================= SENTIMENT =================
st.subheader("Crypto News Sentiment")
sia = SentimentIntensityAnalyzer()

news['vader'] = news['title'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
news['finbert_label'] = news['title'].apply(lambda x: finbert(str(x))[0]['label'])
label_map = {"positive":1,"negative":-1,"neutral":0}
news['finbert_score'] = news['finbert_label'].str.lower().map(label_map)
news['final_score'] = (news['vader'] + news['finbert_score']) / 2

st.line_chart(news['final_score'])

avg_sent = news['final_score'].mean()
if avg_sent > 0.1:
    st.success("Market Sentiment: Bullish 🟢")
elif avg_sent < -0.1:
    st.error("Market Sentiment: Bearish 🔴")
else:
    st.warning("Market Sentiment: Neutral 🟡")

# ================= FORECASTS =================

st.subheader("Historical Price")
st.line_chart(data['Close'])

# ARIMA
@st.cache_resource
def run_arima(series):
    model = ARIMA(series, order=(2,1,2)).fit()
    return model.forecast(30)

st.subheader("ARIMA Forecast")
st.line_chart(run_arima(data['Close']))

# SARIMA
@st.cache_resource
def run_sarima(series):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
    return model.forecast(30)

st.subheader("SARIMA Forecast")
st.line_chart(run_sarima(data['Close']))

# Prophet
@st.cache_resource
def run_prophet(df):
    p_df = df.reset_index()[['Date','Close']]
    p_df.columns = ['ds','y']
    model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
    model.fit(p_df)
    future = model.make_future_dataframe(periods=30)
    return model.predict(future)[['ds','yhat']].set_index('ds').tail(30)

st.subheader("Prophet Forecast")
st.line_chart(run_prophet(data))

# LSTM
@st.cache_resource
def run_lstm(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)

    pred_input = scaled[-60:].reshape(1,60,1)
    preds = []
    for _ in range(30):
        p = model.predict(pred_input, verbose=0)[0][0]
        preds.append(p)
        pred_input = np.append(pred_input[:,1:,:], [[[p]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    future_dates = pd.date_range(series.index[-1], periods=30)
    return pd.DataFrame(preds, index=future_dates)

st.subheader("LSTM Forecast")
st.line_chart(run_lstm(data['Close']))

st.success("🚀 AI Forecasting Platform Running Securely")