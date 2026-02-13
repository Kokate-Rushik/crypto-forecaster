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
import base64

# ================= PAGE CONFIG =================
st.set_page_config(
    layout="wide",
    page_title="Quantara",
    page_icon="🚀"
)

# ================= MODERN RESPONSIVE CSS =================
st.markdown("""
<style>

/* Global Padding */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
    color: white;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1e40af);
    color: white;
    border-radius: 8px;
    height: 45px;
    font-size: 15px;
    border: none;
    width: 100%;
}

/* Card */
.card {
    padding: 30px;
    border-radius: 12px;
    background: #ffffff;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
}

/* Banner */
.banner-container {
    position: relative;
    width: 100%;
    height: 220px;
    margin-bottom: 40px;
}

.banner-container img {
    width: 100%;
    height: 220px;
    object-fit: cover;
    border-radius: 12px;
}

.banner-text {
    position: absolute;
    bottom: 20px;
    left: 40px;
    font-size: 30px;
    font-weight: 600;
    color: #111;
    background: rgba(255,255,255,0.75);
    padding: 8px 18px;
    border-radius: 8px;
}

/* Responsive */
@media (max-width: 768px) {
    .banner-text {
        font-size: 18px;
        left: 15px;
        right: 15px;
        text-align: center;
    }
    .banner-container {
        height: 160px;
    }
    .banner-container img {
        height: 160px;
    }
}

.footer {
    text-align:center;
    padding: 20px;
    color: gray;
    font-size:13px;
}

</style>
""", unsafe_allow_html=True)

# ================= SUPABASE =================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if "user" not in st.session_state:
    st.session_state.user = None

# ================= AUTH FUNCTIONS =================
def login(email, password):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        st.session_state.user = res.user
        st.success("Login successful ✅")
        st.rerun()
    except:
        st.error("Invalid credentials ❌")

def signup(email, password):
    try:
        supabase.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True
        })
        st.success("Signup successful! Please login.")
    except Exception as e:
        st.error(f"Signup failed: {e}")

# ================= LOGIN PAGE =================
if st.session_state.user is None:

    st.markdown("<h1 style='text-align:center;'>🚀 Quantara</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Quantitative Intelligence Reimagined</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 Login", "🆕 Signup"])

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

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("📈 Quantara")
st.sidebar.markdown(f"**Logged in as:** {st.session_state.user.email}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

st.sidebar.markdown("---")

# ================= BANNER =================
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("assets/crypto_banner.png")

st.markdown(f"""
<div class="banner-container">
    <img src="data:image/png;base64,{img_base64}">
    <div class="banner-text">
        🌱 QUANTARA – Quantitative Intelligence Reimagined
    </div>
</div>
""", unsafe_allow_html=True)

# ================= DATA =================
DATASETS = {
    "BTC": "Data/BTC_historical_INR.csv",
    "ETH": "Data/ETH_historical_INR.csv",
    "SOL": "Data/SOL_historical_INR.csv",
    "USDC": "Data/USDC_historical_INR.csv",
    "USDT": "Data/USDT_historical_INR.csv"
}

@st.cache_data
def load_price(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

coin = st.sidebar.selectbox("Select Coin", list(DATASETS.keys()))
data = load_price(DATASETS[coin])

st.title(f"{coin} Forecast Dashboard")
st.markdown("---")

# ================= HISTORICAL =================
st.subheader("📊 Historical Price")
st.line_chart(data['Close'])

# ================= ARIMA =================
@st.cache_resource
def run_arima(series):
    return ARIMA(series, order=(2,1,2)).fit().forecast(30)

st.subheader("📈 ARIMA Forecast")
st.line_chart(run_arima(data['Close']))

# ================= LSTM =================
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
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    pred_input = scaled[-60:].reshape(1,60,1)
    preds = []
    for _ in range(30):
        p = model.predict(pred_input, verbose=0)[0][0]
        preds.append(p)
        pred_input = np.append(pred_input[:,1:,:], [[[p]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    future_dates = pd.date_range(series.index[-1], periods=30)
    return pd.DataFrame(preds, index=future_dates)

st.subheader("🤖 LSTM Deep Learning Forecast")
st.line_chart(run_lstm(data['Close']))

# ================= FOOTER =================
st.markdown("""
<div class="footer">
© 2026 Quantara - Built with ❤️ by Shravan & Rushik
</div>
""", unsafe_allow_html=True)
