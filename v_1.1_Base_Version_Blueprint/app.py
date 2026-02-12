import streamlit as st
import os
from datetime import datetime

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")

st.set_page_config(layout="wide", page_title="Crypto AI Forecast")

marquee_spot = st.empty()

@st.fragment(run_every="1h")  # Heavy lifting happens once an hour
def render_marquee():
    # Initial placeholder while loading
    marquee_spot.markdown(
        '<marquee style="color: #666;">🔄 Syncing market psychology across exchanges...</marquee>', 
        unsafe_allow_html=True
    )
    
    COINS = {
        "Bitcoin": "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/bitcoin_news.csv",
        "Ethereum": "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/ethereum_news.csv",
        "Solana": "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/solana_news.csv"
    }
    
    ticker_parts = []
    for name, url in COINS.items():
        # This call is heavy (FinBERT), but fragment keeps it isolated
        stats = get_market_sentiment_stats(name, url)
        ticker_parts.append(
            f"{name}: Greed {stats['market_greed_percent']}% | "
            f"Fear {stats['market_fear_percent']}% | "
            f"Neutral {stats['market_neutral_percent']}%"
        )
    
    marquee_text = " || ".join(ticker_parts)
    marquee_spot.markdown(
        f'<marquee style="color: #007BFF; font-weight: bold; font-family: sans-serif;">'
        f'● {marquee_text} ●'
        f'</marquee>', 
        unsafe_allow_html=True
    )


BASE_DIR = "."   # ✅ FIXED

st.title("📊 Real-Time Crypto Forecasting Dashboard")
st.markdown("ARIMA • Facebook Prophet • LSTM Deep Learning")

st.sidebar.header("🎛 Control Panel")

model_choice = st.sidebar.selectbox("Select Model", ["Arima", "Fb_Prophet", "LSTM"])
coin_choice = st.sidebar.selectbox("Select Coin", ["BTC", "ETH", "SOL", "USDC", "USDT"])

info = {
    "Arima": "Statistical model for trend & seasonality.",
    "Fb_Prophet": "Meta's forecasting tool for time series.",
    "LSTM": "Deep learning model for complex patterns."
}

st.info(info[model_choice])

col1, col2 = st.columns(2)
col1.metric("Last Updated", datetime.now().strftime("%d %b %Y %H:%M:%S"))
col2.metric("Data Status", "Live Simulation")

st.markdown("---")

coin_folder = os.path.join(BASE_DIR, '\Output', model_choice, coin_choice)

if not os.path.exists(coin_folder):
    st.error(f"Folder not found: {coin_folder}")
else:
    images = [f for f in os.listdir(coin_folder) if f.endswith(".png")]

    if not images:
        st.warning("No images found.")
    else:
        chart_choice = st.selectbox("Select Analysis Chart", images)
        image_path = os.path.join(coin_folder, chart_choice)

        st.markdown(f"## 📈 {chart_choice.replace('_',' ').replace('.png','')}")
        st.image(image_path, use_container_width=True)

st.markdown("---")
st.subheader("📊 Model Strengths")

st.table({
    "Model": ["ARIMA", "Prophet", "LSTM"],
    "Best For": [
        "Short-term statistical patterns",
        "Seasonal financial data",
        "Non-linear deep learning patterns"
    ]
})

st.success("🚀 Multi-model crypto forecasting dashboard running in real-time simulation.")

render_marquee()
