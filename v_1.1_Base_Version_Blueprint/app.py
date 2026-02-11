import streamlit as st
import os
from datetime import datetime

st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")

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

coin_folder = os.path.join(BASE_DIR, model_choice, coin_choice)

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
