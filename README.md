# 🚀 Crypto AI Forecast Dashboard

An end-to-end Machine Learning and NLP platform designed to analyze market sentiment and forecast cryptocurrency price trends. This project integrates statistical models, deep learning, and real-time news analysis into a unified Streamlit interface.



## 🌟 Features

- **Hybrid Sentiment Engine**: A dual-layered approach using **VADER** and **FinBERT** to decode market psychology from live news headlines.
- **Multimodal Forecasting**:
  - **LSTM (Deep Learning)**: Long Short-Term Memory networks for complex pattern recognition.
  - **FB Prophet**: Robust time-series forecasting focused on crypto-market seasonality.
  - **ARIMA**: Classic statistical modeling for linear trend analysis.
- **Live Market Marquee**: A non-blocking, asynchronous "marquee" built with Streamlit Fragments that streams live sentiment across Bitcoin, Ethereum, and Solana.
- **Interactive UI**: A fully responsive dashboard with real-time clock synchronization and isolated component rerendering for high performance.
- **Secure Authentication**: Integrated with **Supabase Auth** for personalized user sessions and secure data access.



## 🏗️ Project Structure

The project follows a modular architecture to ensure scalability and clean code separation:

```text
crypto-forecaster/
├── app.py                  # Main Dashboard (Streamlit)
├── src/
│   ├── sentiment/
│   │   ├── analyzer.py     # Hybrid sentiment processing logic
│   │   └── utils.py        # Shared categorization & map functions
│   ├── models/
│   │   └── sentiment_logic.pkl  # Trained transformer/VADER package
│   └── processing/
│       └── data_loader.py  # News & Price data ingestion
├── Output/
│   ├── Arima/              # Exported forecast plots
│   ├── FB_Prophet/
│   └── LSTM/
└── requirements.txt        # Project dependencies
```
## 📊 Model Performance

| Component | Metric | Score |
| --- | --- | --- |
| **Sentiment Analysis** | Macro F1-Score | **0.80** |
| **Fear Detection** | Recall | **0.94** |
| **Price Forecasting** | Horizon | **30 Days** |

## 🚀 Getting Started

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/crypto-forecaster.git
cd crypto-forecaster

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```



3. **Launch the Dashboard:**
```bash
streamlit run app.py

```



## 🛠️ Tech Stack

* **NLP**: Hugging Face Transformers, NLTK, VADER
* **ML/DL**: TensorFlow, Keras, Scikit-Learn, FBProphet, Statsmodels
* **Frontend**: Streamlit
* **Visualization**: Matplotlib, Plotly

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👥 Contributors

This project was a collaborative effort by the following developers:

- **[Your Name]** ([@Rushik](https://github.com/Kokate-Rushik)): **Project Lead & Lead Developer**. Architected the Hybrid Sentiment Engine, implemented the fragmented Streamlit UI, and refined the core data processing pipelines.
- **Shravan Shidruk** ([@Shravan](https://github.com/shravanshidruk16)): **ML Engineer & Dashboard Developer**. Developed the Analytics and Dashboard components. Specifically responsible for developing and training the **ARIMA**, **FB_Prophet**, and **LSTM** models for the following tickers:
  - **BTC** (Bitcoin)
  - **ETH** (Ethereum)
  - **SOL** (Solana)
  - **USDC** (USD Coin)
  - **USDT** (Tether)
- **Stefie George** ([@Stefie](https://github.com/stefiegeorge-ai)): **Market Sentiment Model Developer**. Designed the initial sentiment analysis prototype and established the text processing foundations in the early project phases.
