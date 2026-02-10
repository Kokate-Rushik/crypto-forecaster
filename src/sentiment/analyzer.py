import pandas as pd
import pickle
from pathlib import Path
from transformers import pipeline

# 1. Setup paths relative to this file
# Assumes structure: root/src/sentiment/analyzer.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "sentiment_logic.pkl"

def map_emotion(text):
    fear_signals = ["fear", "anxiety", "anger", "pessimism", "bearish", "crash", "slips", "pullback", "losses", "rout"]
    greed_signals = ["joy", "optimism", "excitement", "bullish", "moon", "hodl", "rebounds", "rally", "buy the dip"]
    text = str(text).lower()
    if any(sig in text for sig in fear_signals): return "Market Fear"
    elif any(sig in text for sig in greed_signals): return "Market Greed"
    return "Neutral"

def get_market_sentiment_stats(btc_news_df):
    """
    Analyzes a DataFrame of news and returns the percentage 
    of Fear, Greed, and Neutral sentiment.
    """
    # Safety Check: Load the pickled logic
    if not MODEL_PATH.exists():
        return {"error": "Model file not found"}

    with open(MODEL_PATH, 'rb') as f:
        package = pickle.load(f)
    
    sia = package['vader_analyzer']
    emotion_logic = package['emotion_logic']
    label_map = package['label_map']

    # Initialize FinBERT (Streamlit will cache this if using @st.cache_resource)
    # For now, we load it directly
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def predict_sentiment(text):
        v_score = sia.polarity_scores(str(text))['compound']
        f_res = finbert(str(text))[0]
        f_score = label_map.get(f_res['label'].lower(), 0)
        return (v_score + f_score) / 2

    # Perform analysis on the 'title' column
    # Based on your CSV: [date, sources, title, link]
    btc_news_df['psychology'] = btc_news_df['title'].apply(emotion_logic)
    
    # Calculate percentages
    total = len(btc_news_df)
    stats = btc_news_df['psychology'].value_counts(normalize=True) * 100
    
    return {
        "Fear": float(round(stats.get("Market Fear", 0), 2)),
        "Greed": float(round(stats.get("Market Greed", 0), 2)),
        "Neutral": float(round(stats.get("Neutral", 0), 2))
    }

if __name__=="__main__":
    URL = "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/bitcoin_news.csv"
    df = pd.read_csv(URL)
    percentsd = get_market_sentiment_stats(df)
    print(percentsd)

