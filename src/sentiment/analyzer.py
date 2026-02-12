import pandas as pd
import pickle
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from transformers import pipeline

# 1. NEW: Import shared logic
from utils import map_emotion, categorize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "sentiment_logic.pkl"
CACHE_FILE = PROJECT_ROOT / "src" / "sentiment" / "sentiment_cache.json"

def load_heavy_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def get_market_sentiment_stats(symbol, csv_url):
    now = datetime.now()
    
    # 2. Load existing cache
    cache_data = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
    
    symbol_key = f"{symbol.lower()}_analysis"
    last_update_str = cache_data.get("last_update")
    
    # 3. Cache Logic (6-hour window)
    if last_update_str:
        last_update = datetime.strptime(last_update_str, "%Y-%m-%d T %H:%M:%S")
        if (now - last_update) < timedelta(hours=6) and symbol_key in cache_data:
            print(f"⚡ Returning cached data for {symbol}...")
            return cache_data[symbol_key]

    print(f"🔄 Running Full Hybrid Analysis for {symbol}...")
    
    package = load_heavy_model()
    if not package:
        return {"error": "Model file not found"}

    sia = package['vader_analyzer']
    # Use the logic from the PKL package
    emotion_logic = package['emotion_logic'] 
    
    # Initialize FinBERT
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    # 4. Process Data
    current_df = pd.read_csv(csv_url)
    if 'title' not in current_df.columns:
        return {"error": f"Column 'title' missing in {symbol} data"}

    # Hybrid Scoring Logic
    def predict_hybrid(text):
        v_score = sia.polarity_scores(str(text))['compound']
        f_res = finbert(str(text))[0]
        f_map = {"positive": 1, "negative": -1, "neutral": 0}
        f_score = f_map.get(f_res['label'].lower(), 0)
        return (v_score + f_score) / 2

    # Step A: Calculate Hybrid Score
    current_df['hybrid_score'] = current_df['title'].apply(predict_hybrid)
    
    # Step B: Categorize using refined thresholds (0.12 / -0.05)
    current_df['final_label'] = current_df['hybrid_score'].apply(categorize)
    
    # Step C: Fallback to keyword logic if neutral (Optional refinement)
    # This combines both methods for maximum accuracy
    name_map = {1: "Market Greed", -1: "Market Fear", 0: "Neutral"}
    current_df['psychology'] = current_df['final_label'].map(name_map)

    # 5. Calculate Stats
    stats = current_df['psychology'].value_counts(normalize=True) * 100
    
    analysis_result = {
        "market_fear_percent": float(round(stats.get("Market Fear", 0), 2)),
        "market_greed_percent": float(round(stats.get("Market Greed", 0), 2)),
        "market_neutral_percent": float(round(stats.get("Neutral", 0), 2))
    }

    # 6. Update Cache
    cache_data["last_update"] = now.strftime("%Y-%m-%d T %H:%M:%S")
    cache_data[symbol_key] = analysis_result
    
    os.makedirs(CACHE_FILE.parent, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=4)

    return analysis_result

if __name__ == "__main__":
    COINS = {
        "BTC": "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/bitcoin_news.csv",
        "ETH": "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/ethereum_news.csv",
        "SOL": "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/solana_news.csv",
        "USDT": "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/tether_news.csv",
        "USDC": "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/usdc_news.csv"
    }
    
    for symbol, url in COINS.items():
        print(f"\n--- {symbol} ---")
        print(get_market_sentiment_stats(symbol, url))