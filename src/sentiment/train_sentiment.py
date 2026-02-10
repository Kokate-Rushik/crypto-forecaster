import pandas as pd
import pickle
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_PATH = PROJECT_ROOT / 'src' / 'models'

# 1. Global Configuration
URL = "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/bitcoin_news.csv"

# 2. Global Helper Functions (Moved to top level for successful pickling)
def map_emotion(text):
    """Categorizes market sentiment based on keyword signals."""
    fear_signals = ["fear", "anxiety", "anger", "pessimism", "bearish", "crash", "slips", "pullback", "losses", "rout"]
    greed_signals = ["joy", "optimism", "excitement", "bullish", "moon", "hodl", "rebounds", "rally", "buy the dip"]
    text = str(text).lower()
    if any(sig in text for sig in fear_signals):
        return "Market Fear"
    elif any(sig in text for sig in greed_signals):
        return "Market Greed"
    return "Neutral"

def train_and_save_sentiment():
    print("📡 Loading news data for training...")
    try:
        data = pd.read_csv(URL)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Initialize Analyzers
    sia = SentimentIntensityAnalyzer()
    
    # FinBERT for financial nuance
    print("Loading FinBERT (this may take a minute)...")
    # Note: finbert pipeline remains inside for training setup but won't be pickled
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    # 4. Export as PKL
    # We save the global function references and instance data
    sentiment_model_package = {
        'vader_analyzer': sia,
        'label_map': {"positive": 1, "negative": -1, "neutral": 0},
        'emotion_logic': map_emotion  # Reference to the now-global function
    }

    os.makedirs(MODELS_PATH, exist_ok=True)
    # Always open files in binary mode ('wb' or 'rb') when working with Pickle
    with open(MODELS_PATH / 'sentiment_logic.pkl', 'wb') as f:
        # Using the latest protocol (HIGHEST_PROTOCOL) improves performance
        pickle.dump(sentiment_model_package, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("✅ Sentiment logic and VADER instance saved to models/sentiment_logic.pkl")

if __name__ == "__main__":
    train_and_save_sentiment()