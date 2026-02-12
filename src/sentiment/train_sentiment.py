import pandas as pd
import pickle
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from pathlib import Path

# 1. Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_PATH = PROJECT_ROOT / 'src' / 'models'
# Using your new ground truth data
DATA_URL = "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/labeled_crypto_sentiment.csv"

def map_emotion(text):
    """Categorizes market sentiment based on crypto-specific signals."""
    fear_signals = ["fear", "anxiety", "anger", "pessimism", "bearish", "crash", "slips", "pullback", "losses", "rout", "dump", "panic"]
    greed_signals = ["joy", "optimism", "excitement", "bullish", "moon", "hodl", "rebounds", "rally", "buy the dip", "pumping", "ath"]
    text = str(text).lower()
    if any(sig in text for sig in fear_signals): return "Market Fear"
    elif any(sig in text for sig in greed_signals): return "Market Greed"
    return "Neutral"

def train_and_save_sentiment():
    print(f"📡 Loading data for bias analysis from {DATA_URL}...")
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    
    # Formula: total_samples / (n_classes * class_count)
    counts = df['manual_label'].value_counts()
    total = len(df)
    n_classes = len(counts)
    
    # These weights tell the model that 'Fear' is ~2.2x more important to get right 
    # than 'Greed' because it appears less frequently in the data.
    class_weights = {
        'Greed': round(total / (n_classes * counts.get('Greed', 1)), 4),
        'Fear': round(total / (n_classes * counts.get('Fear', 1)), 4),
        'Neutral': round(total / (n_classes * counts.get('Neutral', 1)), 4)
    }
    print(f"⚖️ Calculated Class Weights: {class_weights}")

    # 3. Initialize Analyzers
    sia = SentimentIntensityAnalyzer()
    
    # 4. Export as PKL
    sentiment_model_package = {
        'vader_analyzer': sia,
        'label_map': {"Greed": 1, "Fear": -1, "Neutral": 0},
        'emotion_logic': map_emotion,
        'class_weights': class_weights  # Saved for use in test_sentiment metrics
    }

    os.makedirs(MODELS_PATH, exist_ok=True)
    with open(MODELS_PATH / 'sentiment_logic.pkl', 'wb') as f:
        pickle.dump(sentiment_model_package, f)
    
    print(f"✅ sentiment_logic.pkl updated at {MODELS_PATH}")

if __name__ == "__main__":
    train_and_save_sentiment()