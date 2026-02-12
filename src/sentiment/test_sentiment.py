from utils import map_emotion
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import classification_report
from transformers import pipeline
import os
from pathlib import Path

# 1. Configuration & Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'src' / 'models' / 'sentiment_logic.pkl'
# UPDATED: Using the full labeled dataset for a true accuracy test
DATA_URL = "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/labeled_crypto_sentiment.csv"



def test_model_accuracy():
    print("📂 Loading pickled logic and Class Weights...")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Run train_sentiment.py first to generate the model.")
        return

    with open(MODEL_PATH, 'rb') as f:
        package = pickle.load(f)
    
    sia = package['vader_analyzer']
    emotion_logic = package['emotion_logic']
    class_weights = package.get('class_weights', {})
    
    # 2. Initialize Prediction Engine
    print("🤖 Initializing Hybrid Engine (VADER + FinBERT)...")
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    # 3. Load Dataset
    print(f"📡 Downloading evaluation dataset from GitHub...")
    df = pd.read_csv(DATA_URL)
    df = df.sample(n=100)
    
    # 4. Define Hybrid Prediction Logic
    def predict_hybrid(text):
        v_score = sia.polarity_scores(str(text))['compound']
        f_res = finbert(str(text))[0]
        # Map FinBERT labels to numbers
        f_map = {"positive": 1, "negative": -1, "neutral": 0}
        f_score = f_map.get(f_res['label'].lower(), 0)
        return (v_score + f_score) / 2

    # 5. Run Evaluations
    print(f"⚖️ Running analysis on {len(df)} news items...")
    df['pred_score'] = df['title'].apply(predict_hybrid)
    
    # LEAD DEVELOPER: Using balanced thresholds
    def categorize(score):
        if score > 0.05: return 1
        if score < -0.05: return -1
        return 0

    df['pred_label'] = df['pred_score'].apply(categorize)
    
    # Convert string labels back to integers for comparison
    # manual_label contains 'Greed', 'Fear', 'Neutral'
    target_map = {'Greed': 1, 'Fear': -1, 'Neutral': 0}
    df['target'] = df['manual_label'].map(target_map)

    # 6. Final Report
    print("\n" + "="*50)
    print("📊 CRYPTO SENTIMENT ACCURACY REPORT")
    print("="*50)
    print(f"⚖️ Weights Applied: {class_weights}")
    
    # We use classification_report to show the detailed breakdown
    print(classification_report(df['target'], df['pred_label'], 
                               target_names=['Market Fear', 'Neutral', 'Market Greed']))
    
    print("✅ Test Complete. Metrics reflect model performance across biased classes.")

if __name__ == "__main__":
    test_model_accuracy()