import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, classification_report
from transformers import pipeline
import os
from pathlib import Path

# 1. Configuration & Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'src' / 'models'/'sentiment_logic.pkl'
DATA_URL = "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/bitcoin_news.csv"

# --- NEW: ADD THE GLOBAL FUNCTION HERE SO PICKLE CAN FIND IT ---
def map_emotion(text):
    fear_signals = ["fear", "anxiety", "anger", "pessimism", "bearish", "crash", "slips", "pullback", "losses", "rout"]
    greed_signals = ["joy", "optimism", "excitement", "bullish", "moon", "hodl", "rebounds", "rally", "buy the dip"]
    text = str(text).lower()
    if any(sig in text for sig in fear_signals): return "Market Fear"
    elif any(sig in text for sig in greed_signals): return "Market Greed"
    return "Neutral"
# --------------------------------------------------------------

def test_model_accuracy():
    # 2. Load the Pickle
    print("📂 Loading pickled logic...")
    
    # Safety Check: Ensure file is not empty or missing
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        print("❌ Error: Pickle file is missing or empty. Run training script first.")
        return

    with open(MODEL_PATH, 'rb') as f:
        # Now this will work because 'map_emotion' is in the global scope
        package = pickle.load(f)
    
    sia = package['vader_analyzer']
    label_map = package['label_map']
    # You can also access the function from the package now:
    emotion_logic = package['emotion_logic'] 
    
    # 3. Load FinBERT
    print("🤖 Re-initializing FinBERT for testing...")
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    # 4. Fetch Test Data
    print("📡 Fetching latest data from GitHub...")
    df = pd.read_csv(DATA_URL).head(100)
    
    # 5. Define Hybrid Prediction Logic
    def predict_hybrid(text):
        v_score = sia.polarity_scores(str(text))['compound']
        f_res = finbert(str(text))[0]
        f_score = label_map.get(f_res['label'].lower(), 0)
        return (v_score + f_score) / 2

    # 6. Run Predictions
    print("⚖️ Calculating scores...")
    df['pred_score'] = df['title'].apply(predict_hybrid)
    
    def categorize(score):
        if score > 0.05: return 1
        if score < -0.05: return -1
        return 0

    df['pred_label'] = df['pred_score'].apply(categorize)
    # Adding psychology check to see it in action
    df['psychology'] = df['title'].apply(emotion_logic)

    # 7. Metrics
    if 'manual_label' in df.columns:
        mse = mean_squared_error(df['manual_label'], df['pred_score'])
        print(f"\n📈 Mean Squared Error: {mse:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(df['manual_label'], df['pred_label'], 
                                   target_names=['Negative', 'Neutral', 'Positive']))
    else:
        print("\n⚠️ Note: No 'manual_label' found. Predicted Sentiment Distribution:")
        print(df['pred_label'].value_counts(normalize=True))
        print("\n🧠 Market Psychology Preview (First 5):")
        print(df[['title', 'psychology']].head())

    print("\n✅ Test Complete.")

if __name__ == "__main__":
    test_model_accuracy()