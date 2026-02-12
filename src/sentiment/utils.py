def map_emotion(text):
    """Categorizes market sentiment based on keyword signals."""
    fear_signals = ["fear", "anxiety", "anger", "pessimism", "bearish", "crash", "slips", "pullback", "losses", "rout", "dump", "panic"]
    greed_signals = ["joy", "optimism", "excitement", "bullish", "moon", "hodl", "rebounds", "rally", "buy the dip", "pumping", "ath"]
    text = str(text).lower()
    if any(sig in text for sig in fear_signals):
        return "Market Fear"
    elif any(sig in text for sig in greed_signals):
        return "Market Greed"
    return "Neutral"

def categorize(score):
    if score > 0.05: return 1
    if score < -0.05: return -1
    return 0