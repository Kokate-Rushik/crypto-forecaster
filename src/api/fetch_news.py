import pandas as pd

def get_btc_news():
    url = "https://raw.githubusercontent.com/Kokate-Rushik/news-automate/main/news/bitcoin_news.csv"
    df = pd.read_csv(url)
    return df
