import yfinance as yf
import pandas as pd
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import RAW_DATA_DIR


def download_crypto_data():
    # List of cryptocoin we will track
    tickers = ["USDT-USD", "BTC-USD", "ETH-USD", "USDC-USD", "SOL-USD", "FDUSD-USD", "XRP-USD", "BNB-USD", "USD136148-USD", "WETH-USD", "DOGE-USD", "WBNB-USD", "SUI20947-USD", "TRX-USD", "BREV-USD", "AXS-USD", "ADA-USD", "SOL16116-USD", "USDTZ30628-USD", "VBNB-USD", "ZEC-USD", "MIM-USD"]


    print(f"Downloading data for: {', '.join(tickers)}")

    for ticker in tickers:
        data = yf.download(ticker, period="5y", interval="1d")

        if not data.empty:
            file_name = f"{ticker.replace('-USD', '')}_historical.csv"
            data.to_csv(os.path.join(raw_data_path, file_name))
            print(f"Saved {ticker} to {file_name}")

        else:
            print(f"Failed to download {ticker}")
        
        
    current_prices = {}
    for ticker in tickers:
        coin = yf.Ticker(ticker)
        current_prices[ticker] = coin.history(period="1d")['Close'].iloc[-1]

    print("\n---- Current Market Prices ----")
    for coin, price in current_prices.items():
        print(f"{coin}: ${price:.2f}")
    
if __name__ == "__main__":
    download_crypto_data()