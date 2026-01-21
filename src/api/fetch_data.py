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

    # getting all history in one request
    all_history = yf.download(tickers, period="5y", interval="1d", group_by="ticker")

    for ticker in tickers:
        try:
            ticker_data = all_history[ticker].dropna()
            
            if not ticker_data.empty:
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    ticker_data.columns = ticker_data.columns.get_level_values(0)
                
                clean_name = ticker.replace('-USD', '')
                file_name = f"{clean_name}_historical.csv"
                save_path = os.path.join(RAW_DATA_DIR, file_name)
                
                ticker_data.to_csv(save_path)
                print(f"Saved {ticker}")
            
            # Politeness delay (prevents IP flags)
            time.sleep(1) 
            
        except Exception as e:
            print(f"Failed to save {ticker}: {e}")
        
        
        
    print("\n Fetching latest market prices...")
    
    for ticker in tickers:
        try:
           
            clean_name = ticker.replace('-USD', '')
            file_path = os.path.join(RAW_DATA_DIR, f"{clean_name}_historical.csv")
            
            temp_df = pd.read_csv(file_path)
            
            last_price = temp_df['Close'].iloc[-1]
            
            print(f"{ticker}: ${last_price:.2f}")
        except Exception:
            print(f"{ticker}: Data not found in local CSV")
    
if __name__ == "__main__":
    download_crypto_data()