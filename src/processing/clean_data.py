import pandas as pd
import yfinance as yf
import os
import glob
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def clean_all_files():

    # Current USD/INR exchange rate
    usd_inr = yf.Ticker("INR=X").history(period="1d")['Close'].iloc[-1]

    raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))

    if not raw_files:
        print("No CSV files found in data/raw. Please run fetch_data.py first!")
        return
    
    for file_path in raw_files:
        file_name = os.path.basename(file_path)
        print(f"Cleaning {file_name}...")

        df = pd.read_csv(file_path, header = [0, 1])
        df.columns = df.columns.get_level_values(0)

        date_cols = df.columns[0]
        df = df.rename(columns={date_cols: 'Date'})

        df = df[df['Date'].str.contains(r'\d', na=False)]

        price_cols = ['Close', 'High', 'Low', 'Open']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col] * usd_inr

        new_name = file_name.replace(".csv", "_INR.csv")
        save_path = os.path.join(PROCESSED_DATA_DIR, new_name)

        df.to_csv(save_path, index=False)
        print(f"Processed and saved tp {save_path}")

if __name__ == "__main__":
    clean_all_files()
