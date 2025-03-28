import os
import time
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY"

def read_tickers_from_file(filename):
    try:
        with open(filename, "r") as file:
            tickers = [line.strip() for line in file if line.strip()]
        return tickers
    except Exception as e:
        print(f"Error reading tickers from {filename}: {e}")
        return []

def fetch_stock_data(tickers, output_folder="stock_data"):
    # EMPTY THE STOCK DATA FOLDER FIRST
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_folder)

    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")
            if data.empty:
                print(f"Warning: No data found for {ticker}. Skipping.")
                continue
            data.columns = ["Open", "High", "Low", "Close", "Volume"]
            output_file = os.path.join(output_folder, f"{ticker}.csv")
            data.to_csv(output_file)
            print(f"Data for {ticker} saved to {output_file}")
            time.sleep(15)  # Alpha Vantage free tier rate limit
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

if __name__ == "__main__":
    ticker_file = "tickers.txt"
    tickers = read_tickers_from_file(ticker_file)
    if tickers:
        fetch_stock_data(tickers)
    else:
        print("No valid tickers found in the file.")
