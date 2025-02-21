import os
import time
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY"

def read_tickers_from_file(filename):
    """
    Read stock tickers from a text file.

    Parameters:
        filename (str): Path to the file containing stock tickers.

    Returns:
        list: A list of stock tickers.
    """
    try:
        with open(filename, "r") as file:
            tickers = [line.strip() for line in file if line.strip()]  # Remove empty lines
        return tickers
    except Exception as e:
        print(f"Error reading tickers from {filename}: {e}")
        return []

def fetch_stock_data(tickers, output_folder="stock_data"):
    """
    Fetch historical stock data for the given tickers using Alpha Vantage and save them as CSV files.

    Parameters:
        tickers (list): List of stock tickers to fetch data for.
        output_folder (str): Folder to save the CSV files. Default is 'stock_data'.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            # Fetch free daily stock data
            data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")
            if data.empty:
                print(f"Warning: No data found for {ticker}. Skipping.")
                continue
            # Rename columns for better readability
            data.columns = ["Open", "High", "Low", "Close", "Volume"]
            # Save to CSV
            output_file = os.path.join(output_folder, f"{ticker}.csv")
            data.to_csv(output_file)
            print(f"Data for {ticker} saved to {output_file}")
            # Add a delay to avoid hitting rate limits
            time.sleep(15)  # Free plan allows only 5 requests per minute
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

if __name__ == "__main__":
    ticker_file = "tickers.txt"
    tickers = read_tickers_from_file(ticker_file)
    if tickers:
        fetch_stock_data(tickers)
    else:
        print("No valid tickers found in the file.")
