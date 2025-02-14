import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(tickers, start_date, end_date, output_folder="stock_data"):
    """
    Fetch historical stock data for the given tickers and save them as CSV files.

    Parameters:
        tickers (list): List of stock tickers to fetch data for.
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.
        output_folder (str): Folder to save the CSV files. Default is 'stock_data'.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            # Fetch the data using yfinance
            data = yf.download(ticker, start=start_date, end=end_date)

            # Save the data to a CSV file
            output_file = os.path.join(output_folder, f"{ticker}.csv")
            data.to_csv(output_file)
            print(f"Data for {ticker} saved to {output_file}")
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")

if __name__ == "__main__":
    # Define the stock tickers to fetch data for
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # Define the date range
    start_date = "2015-01-01"
    end_date = "2025-01-01"

    # Call the function to fetch and save the data
    fetch_stock_data(tickers, start_date, end_date)

