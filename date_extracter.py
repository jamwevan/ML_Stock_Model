import os
import pandas as pd

# Folder containing your processed feature CSVs
data_folder = "processed_data"
output_file = "ticker_date_ranges.txt"
tickers_file = "tickers.txt"

# Read tickers.txt to preserve order
with open(tickers_file, "r") as f:
    tickers = [line.strip() for line in f if line.strip()]

with open(output_file, "w") as out:
    for ticker in tickers:
        filename = f"{ticker}_features.csv"
        path = os.path.join(data_folder, filename)
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            start_date = df["date"].min().strftime("%Y-%m-%d")
            end_date = df["date"].max().strftime("%Y-%m-%d")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            start_date, end_date = "ERROR", "ERROR"
        out.write(f"{ticker},{start_date},{end_date}\n")
