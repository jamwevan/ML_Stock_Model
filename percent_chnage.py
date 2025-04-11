import pandas as pd
import os

def calculate_percent_change(ticker):
    pred_path = os.path.join("predictions", f"{ticker}_iterative_predictions.csv")

    if not os.path.exists(pred_path):
        print(f"Missing prediction file for {ticker}")
        return None

    df = pd.read_csv(pred_path)

    if "date" not in df.columns or "Close" not in df.columns:
        print(f"Missing required columns in {ticker}'s prediction file.")
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "Close"]).sort_values("date")

    if len(df) < 2:
        print(f"Not enough data for {ticker}")
        return None

    start_price = df.iloc[0]["Close"]
    end_price = df.iloc[-1]["Close"]
    percent_change = ((end_price - start_price) / start_price) * 100

    return percent_change

def main():
    with open("tickers.txt", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    print("Percent Change from Start to End of Prediction Window:")
    print("------------------------------------------------------")
    for ticker in tickers:
        change = calculate_percent_change(ticker)
        if change is not None:
            print(f"{ticker}: {change:+.2f}%")

if __name__ == "__main__":
    main()
