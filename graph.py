import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(ticker):
    pred_path = os.path.join("predictions", f"{ticker}_iterative_predictions.csv")
    actual_path = os.path.join("stock_data", f"{ticker}.csv")

    if not os.path.exists(pred_path) or not os.path.exists(actual_path):
        print(f"Missing file(s) for {ticker}")
        return

    pred_df = pd.read_csv(pred_path)
    actual_df = pd.read_csv(actual_path)

    if "date" not in pred_df.columns or "Close" not in pred_df.columns:
        print(f"Missing 'date' or 'Close' in prediction file for {ticker}")
        return
    if "date" not in actual_df.columns or "Close" not in actual_df.columns:
        print(f"Missing 'date' or 'Close' in actual file for {ticker}")
        return

    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")
    actual_df["date"] = pd.to_datetime(actual_df["date"], errors="coerce")

    pred_df = pred_df.dropna(subset=["date"]).sort_values("date").tail(20)
    actual_df = actual_df.dropna(subset=["date"]).sort_values("date")

    # Merge and align
    merged = pd.merge(
        pred_df[["date", "Close"]],
        actual_df[["date", "Close"]],
        on="date",
        how="left",
        suffixes=("_predicted", "_actual")
    )

    if merged["Close_actual"].isnull().all():
        print(f"No matching actual data found for {ticker}'s prediction dates.")
        return

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(merged["date"], merged["Close_predicted"], label="Predicted Close", marker="o")
    plt.plot(merged["date"], merged["Close_actual"], label="Actual Close", marker="x", linestyle="--")
    plt.title(f"{ticker} — Actual vs. Predicted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    prediction_dir = "predictions"
    files = os.listdir(prediction_dir)

    # Loop through all prediction files
    for file in files:
        if file.endswith("_iterative_predictions.csv"):
            ticker = file.split("_")[0]
            print(f"Plotting {ticker}...")
            plot_comparison(ticker)

if __name__ == "__main__":
    main()
