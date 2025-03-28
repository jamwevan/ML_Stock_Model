import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_predictions(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if 'date' not in df.columns or 'Close' not in df.columns:
        print("CSV must contain 'date' and 'Close' columns.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values("date")

    # Only plot future prediction range
    pred_df = df.tail(20)

    plt.figure(figsize=(10, 5))
    plt.plot(pred_df["date"], pred_df["Close"], marker="o", linestyle="-")
    plt.title("Predicted Close Prices (Iterative Forecast)")
    plt.xlabel("Date")
    plt.ylabel("Predicted Close")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_path = "predictions/MMM_iterative_predictions.csv"
    plot_predictions(csv_path)
