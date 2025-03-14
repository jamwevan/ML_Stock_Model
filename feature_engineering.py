import pandas as pd
import os

# Function to load stock data from CSV

def load_stock_data(ticker, folder="stock_data"):
    """
    Load raw stock data from CSV files.
    """
    file_path = os.path.join(folder, f"{ticker}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    else:
        print(f"File not found: {file_path}")
        return None

# Function to read tickers from a text file
def read_tickers_from_file(filename="tickers.txt"):
    """
    Read stock tickers from a text file.
    """
    try:
        with open(filename, "r") as file:
            tickers = [line.strip() for line in file if line.strip()]
        return tickers
    except Exception as e:
        print(f"Error reading tickers from {filename}: {e}")
        return []

# Function to compute trend indicators such as moving averages and MACD
def compute_trend_indicators(df):
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    return df

# Function to compute momentum indicators such as RSI and momentum
def compute_momentum_indicators(df):
    df["RSI_14"] = 100 - (100 / (1 + df["Close"].diff().apply(lambda x: max(x, 0)).rolling(14).mean() /
                                  df["Close"].diff().apply(lambda x: abs(min(x, 0))).rolling(14).mean()))
    df["Momentum_10"] = df["Close"].diff(periods=10)
    return df

# Function to compute volatility indicators such as Bollinger Bands and ATR
def compute_volatility_indicators(df):
    df["Bollinger_Upper"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
    df["Bollinger_Lower"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()
    df["ATR_14"] = df["High"].rolling(14).max() - df["Low"].rolling(14).min()
    return df

# Function to compute volume-based indicators such as moving averages and OBV
def compute_volume_indicators(df):
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["OBV"] = (df["Volume"] * ((df["Close"].diff() > 0).astype(int) - (df["Close"].diff() < 0).astype(int))).cumsum()
    return df

# Function to compute lagged returns (previous closing prices)
def compute_lagged_returns(df):
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_5"] = df["Close"].shift(5)
    df["Lag_10"] = df["Close"].shift(10)
    return df

# Function to add all computed features to the dataset
def add_features(df):
    df = compute_trend_indicators(df)
    df = compute_momentum_indicators(df)
    df = compute_volatility_indicators(df)
    df = compute_volume_indicators(df)
    df = compute_lagged_returns(df)
    return df

# Function to save the processed dataset with computed features
def save_transformed_data(df, ticker, folder="processed_data"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    df.to_csv(os.path.join(folder, f"{ticker}_features.csv"))
    print(f"Processed data saved for {ticker}")

# Function to run feature engineering on all tickers
def process_stock_data(ticker):
    df = load_stock_data(ticker)
    if df is not None:
        df = add_features(df)
        df = df.dropna()  # Remove rows with NaN values
        save_transformed_data(df, ticker)

# Main execution logic
def main():
    tickers = read_tickers_from_file("tickers.txt")
    for ticker in tickers:
        print(f"Processing {ticker}...")
        process_stock_data(ticker)

if __name__ == "__main__":
    main()
