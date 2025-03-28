import pandas as pd
import numpy as np
import os
import calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

US_MARKET_HOLIDAYS_2025 = pd.to_datetime([
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18',
    '2025-05-26', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
])

def adjust_date(date_str):
    try:
        return pd.to_datetime(date_str)
    except Exception:
        parts = date_str.split('-')
        if len(parts) != 3:
            return None
        try:
            year, month, day = map(int, parts)
            day = min(day, calendar.monthrange(year, month)[1])
            return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
        except Exception:
            return None

def get_trading_days(start_date, end_date):
    start_dt = adjust_date(start_date)
    end_dt = adjust_date(end_date)
    if start_dt is None or end_dt is None or start_dt > end_dt:
        return pd.DatetimeIndex([])
    all_bdays = pd.date_range(start=start_dt, end=end_dt, freq='B')
    return pd.DatetimeIndex([d for d in all_bdays if d not in US_MARKET_HOLIDAYS_2025])

def train_model_for_ticker(ticker, test_size=0.2, random_state=42):
    csv_path = f'processed_data/{ticker}_features.csv'
    df = pd.read_csv(csv_path)
    date_col = 'date' if 'date' in df.columns else 'Date' if 'Date' in df.columns else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    df_for_training = df.copy()
    if date_col:
        df_for_training.drop(columns=[date_col], inplace=True)
    X = df_for_training.select_dtypes(include=[np.number]).drop(columns=['target'])
    y = df_for_training['target']
    feature_columns = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, df, mean_squared_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred), feature_columns

def simulate_future_ohlcv(last_row):
    close = last_row['Close']
    open_price = close + np.random.normal(0, 0.2)
    high = max(open_price, close) + abs(np.random.normal(0.3, 0.1))
    low = min(open_price, close) - abs(np.random.normal(0.3, 0.1))
    volume = last_row['Volume'] * np.random.uniform(0.95, 1.05)
    return open_price, high, low, volume

def predict_iteratively(ticker, model, df, start_date, end_date, recalc_indicators_fn, feature_columns):
    if 'date' not in df.columns:
        print(f"[{ticker}] No date column found; can't do iterative predictions.")
        return
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    last_known_date = df['date'].max()
    start_dt = adjust_date(start_date)
    end_dt = adjust_date(end_date)
    if start_dt is None or end_dt is None or last_known_date >= end_dt:
        print(f"[{ticker}] No future days to predict between {start_date} and {end_date}.")
        return
    future_days = pd.date_range(start=max(start_dt, last_known_date + pd.Timedelta(days=1)), end=end_dt, freq='B')
    rolling_window = df.tail(20).copy()
    if 'target' in rolling_window.columns:
        rolling_window.drop(columns=['target'], inplace=True)
    predictions = []
    for day in future_days:
        last_row = rolling_window.iloc[-1].copy()
        open_price, high, low, volume = simulate_future_ohlcv(last_row)
        new_row = last_row.copy()
        new_row['date'] = day
        new_row['Open'] = open_price
        new_row['High'] = high
        new_row['Low'] = low
        new_row['Volume'] = volume
        new_row['Close'] = last_row['Close']
        rolling_window = pd.concat([rolling_window, new_row.to_frame().T], ignore_index=True)
        rolling_window = recalc_indicators_fn(rolling_window)
        latest_row = rolling_window.iloc[-1]
        X_input = pd.DataFrame([latest_row[feature_columns].values], columns=feature_columns)
        predicted_close = model.predict(X_input)[0]
        rolling_window.at[rolling_window.index[-1], 'Close'] = predicted_close
        predictions.append(rolling_window.iloc[-1].copy())
    output_df = pd.DataFrame(predictions)
    os.makedirs('predictions', exist_ok=True)
    output_csv = f'predictions/{ticker}_iterative_predictions.csv'
    output_df.to_csv(output_csv, index=False)
    print(f"[{ticker}] Iterative predictions saved to {output_csv}")

def main():
    future_start_date = '2025-02-29'
    future_end_date = '2025-03-27'
    from feature_engineering import recalc_indicators
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    for ticker in tickers:
        print(f"Training model for {ticker}...")
        model, df, mse, rmse, mae, r2, feature_columns = train_model_for_ticker(ticker)
        print(f"--- {ticker} Results ---\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR^2: {r2:.4f}\n")
        predict_iteratively(ticker, model, df, future_start_date, future_end_date, recalc_indicators, feature_columns)

if __name__ == '__main__':
    main()
