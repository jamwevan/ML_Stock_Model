# Import the pandas library for data manipulation
import pandas as pd
# Import train_test_split to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import RandomForestRegressor for regression modeling using an ensemble of trees
from sklearn.ensemble import RandomForestRegressor
# Import evaluation metrics: Mean Squared Error, Mean Absolute Error, and R² score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Import numpy for numerical operations (e.g., square root)
import numpy as np

# Define a function that trains a regression model for a given ticker
def train_model_for_ticker(ticker, test_size=0.2, random_state=42):
    """
    Loads the CSV for a single ticker, creates a 'target' column for the next day's Close,
    drops the date column if present, and trains a RandomForestRegressor using all numeric
    columns (except 'target') as features. Returns the trained model and regression metrics.
    """
    # Construct the CSV file path for the given ticker
    csv_path = f'processed_data/{ticker}_features.csv'
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    # Create a new column 'target' which is tomorrow's closing price by shifting 'Close' up by one row
    df['target'] = df['Close'].shift(-1)
    # Drop the last row that now contains a NaN value for 'target' (because there is no next day)
    df.dropna(inplace=True)
    # Check if there is a column named 'date'; if so, drop it
    if 'date' in df.columns:
        # Drop the 'date' column from the DataFrame
        df.drop(columns=['date'], inplace=True)
    # If there is no 'date' column, check for 'Date' and drop it if it exists
    elif 'Date' in df.columns:
        # Drop the 'Date' column from the DataFrame
        df.drop(columns=['Date'], inplace=True)
    # Select all columns with numeric data types, then drop the 'target' column from these features
    X = df.select_dtypes(include=[np.number]).drop(columns=['target'])
    # Set the target variable (y) to the 'target' column which contains the next day's Close price
    y = df['target']
    # Split the data into training and testing sets using the specified test size and random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Initialize a RandomForestRegressor with 100 trees and the provided random state for reproducibility
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    # Train the RandomForestRegressor model on the training data
    model.fit(X_train, y_train)
    # Use the trained model to predict the target variable on the test data
    y_pred = model.predict(X_test)
    # Calculate the Mean Squared Error between the actual and predicted values
    mse = mean_squared_error(y_test, y_pred)
    # Compute the Root Mean Squared Error by taking the square root of the MSE
    rmse = np.sqrt(mse)
    # Calculate the Mean Absolute Error between the actual and predicted values
    mae = mean_absolute_error(y_test, y_pred)
    # Compute the R² score to determine the proportion of variance explained by the model
    r2 = r2_score(y_test, y_pred)
    # Return the trained model along with all computed regression metrics
    return model, mse, rmse, mae, r2

# Define the main function that processes multiple tickers
def main():
    # Open the 'tickers.txt' file and read in each ticker symbol, stripping whitespace
    with open('tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    # Create an empty dictionary to store the model and metrics for each ticker
    results = {}
    # Iterate over each ticker in the list
    for ticker in tickers:
        # Print a message indicating the start of training for the current ticker
        print(f"Training model for {ticker}...")
        # Train the model for the current ticker and retrieve regression metrics
        model, mse, rmse, mae, r2 = train_model_for_ticker(ticker)
        # Store the resulting model and metrics in the results dictionary, keyed by the ticker symbol
        results[ticker] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        # Print the evaluation results for the current ticker in a formatted manner
        print(f"--- {ticker} Results ---")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R^2:  {r2:.4f}")
        print("--------------------------------------------------\n")
# TODO: Further analysis with the 'results' dictionary 
# Standard boilerplate to execute main() when the script is run directly
if __name__ == '__main__':
    main()
