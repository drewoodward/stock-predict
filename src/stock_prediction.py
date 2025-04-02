# stock_predictions.py

import os
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from XGBoost import XGBClassifier

def load_sp500_data():
    """Load SP500 data from CSV or download via yfinance."""
    if os.path.exists("sp500.csv"):
        sp500 = pd.read_csv("sp500.csv", index_col=0)
    else:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        sp500.to_csv("sp500.csv")
    sp500.index = pd.to_datetime(sp500.index)
    return sp500

def preprocess_data(sp500):
    """Clean and preprocess the SP500 DataFrame."""
    # Remove columns that are not needed
    for col in ["Dividends", "Stock Splits"]:
        if col in sp500.columns:
            del sp500[col]
    
    # Create the 'Tomorrow' and 'Target' columns
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    
    # Filter for data after January 1, 1990 and drop any rows with missing values
    sp500 = sp500.loc["1990-01-01":].dropna().copy()
    return sp500

def train_rf_model(sp500, predictors=["Close", "Volume", "Open", "High", "Low"]):
    """Train a RandomForestClassifier on the SP500 data."""
    # Use all but the last 100 days for training
    train = sp500.iloc[:-100]
    test = sp500.iloc[-100:]
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model.fit(train[predictors], train["Target"])
    
    # Optionally evaluate the model
    preds = model.predict(test[predictors])
    precision = precision_score(test["Target"], preds)
    print("Random Forest Model Precision:", precision)
    return model

def add_additional_predictors(sp500, horizons=[2, 5, 60, 250, 1000]):
    """Add extra predictors based on rolling averages and trends."""
    new_predictors = []
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors += [ratio_column, trend_column]
    
    # Drop rows with NaNs from the new predictors (except for 'Tomorrow')
    cols_to_check = [col for col in sp500.columns if col != "Tomorrow"]
    sp500 = sp500.dropna(subset=cols_to_check)
    return sp500, new_predictors

def train_xgb_model(sp500, predictors):
    """Train an XGBoost model on the SP500 data using backtesting."""
    def backtest(data, model, predictors, start=2500, step=250):
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            model.fit(train[predictors], train["Target"])
            preds = model.predict(test[predictors])
            preds = pd.Series(preds, index=test.index, name="Predictions")
            combined = pd.concat([test["Target"], preds], axis=1)
            all_predictions.append(combined)
        return pd.concat(all_predictions)
    
    model = XGBClassifier(random_state=1, learning_rate=0.1, n_estimators=500)
    predictions = backtest(sp500, model, predictors)
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    print("XGBoost Model Precision:", precision)
    return model, predictions

def predict_next_day(model, sp500, predictors=["Close", "Volume", "Open", "High", "Low"]):
    """Make a prediction for the next day using the latest available data."""
    latest_data = sp500.iloc[-1][predictors].to_frame().T
    prediction = int(model.predict(latest_data)[0])
    prediction_date = sp500.index[-1].strftime("%Y-%m-%d")
    return prediction, prediction_date

if __name__ == '__main__':
    # Example usage when running this module as a script:
    sp500 = load_sp500_data()
    sp500 = preprocess_data(sp500)
    model = train_rf_model(sp500)
    pred, date = predict_next_day(model, sp500)
    print(f"Prediction for {date}: {pred}")
