import os
import joblib
import yfinance as yf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml.model import create_model
from ml.model import create_features

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "stock_model.pkl"
)


FEATURES = [
    "return_1d",
    "return_5d",
    "ma_ratio",
    "volatility",
    "volume_change"
]

tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "NFLX", "AMD", "INTC"]

def train_model(ticker: str = "AAPL"):

    all_data = []
    for ticker in tickers:
        print(f"Downloading {ticker}....")
        data = yf.download(
                ticker,
                period="5y",
                auto_adjust=True,
                progress=False
            )
        data = create_features(data)
        all_data = data.append(data)

    combined = pd.concat(all_data)
    

    if data.empty:
        raise ValueError(
            f"No historical data found for {ticker}"
        )

    X = data[FEATURES]
    y = data["target"]

    # Do NOT shuffle time-series data.
    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = create_model()

    print("Training model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"\nModel accuracy: {accuracy:.2%}\n")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":

    import sys

    ticker = "AAPL"

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()

    train_model(ticker)