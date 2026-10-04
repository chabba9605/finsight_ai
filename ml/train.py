import os
import joblib
import yfinance as yf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml.model import create_model


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


def create_features(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()

    data["return_1d"] = data["Close"].pct_change()

    data["return_5d"] = data["Close"].pct_change(5)

    data["ma_5"] = data["Close"].rolling(5).mean()

    data["ma_20"] = data["Close"].rolling(20).mean()

    data["ma_ratio"] = data["ma_5"] / data["ma_20"]

    data["volatility"] = (
        data["Close"]
        .pct_change()
        .rolling(10)
        .std()
    )

    data["volume_change"] = data["Volume"].pct_change()

    # Target:
    # 1 = price goes up next trading day
    # 0 = price goes down next trading day

    data["target"] = (
        data["Close"].shift(-1) > data["Close"]
    ).astype(int)

    data = data.dropna()

    return data


def train_model(ticker: str = "AAPL"):

    print(f"Downloading historical data for {ticker}...")

    data = yf.download(
        ticker,
        period="5y",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No historical data found for {ticker}"
        )

    data = create_features(data)

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