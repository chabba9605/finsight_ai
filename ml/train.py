import os

import joblib
import pandas as pd
import yfinance as yf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ml.model import (
    FEATURES,
    create_features,
    create_model,
)


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "stock_model.pkl",
)


TICKERS = [
    "AAPL",
    "NVDA",
    "MSFT",
    "TSLA",
    "GOOGL",
    "AMZN",
    "META",
    "NFLX",
    "AMD",
    "INTC",
]


def download_stock_data(ticker: str) -> pd.DataFrame:
    """
    Download five years of historical market data
    for a single ticker.
    """

    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        period="5y",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No historical data found for {ticker}"
        )

    data = create_features(
        data,
        include_target=True,
    )

    # Keep track of which company each observation belongs to.
    data["ticker"] = ticker

    return data


def train_model():
    """
    Train the Random Forest model using historical data
    from multiple stocks.
    """

    all_data = []

    for ticker in TICKERS:
        try:
            data = download_stock_data(ticker)
            all_data.append(data)

        except Exception as e:
            print(
                f"Warning: Could not process {ticker}: {e}"
            )

    if not all_data:
        raise ValueError(
            "No training data could be downloaded."
        )

    # Correctly combine all ticker datasets.
    combined = pd.concat(
        all_data,
        axis=0,
    )

    # Sort by date so that the train/test split remains
    # chronological rather than random.
    combined = combined.sort_index()

    X = combined[FEATURES]
    y = combined["target"]

    if len(X) < 100:
        raise ValueError(
            "Not enough training data available."
        )

    # Do NOT shuffle time-series data.
    split_index = int(len(combined) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print()
    print("Training data:")
    print(f"Total samples: {len(combined)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print()

    model = create_model()

    print("Training Random Forest model...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print(
        f"Model accuracy: {accuracy:.2%}"
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    # Save the trained model.
    joblib.dump(
        model,
        MODEL_PATH,
    )

    print()
    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    train_model()