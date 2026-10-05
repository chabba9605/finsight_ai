import os

import joblib
import yfinance as yf

from ml.model import (
    FEATURES,
    create_features,
)


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "stock_model.pkl",
)


def predict_stock(ticker: str) -> dict:
    """
    Generate an ML prediction for a stock.

    Returns:
        ticker
        prediction
        probability_up
        probability_down
        score
    """

    ticker = ticker.upper().strip()

    if not ticker:
        raise ValueError(
            "Ticker symbol cannot be empty."
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "ML model not found. "
            "Run: python -m ml.train"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Getting recent ML data for {ticker}..."
    )

    data = yf.download(
        ticker,
        period="3mo",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No market data found for {ticker}"
        )

    # Prediction does not need the target column.
    data = create_features(
        data,
        include_target=False,
    )

    if data.empty:
        raise ValueError(
            f"Not enough data to generate ML features for {ticker}"
        )

    latest = data[
        FEATURES
    ].iloc[[-1]]

    prediction = int(
        model.predict(latest)[0]
    )

    probabilities = model.predict_proba(
        latest
    )[0]

    probability_down = float(
        probabilities[0]
    )

    probability_up = float(
        probabilities[1]
    )

    if prediction == 1:
        direction = "UP"
        score = probability_up
    else:
        direction = "DOWN"
        score = probability_down

    return {
        "ticker": ticker,
        "prediction": direction,
        "probability_up": probability_up,
        "probability_down": probability_down,
        "score": float(score),
    }


if __name__ == "__main__":

    import sys

    ticker = "AAPL"

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()

    result = predict_stock(
        ticker
    )

    print()
    print("ML Prediction")
    print("----------------")
    print(
        f"Ticker: {result['ticker']}"
    )
    print(
        f"Prediction: {result['prediction']}"
    )
    print(
        f"Probability UP: "
        f"{result['probability_up']:.2%}"
    )
    print(
        f"Probability DOWN: "
        f"{result['probability_down']:.2%}"
    )