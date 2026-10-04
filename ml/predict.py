import os
import joblib
import yfinance as yf
import pandas as pd


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

    data["ma_ratio"] = (
        data["ma_5"] / data["ma_20"]
    )

    data["volatility"] = (
        data["Close"]
        .pct_change()
        .rolling(10)
        .std()
    )

    data["volume_change"] = (
        data["Volume"].pct_change()
    )

    data = data.dropna()

    return data


def predict_stock(ticker: str):

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "ML model not found. "
            "Run: python -m ml.train AAPL"
        )

    model = joblib.load(MODEL_PATH)

    print(
        f"Getting recent data for {ticker}..."
    )

    data = yf.download(
        ticker,
        period="3mo",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No data found for {ticker}"
        )

    data = create_features(data)

    latest = data[FEATURES].iloc[[-1]]

    prediction = int(
        model.predict(latest)[0]
    )

    probabilities = model.predict_proba(
        latest
    )[0]

    probability_up = float(
        probabilities[1]
    )

    probability_down = float(
        probabilities[0]
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
        "score": float(score)
    }


if __name__ == "__main__":

    import sys

    ticker = "AAPL"

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()

    result = predict_stock(ticker)

    print("\nML Prediction")
    print("----------------")
    print(f"Ticker: {result['ticker']}")
    print(f"Prediction: {result['prediction']}")
    print(
        f"Probability UP: "
        f"{result['probability_up']:.2%}"
    )
    print(
        f"Probability DOWN: "
        f"{result['probability_down']:.2%}"
    )