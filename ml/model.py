from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def create_model():
    """
    Create the machine learning model used by FinSight.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    return model


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
