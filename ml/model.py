from sklearn.ensemble import RandomForestClassifier
import pandas as pd


FEATURES = [
    "return_1d",
    "return_5d",
    "ma_ratio",
    "volatility",
    "volume_change",
]


def create_model():
    """
    Create the machine learning model used by FinSight AI.

    The model predicts whether the next trading day's
    closing price will be higher or lower than the current
    closing price.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return model


def _prepare_price_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise yfinance data so that Close and Volume are
    always accessible as normal columns.

    Recent versions of yfinance can sometimes return
    MultiIndex columns.
    """

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in data.columns
        ]

    required_columns = {"Close", "Volume"}

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required market columns: {sorted(missing_columns)}"
        )

    return data


def create_features(
    data: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Create technical features for the ML model.

    Features:
        return_1d
        return_5d
        ma_ratio
        volatility
        volume_change

    When include_target=True, also creates:

        target = 1 if next day's close is higher,
                 0 otherwise.
    """

    data = _prepare_price_columns(data)

    data["return_1d"] = data["Close"].pct_change()

    data["return_5d"] = data["Close"].pct_change(5)

    data["ma_5"] = data["Close"].rolling(window=5).mean()

    data["ma_20"] = data["Close"].rolling(window=20).mean()

    data["ma_ratio"] = data["ma_5"] / data["ma_20"]

    data["volatility"] = (
        data["Close"]
        .pct_change()
        .rolling(window=10)
        .std()
    )

    data["volume_change"] = data["Volume"].pct_change()

    if include_target:
        # 1 = next trading day's close is higher
        # 0 = next trading day's close is lower or equal
        data["target"] = (
            data["Close"].shift(-1) > data["Close"]
        ).astype("float")

        # The final row has no known next-day price.
        # Therefore its target is invalid and must be removed.
        data.loc[
            data["Close"].shift(-1).isna(),
            "target"
        ] = pd.NA

        data = data.dropna(
            subset=FEATURES + ["target"]
        )

        data["target"] = data["target"].astype(int)

    else:
        data = data.dropna(
            subset=FEATURES
        )

    return data