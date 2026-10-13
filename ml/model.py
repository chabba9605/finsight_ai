import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FEATURES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_ratio",
    "trend_50d",
    "volatility",
    "volume_change",
    "rsi_14",
    "macd_hist",
]


def create_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )


def _prepare_price_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in data.columns
        ]

    required = {"Close", "Volume"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return data


def create_features(
    data: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    data = _prepare_price_columns(data)

    close = data["Close"]
    volume = data["Volume"]
    daily_return = close.pct_change()

    # Returns and moving-average trends
    data["return_1d"] = daily_return
    data["return_5d"] = close.pct_change(5)
    data["return_10d"] = close.pct_change(10)

    ma_5 = close.rolling(5).mean()
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()

    data["ma_ratio"] = ma_5 / ma_20
    data["trend_50d"] = close / ma_50 - 1

    # Recent volatility and volume change
    data["volatility"] = daily_return.rolling(10).std()
    data["volume_change"] = volume.pct_change()

    # RSI (14 trading days)
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    relative_strength = avg_gain / avg_loss
    data["rsi_14"] = 100 - (100 / (1 + relative_strength))

    # MACD histogram, scaled by price
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()

    data["macd_hist"] = (macd - signal) / close

    # Predict whether the next close is higher
    if include_target:
        next_close = close.shift(-1)
        data["target"] = (next_close > close).astype(float)
        data.loc[next_close.isna(), "target"] = float("nan")

    data = data.replace([float("inf"), float("-inf")], float("nan"))

    columns_to_check = FEATURES + (["target"] if include_target else [])
    data = data.dropna(subset=columns_to_check)

    if include_target:
        data["target"] = data["target"].astype(int)

    return data