import os

import joblib
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from ml.model import FEATURES, create_features


MODEL_PATH = os.path.join(os.path.dirname(__file__), "stock_model.pkl")

TICKERS = [
    "AAPL", "NVDA", "MSFT", "TSLA", "GOOGL",
    "AMZN", "META", "NFLX", "AMD", "INTC",
]


def make_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=20,
            random_state=42,
        ),
    }


def download_stock_data(ticker):
    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        period="5y",
        auto_adjust=True,
        progress=False,
        threads=False,
        multi_level_index=False,
        timeout=30,
    )

    if data.empty:
        raise ValueError(f"No historical data for {ticker}")

    data = create_features(data, include_target=True)
    data["ticker"] = ticker
    return data


def train_model():
    datasets = []

    for ticker in TICKERS:
        try:
            datasets.append(download_stock_data(ticker))
        except Exception as error:
            print(f"Warning: {ticker} failed: {error}")

    if not datasets:
        raise ValueError("No training data downloaded.")

    data = pd.concat(datasets).sort_index()
    data = data.replace([float("inf"), float("-inf")], float("nan"))
    data = data.dropna(subset=FEATURES + ["target"])

    dates = data.index.unique().sort_values()
    n_dates = len(dates)

    if n_dates < 200:
        raise ValueError("Not enough historical dates.")

    # Expanding-window validation: each fold trains on the past
    # and validates on the following, unseen dates.
    fold_starts = [0.50, 0.60, 0.70, 0.80]
    scores = {name: [] for name in make_models()}

    print(f"\nTotal rows: {len(data)}")
    print(f"Unique trading dates: {n_dates}")
    print("\nWalk-forward validation:")

    for fold_number, fraction in enumerate(fold_starts, start=1):
        train_end = int(n_dates * fraction)
        val_end = int(n_dates * (fraction + 0.10))

        train = data.loc[data.index < dates[train_end]]
        validation = data.loc[
            (data.index >= dates[train_end])
            & (data.index < dates[val_end])
        ]

        X_train, y_train = train[FEATURES], train["target"]
        X_val, y_val = validation[FEATURES], validation["target"]

        print(f"\nFold {fold_number} — train: {len(train)}, "
              f"validation: {len(validation)}")

        for name, model in make_models().items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            accuracy = accuracy_score(y_val, predictions)
            balanced = balanced_accuracy_score(y_val, predictions)

            scores[name].append(balanced)

            print(
                f"{name}: accuracy={accuracy:.2%}, "
                f"balanced accuracy={balanced:.2%}"
            )

    print("\nAverage walk-forward balanced accuracy:")
    for name, fold_scores in scores.items():
        print(f"{name}: {sum(fold_scores) / len(fold_scores):.2%}")

    # Choose using validation periods only, not the final holdout.
    best_name = max(
        scores,
        key=lambda name: sum(scores[name]) / len(scores[name]),
    )
    print(f"\nSelected model: {best_name}")

    # Evaluate the chosen model on the latest 10% of dates.
    holdout_start = int(n_dates * 0.90)

    training = data.loc[data.index < dates[holdout_start]]
    holdout = data.loc[data.index >= dates[holdout_start]]

    selected_model = make_models()[best_name]
    selected_model.fit(training[FEATURES], training["target"])

    predictions = selected_model.predict(holdout[FEATURES])

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(training[FEATURES], training["target"])
    baseline_predictions = baseline.predict(holdout[FEATURES])

    print("\nFinal holdout results:")
    print(f"Model accuracy: {accuracy_score(holdout['target'], predictions):.2%}")
    print(
        "Model balanced accuracy: "
        f"{balanced_accuracy_score(holdout['target'], predictions):.2%}"
    )
    print(
        f"Baseline accuracy: "
        f"{accuracy_score(holdout['target'], baseline_predictions):.2%}"
    )
    print(
        "Baseline balanced accuracy: "
        f"{balanced_accuracy_score(holdout['target'], baseline_predictions):.2%}"
    )

    # After evaluation, retrain the chosen model on all available data.
    selected_model.fit(data[FEATURES], data["target"])
    joblib.dump(selected_model, MODEL_PATH)

    print(f"\nSaved {best_name} to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()