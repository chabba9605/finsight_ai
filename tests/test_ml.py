
import numpy as np
import pandas as pd
import pytest

from ml import predict as predictor


class FakeModel:
    def predict(self, features):
        return np.array([0])

    def predict_proba(self, features):
        return np.array([[0.55, 0.45]])


def fake_market_data():
    rng = np.random.default_rng(42)
    rows = 150

    prices = 100 + np.cumsum(
        rng.normal(loc=0.2, scale=1.5, size=rows)
    )

    volumes = rng.integers(
        1_000_000, 5_000_000, size=rows
    )

    return pd.DataFrame({
        "Close": prices,
        "Volume": volumes,
    }, index=pd.bdate_range(
        "2026-01-01", periods=rows
    ))


def setup_mock_model(monkeypatch):
    monkeypatch.setattr(
        predictor.os.path,
        "exists",
        lambda path: True,
    )
    monkeypatch.setattr(
        predictor.joblib,
        "load",
        lambda path: FakeModel(),
    )


def test_valid_prediction(monkeypatch):
    setup_mock_model(monkeypatch)

    monkeypatch.setattr(
        predictor.yf,
        "download",
        lambda *args, **kwargs: fake_market_data(),
    )

    result = predictor.predict_stock(" nvda ")

    assert result["ticker"] == "NVDA"
    assert result["prediction"] == "DOWN"
    assert result["score"] == pytest.approx(0.55)


def test_probabilities(monkeypatch):
    setup_mock_model(monkeypatch)

    monkeypatch.setattr(
        predictor.yf,
        "download",
        lambda *args, **kwargs: fake_market_data(),
    )

    result = predictor.predict_stock("AAPL")

    assert 0 <= result["probability_up"] <= 1
    assert 0 <= result["probability_down"] <= 1

    assert (
        result["probability_up"]
        + result["probability_down"]
    ) == pytest.approx(1.0)


def test_empty_ticker():
    with pytest.raises(ValueError, match="Ticker symbol cannot be empty"):
        predictor.predict_stock("   ")


def test_missing_market_data(monkeypatch):
    setup_mock_model(monkeypatch)

    monkeypatch.setattr(
        predictor.yf,
        "download",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="No market data"):
        predictor.predict_stock("INVALID123")


def test_missing_model(monkeypatch):
    monkeypatch.setattr(
        predictor.os.path,
        "exists",
        lambda path: False,
    )

    with pytest.raises(FileNotFoundError, match="ML model not found"):
        predictor.predict_stock("NVDA")
