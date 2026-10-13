"""Offline tests for FinSight AI's historical-price endpoint."""

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import stock_history


app = FastAPI()
app.include_router(stock_history.router)
client = TestClient(app)


class FakeStock:
    def __init__(self, frame=None, error=None, currency="USD"):
        self.frame = frame
        self.error = error
        self.fast_info = {"currency": currency}
        self.requested_period = None

    def history(self, *, period, interval, auto_adjust):
        self.requested_period = period
        assert interval == "1d"
        assert auto_adjust is True
        if self.error:
            raise self.error
        return self.frame


@pytest.fixture
def market_data():
    return pd.DataFrame(
        {"Close": [100.0, 102.34567, 104.5]},
        index=pd.to_datetime(["2026-10-07", "2026-10-08", "2026-10-09"]),
    )


def test_history_success(monkeypatch, market_data):
    stock = FakeStock(frame=market_data)
    monkeypatch.setattr(stock_history.yf, "Ticker", lambda symbol: stock)

    response = client.get("/stock/NVDA/history?period=3mo")

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "NVDA"
    assert data["period"] == "3mo"
    assert data["currency"] == "USD"
    assert data["prices"] == [
        {"date": "2026-10-07", "close": 100.0},
        {"date": "2026-10-08", "close": 102.3457},
        {"date": "2026-10-09", "close": 104.5},
    ]
    assert stock.requested_period == "3mo"


def test_ticker_normalization(monkeypatch, market_data):
    received = []
    monkeypatch.setattr(
        stock_history.yf,
        "Ticker",
        lambda symbol: (received.append(symbol) or FakeStock(frame=market_data)),
    )

    response = client.get("/stock/%20aapl%20/history")
    assert response.status_code == 200
    assert received == ["AAPL"]
    assert response.json()["period"] == "1mo"


@pytest.mark.parametrize("ticker", ["BAD$TICKER", "TOO_LONG_TICKER_123"])
def test_invalid_ticker(ticker):
    response = client.get(f"/stock/{ticker}/history")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid stock ticker"


def test_invalid_period():
    response = client.get("/stock/NVDA/history?period=5y")
    assert response.status_code == 422


def test_missing_prices(monkeypatch):
    monkeypatch.setattr(
        stock_history.yf,
        "Ticker",
        lambda symbol: FakeStock(frame=pd.DataFrame()),
    )

    response = client.get("/stock/NVDA/history")
    assert response.status_code == 404
    assert "No historical prices" in response.json()["detail"]


def test_provider_failure(monkeypatch):
    monkeypatch.setattr(
        stock_history.yf,
        "Ticker",
        lambda symbol: FakeStock(error=RuntimeError("provider timeout")),
    )

    response = client.get("/stock/NVDA/history")
    assert response.status_code == 502
    assert response.json()["detail"] == "Market data provider unavailable"
