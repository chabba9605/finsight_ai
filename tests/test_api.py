
import pytest
from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Replace LangGraph execution with predictable test data."""

    def fake_invoke(state, config):
        return {
            "ticker": state["ticker"],
            "market_data": {
                "ticker": state["ticker"],
                "current_price": 225.50,
                "5d_return": -2.5,
            },
            "ml_prediction": {
                "ticker": state["ticker"],
                "prediction": "DOWN",
                "probability_up": 0.45,
                "probability_down": 0.55,
                "score": 0.55,
            },
            "analysis": {
                "recommendation": "HOLD",
                "confidence": "MEDIUM",
                "reason": "Mixed market signals.",
                "key_financial_factors": [
                    "Recent price decline"
                ],
                "news_impact": [
                    "Mixed financial news"
                ],
                "main_risk": "Market volatility",
                "data_limitation": "Limited recent data",
            },
            "sources": [
                "https://example.com/news"
            ],
        }

    monkeypatch.setattr(
        main.pipeline,
        "invoke",
        fake_invoke,
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_chat(mock_pipeline):
    response = client.post(
        "/chat",
        json={
            "session_id": "test001",
            "ticker": "nvda",
            "message": "Analyse NVIDIA stock",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["ticker"] == "NVDA"
    assert data["session_id"] == "test001"
    assert data["analysis"]["recommendation"] == "HOLD"
    assert data["ml_prediction"]["prediction"] == "DOWN"
    assert len(data["sources"]) == 1


def test_missing_ticker():
    response = client.post(
        "/chat",
        json={
            "session_id": "test002",
            "message": "Analyse stock",
        },
    )

    assert response.status_code == 400
    assert "ticker is required" in response.json()["detail"]


def test_invalid_request():
    response = client.post(
        "/chat",
        json={
            "session_id": "",
            "ticker": "NVDA",
            "message": "",
        },
    )

    assert response.status_code == 422


def test_invalid_ticker(monkeypatch):
    def fake_invoke(state, config):
        raise ValueError(
            "No market data found for INVALID123"
        )

    monkeypatch.setattr(
        main.pipeline,
        "invoke",
        fake_invoke,
    )

    response = client.post(
        "/chat",
        json={
            "session_id": "test003",
            "ticker": "INVALID123",
            "message": "Analyse this stock",
        },
    )

    assert response.status_code == 400
    assert "No market data found" in response.json()["detail"]


def test_missing_market_data(monkeypatch):
    def fake_invoke(state, config):
        return {
            "ticker": "NVDA",
            "analysis": {"recommendation": "HOLD"},
        }

    monkeypatch.setattr(
        main.pipeline,
        "invoke",
        fake_invoke,
    )

    response = client.post(
        "/chat",
        json={
            "session_id": "test004",
            "ticker": "NVDA",
            "message": "Analyse NVIDIA",
        },
    )

    assert response.status_code == 500
    assert "Market data was not returned" in response.json()["detail"]
