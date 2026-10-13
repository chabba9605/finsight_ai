"""Tests for POST /chat/stream. All LangGraph operations are mocked."""
import json

from fastapi.testclient import TestClient

import chat_stream
from main import app

client = TestClient(app)

REQUEST = {
    "session_id": "stream-test-01",
    "ticker": " nvda ",
    "message": "Analyse NVIDIA",
}

ANALYSIS = {
    "recommendation": "HOLD",
    "confidence": "MEDIUM",
    "reason": "Mixed signals.",
    "key_financial_factors": ["Recent price change"],
    "news_impact": ["Recent earnings coverage [Source 1]"],
    "main_risk": "Volatility",
    "data_limitation": "Limited data",
}


def successful_stream(state, config, stream_mode):
    assert state["ticker"] == "NVDA"
    assert config["configurable"]["thread_id"] == REQUEST["session_id"]
    assert stream_mode == "updates"
    yield {"data_agent": {
        "ticker": "NVDA",
        "market_data": {"ticker": "NVDA", "current_price": 200.0, "5d_return": 2.5},
        "ml_prediction": None,
    }}
    yield {"retrieval_agent": {"sources": ["https://example.com/nvda"]}}
    yield {"analysis_agent": {"analysis": ANALYSIS}}


def events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_stream_completes_and_reports_stages(monkeypatch):
    monkeypatch.setattr(chat_stream.pipeline, "stream", successful_stream)
    response = client.post("/chat/stream", json=REQUEST)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    items = events(response)
    assert [item["type"] for item in items] == [
        "started", "stage_completed", "stage_completed", "stage_completed", "completed"
    ]
    assert [item["stage"] for item in items if item["type"] == "stage_completed"] == [
        "market_data", "news_retrieval", "ai_analysis"
    ]
    report = items[-1]["result"]
    assert report["ticker"] == "NVDA"
    assert report["session_id"] == REQUEST["session_id"]
    assert report["market_data"]["five_day_return"] == 2.5
    assert report["analysis"]["recommendation"] == "HOLD"
    assert report["sources"] == ["https://example.com/nvda"]


def test_missing_ticker_returns_400():
    response = client.post("/chat/stream", json={**REQUEST, "ticker": ""})
    assert response.status_code == 400
    assert "ticker" in response.json()["detail"].lower()


def test_whitespace_ticker_returns_400():
    response = client.post("/chat/stream", json={**REQUEST, "ticker": "   "})
    assert response.status_code == 400


def test_missing_message_returns_422():
    response = client.post("/chat/stream", json={**REQUEST, "message": ""})
    assert response.status_code == 422


def test_value_error_emitted_as_stream_event(monkeypatch):
    def failing_stream(*args, **kwargs):
        raise ValueError("No market data found for INVALID123")
        yield  # turns function into generator

    monkeypatch.setattr(chat_stream.pipeline, "stream", failing_stream)
    response = client.post("/chat/stream", json={**REQUEST, "ticker": "INVALID123"})
    assert response.status_code == 200  # streaming headers have already been sent
    items = events(response)
    assert items[0]["type"] == "started"
    assert items[-1] == {
        "type": "error", "message": "No market data found for INVALID123"
    }
    assert not any(item["type"] == "completed" for item in items)


def test_internal_error_is_sanitised(monkeypatch):
    def failing_stream(*args, **kwargs):
        raise RuntimeError("secret internal traceback")
        yield

    monkeypatch.setattr(chat_stream.pipeline, "stream", failing_stream)
    response = client.post("/chat/stream", json=REQUEST)
    items = events(response)
    assert items[-1]["type"] == "error"
    assert "secret internal traceback" not in response.text
    assert not any(item["type"] == "completed" for item in items)


def test_incomplete_output_emits_error(monkeypatch):
    def incomplete_stream(*args, **kwargs):
        yield {"data_agent": {"ticker": "NVDA"}}

    monkeypatch.setattr(chat_stream.pipeline, "stream", incomplete_stream)
    response = client.post("/chat/stream", json=REQUEST)
    items = events(response)
    assert items[1] == {"type": "stage_completed", "stage": "market_data"}
    assert items[-1]["type"] == "error"
    assert not any(item["type"] == "completed" for item in items)
