
from fastapi.testclient import TestClient

import chat_stream
from main import app


client = TestClient(app)

PAYLOAD = {
    "session_id": "rate-limit-test",
    "ticker": "NVDA",
    "message": "Analyse NVIDIA stock",
}


def fake_stream(state, config, stream_mode):
    yield {
        "data_agent": {
            "ticker": "NVDA",
            "market_data": {
                "ticker": "NVDA",
                "current_price": 200.0,
                "5d_return": 1.0,
            },
            "ml_prediction": None,
        }
    }

    yield {
        "analysis_agent": {
            "analysis": {
                "recommendation": "HOLD",
                "confidence": "LOW",
                "reason": "Mixed market signals.",
                "key_financial_factors": [
                    "Uncertain short-term momentum"
                ],
                "news_impact": [
                    "Limited news evidence"
                ],
                "main_risk": "Price volatility",
                "data_limitation": "Insufficient data",
            }
        }
    }


def test_shared_analysis_rate_limit(monkeypatch):
    monkeypatch.setattr(
        chat_stream.pipeline,
        "stream",
        fake_stream,
    )

    for request_number in range(1, 6):
        response = client.post(
            "/chat/stream",
            json=PAYLOAD,
        )

        assert response.status_code == 200, (
            f"Request {request_number}: "
            f"{response.status_code} {response.text}"
        )

        assert '"type": "completed"' in response.text

    response = client.post(
        "/chat/stream",
        json=PAYLOAD,
    )

    assert response.status_code == 429, (
        "Rate limiter did not block the sixth request. "
        f"Received {response.status_code}: "
        f"{response.text}"
    )

    response = client.get("/health")
    assert response.status_code == 200
