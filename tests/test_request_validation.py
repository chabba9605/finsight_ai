
import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)

VALID_PAYLOAD = {
    "session_id": "validation-test",
    "ticker": "NVDA",
    "message": "Analyse NVIDIA stock",
}


@pytest.mark.parametrize(
    "changes",
    [
        {"message": "   "},
        {"message": "A" * 2001},
        {"session_id": "   "},
        {"session_id": "A" * 101},
        {"ticker": "BAD!TICKER"},
        {"ticker": "A" * 16},
    ],
)
def test_reject_invalid_input(changes):
    response = client.post(
        "/chat/stream",
        json={**VALID_PAYLOAD, **changes},
    )

    assert response.status_code == 422


def test_missing_ticker_remains_400():
    response = client.post(
        "/chat/stream",
        json={**VALID_PAYLOAD, "ticker": ""},
    )

    assert response.status_code == 400


def test_health_unaffected():
    response = client.get("/health")

    assert response.status_code == 200


def test_ticker_normalization():
    from schema import ChatRequest

    result = ChatRequest(
        session_id="abc123",
        ticker=" nvda ",
        message="  Analyse NVIDIA  ",
    )

    assert result.ticker == "NVDA"
    assert result.message == "Analyse NVIDIA"
