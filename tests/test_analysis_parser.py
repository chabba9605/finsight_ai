
import pytest

from nodes.analysis_agent import parse_analysis


VALID_REPORT = """
RECOMMENDATION: HOLD
CONFIDENCE: LOW

REASON:
Current evidence does not establish a reliable
directional advantage.

KEY FINANCIAL FACTORS:
- Short-term momentum is uncertain.

NEWS IMPACT:
- News evidence is limited.

MAIN RISK:
Price volatility.

DATA LIMITATION:
Insufficient forward-looking evidence.
"""


def test_valid_recommendation():
    result = parse_analysis(VALID_REPORT)

    assert result["recommendation"] == "HOLD"
    assert result["confidence"] == "LOW"
    assert result["reason"]


def test_placeholder_recommendation_rejected():
    report = VALID_REPORT.replace(
        "RECOMMENDATION: HOLD",
        "RECOMMENDATION: BUY / HOLD / SELL",
    )

    with pytest.raises(ValueError):
        parse_analysis(report)


def test_missing_reason_rejected():
    report = VALID_REPORT.replace(
        "Current evidence does not establish a reliable\n"
        "directional advantage.",
        "",
    )

    with pytest.raises(ValueError):
        parse_analysis(report)


def test_invalid_confidence_rejected():
    report = VALID_REPORT.replace(
        "CONFIDENCE: LOW",
        "CONFIDENCE: UNKNOWN",
    )

    with pytest.raises(ValueError):
        parse_analysis(report)
