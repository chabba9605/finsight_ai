
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=100,
        description="Unique conversation/session identifier",
    )

    message: str = Field(
        min_length=1,
        max_length=2000,
        description="User's financial research question",
    )

    ticker: Optional[str] = Field(
        default=None,
        max_length=15,
        description="Stock ticker symbol, e.g. AAPL",
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Session ID cannot be empty.")

        return value

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value

    @field_validator("ticker")
    @classmethod
    def validate_ticker(
        cls, value: Optional[str]
    ) -> Optional[str]:
        if value is None:
            return None

        value = value.strip().upper()

        # Preserve the existing HTTP 400 response
        # for missing or blank ticker symbols.
        if not value:
            return value

        if not re.fullmatch(
            r"[A-Z0-9^][A-Z0-9.^-]*", value
        ):
            raise ValueError(
                "Invalid ticker symbol."
            )

        return value


class MarketDataResponse(BaseModel):
    ticker: str
    current_price: float
    five_day_return: float


class MLPredictionResponse(BaseModel):
    ticker: str
    prediction: str
    probability_up: float
    probability_down: float
    score: float


class AnalysisResponse(BaseModel):
    recommendation: str
    confidence: str
    reason: str
    key_financial_factors: List[str]
    news_impact: List[str]
    main_risk: str
    data_limitation: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    ticker: str
    message: str
    market_data: MarketDataResponse
    ml_prediction: Optional[MLPredictionResponse] = None
    analysis: AnalysisResponse
    sources: List[str]


class HealthResponse(BaseModel):
    status: str
