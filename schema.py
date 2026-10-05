from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        description="Unique conversation/session identifier",
    )

    message: str = Field(
        min_length=1,
        description="User's financial question",
    )

    ticker: Optional[str] = Field(
        default=None,
        description="Stock ticker symbol, e.g. AAPL",
    )


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

    ml_prediction: Optional[
        MLPredictionResponse
    ] = None

    analysis: AnalysisResponse

    sources: List[str]


class HealthResponse(BaseModel):
    status: str