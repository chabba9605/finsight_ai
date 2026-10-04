from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    session_id: str
    message: str
    ticker: Optional[str] = None


class MarketDataResponse(BaseModel):
    ticker: str
    current_price: float
    five_day_return: float


class AnalysisResponse(BaseModel):
    recommendation: str
    confidence: str
    reason: str
    key_financial_factors: list[str]
    news_impact: list[str]
    main_risk: str
    data_limitation: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    ticker: str
    message: str
    market_data: MarketDataResponse
    analysis: dict
    sources: list[str]


class HealthResponse(BaseModel):
    status: str