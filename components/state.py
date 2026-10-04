from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict

class FinSightState(TypedDict):
    session_id: str
    ticker: str
    question: str
    chat_history: List
    market_data: Optional[Dict[str, Any]]
    retrieved_context: Optional[str]
    sources: Optional[List[str]]
    analysis: Optional[str]
    ml_score: Optional[float]