"""Stream completed LangGraph agent stages as newline-delimited JSON."""
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from graph import pipeline
from schema import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)

STAGES = {
    "data_agent": "market_data",
    "retrieval_agent": "news_retrieval",
    "analysis_agent": "ai_analysis",
}


def event(kind: str, **data) -> str:
    return json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n"


def format_report(request: ChatRequest, state: dict) -> dict:
    market = state.get("market_data")
    analysis = state.get("analysis")
    if not market or not analysis:
        raise RuntimeError("Incomplete analysis returned by pipeline")

    return ChatResponse.model_validate({
        "session_id": request.session_id,
        "ticker": state["ticker"],
        "message": request.message,
        "market_data": {
            "ticker": market["ticker"],
            "current_price": float(market["current_price"]),
            "five_day_return": float(market["5d_return"]),
        },
        "ml_prediction": state.get("ml_prediction"),
        "analysis": analysis,
        "sources": state.get("sources", []),
    }).model_dump(mode="json")


@router.post("/chat/stream")
def stream_chat(request: ChatRequest):
    if not request.ticker or not request.ticker.strip():
        raise HTTPException(status_code=400, detail="A stock ticker is required.")

    def generate():
        config = {"configurable": {"thread_id": request.session_id}}
        state = {
            "session_id": request.session_id,
            "ticker": request.ticker.strip().upper(),
            "question": request.message,
        }
        merged = dict(state)
        yield event("started")
        try:
            # Each update is emitted only AFTER the actual LangGraph node finishes.
            for update in pipeline.stream(state, config=config, stream_mode="updates"):
                for node, changes in update.items():
                    if isinstance(changes, dict):
                        merged.update(changes)
                    if node in STAGES:
                        yield event("stage_completed", stage=STAGES[node])

            # The merged node updates contain this request's complete final output.
            yield event("completed", result=format_report(request, merged))
        except ValueError as exc:
            yield event("error", message=str(exc))
        except Exception:
            logger.exception("Streaming research failed")
            yield event("error", message="Research could not be completed. Please try again.")

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
