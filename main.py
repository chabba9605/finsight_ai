from fastapi import FastAPI, HTTPException

from schema import (
    ChatRequest,
    ChatResponse,
    HealthResponse
)

from graph import pipeline


app = FastAPI(
    title="FinSight AI",
    description="AI-powered financial research and stock analysis API",
    version="1.0.0"
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        result = pipeline.invoke(
            {
                "session_id": request.session_id,
                "ticker": request.ticker or "",
                "question": request.message,
                "chat_history": [],
                "market_data": None,
                "retrieved_context": None,
                "sources": None,
                "analysis": None,
                "ml_score": None
            },
            config={
                "configurable": {
                    "thread_id": request.session_id
                }
            }
        )

        market_data = result.get("market_data")

        if not market_data:
            raise HTTPException(
                status_code=500,
                detail="Market data was not returned."
            )

        return {
            "session_id": request.session_id,
            "ticker": result["ticker"],
            "message": request.message,

            "market_data": {
                "ticker": market_data["ticker"],
                "current_price": float(market_data["current_price"]),
                "five_day_return": float(market_data["5d_return"])
            },

            "analysis": result["analysis"],

            "sources": result.get("sources", [])
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )