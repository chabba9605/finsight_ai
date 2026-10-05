from fastapi import FastAPI, HTTPException

from graph import pipeline

from schema import (
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MLPredictionResponse,
)


app = FastAPI(
    title="FinSight AI",
    description=(
        "AI-powered financial research "
        "and stock analysis API"
    ),
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    Health check endpoint.
    """

    return {
        "status": "ok"
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):
    """
    Run a financial research conversation
    through the LangGraph pipeline.
    """

    if not request.ticker:
        raise HTTPException(
            status_code=400,
            detail=(
                "A stock ticker is required."
            ),
        )

    ticker = request.ticker.upper().strip()

    try:

        # IMPORTANT:
        # We intentionally do not reset chat_history here.
        # LangGraph's checkpointer uses session_id/thread_id
        # to retrieve previous conversation state.
        result = pipeline.invoke(
            {
                "session_id": request.session_id,
                "ticker": ticker,
                "question": request.message,
            },
            config={
                "configurable": {
                    "thread_id": request.session_id
                }
            },
        )

        market_data = result.get(
            "market_data"
        )

        if not market_data:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Market data was not returned."
                ),
            )

        analysis = result.get(
            "analysis"
        )

        if not analysis:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Analysis was not returned."
                ),
            )

        ml_prediction = result.get(
            "ml_prediction"
        )

        return {
            "session_id": request.session_id,

            "ticker": result[
                "ticker"
            ],

            "message": request.message,

            "market_data": {
                "ticker": market_data[
                    "ticker"
                ],

                "current_price": float(
                    market_data[
                        "current_price"
                    ]
                ),

                "five_day_return": float(
                    market_data[
                        "5d_return"
                    ]
                ),
            },

            "ml_prediction": (
                MLPredictionResponse(
                    **ml_prediction
                )
                if ml_prediction
                else None
            ),

            "analysis": AnalysisResponse(
                **analysis
            ),

            "sources": result.get(
                "sources",
                [],
            ),
        }

    except HTTPException:
        raise

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        print(
            f"FinSight error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred "
                "while processing the request."
            ),
        )