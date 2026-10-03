from fastapi import FastAPI
from graph import pipeline

app = FastAPI()


@app.get("/")
def home():
    return {"data": "FinSight AI is running"}


@app.get("/analyse/{ticker}")
def analyse(ticker: str, question: str = "Should I buy this week?"):

    result = pipeline.invoke({
    "ticker": ticker,
    "question": question,
    "market_data": None,
    "retrieved_context": None,
    "sources": None,
    "analysis": None,
    "ml_score": None
})

    return {
        "ticker": result["ticker"],
        "question": result["question"],
        "market_data": result["market_data"],
        "analysis": result["analysis"],
        "sources": result["sources"]
    }

    return result