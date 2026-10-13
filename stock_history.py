"""Historical daily closing prices for the FinSight AI frontend."""
from typing import Literal

import math
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/stock", tags=["Stock history"])

Period = Literal["1mo", "3mo", "6mo", "1y"]


class PricePoint(BaseModel):
    date: str
    close: float


class HistoryResponse(BaseModel):
    ticker: str
    period: Period
    currency: str
    prices: list[PricePoint]


@router.get("/{ticker}/history", response_model=HistoryResponse)
def stock_history(ticker: str, period: Period = "1mo"):
    symbol = ticker.strip().upper()
    if not symbol or len(symbol) > 12 or not all(c.isalnum() or c in '.-^' for c in symbol):
        raise HTTPException(status_code=400, detail="Invalid stock ticker")

    try:
        stock = yf.Ticker(symbol)
        frame = stock.history(period=period, interval="1d", auto_adjust=True)
    except Exception as exc:
        print(f"History download error for {symbol}: {exc}")
        raise HTTPException(status_code=502, detail="Market data provider unavailable") from exc

    if frame is None or frame.empty or "Close" not in frame.columns:
        raise HTTPException(status_code=404, detail=f"No historical prices found for {symbol}")

    prices = []
    for index, price in frame["Close"].items():
        if pd.notna(price) and math.isfinite(float(price)):
            prices.append({"date": index.strftime("%Y-%m-%d"), "close": round(float(price), 4)})

    if not prices:
        raise HTTPException(status_code=404, detail=f"No historical prices found for {symbol}")

    # Most supported US equity tickers trade in USD. The currency is obtained
    # from Yahoo's quote metadata when available, not assumed.
    try:
        currency = stock.fast_info.get("currency") or ""
    except Exception:
        currency = ""

    return {"ticker": symbol, "period": period, "currency": currency, "prices": prices}
