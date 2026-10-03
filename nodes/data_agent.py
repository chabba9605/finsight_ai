import yfinance as yf
from components.state import FinSightState

def data_agent(state: FinSightState) -> FinSightState:
    ticker = state["ticker"]
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    state["market_data"] = {
        "ticker": ticker,
        "current_price": round(hist["Close"].iloc[-1], 2),
        "5d_return": round((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100, 2)
    }
    return state