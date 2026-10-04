import yfinance as yf

from ml.predict import predict_stock


def data_agent(state):

    ticker = state["ticker"].upper()

    stock = yf.Ticker(ticker)

    history = stock.history(
        period="5d"
    )

    if history.empty:
        raise ValueError(
            f"No market data found for {ticker}"
        )

    current_price = float(
        history["Close"].iloc[-1]
    )

    five_day_return = float(
        (
            (
                current_price
                - float(history["Close"].iloc[0])
            )
            / float(history["Close"].iloc[0])
        ) * 100
    )

    ml_prediction = predict_stock(ticker)

    return {
        "ticker": ticker,

        "market_data": {
            "ticker": ticker,
            "current_price": current_price,
            "5d_return": five_day_return
        },

        "ml_score": float(
            ml_prediction["score"]
        )
    }