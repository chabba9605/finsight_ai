import yfinance as yf

from components.state import FinSightState
from ml.predict import predict_stock


def data_agent(
    state: FinSightState,
) -> FinSightState:
    """
    Fetch current market data and generate the ML prediction.

    The ML prediction is calculated here once and passed
    through the LangGraph state to the analysis agent.
    """

    ticker = state["ticker"].upper().strip()

    if not ticker:
        raise ValueError(
            "A stock ticker is required."
        )

    stock = yf.Ticker(
        ticker
    )

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

    first_price = float(
        history["Close"].iloc[0]
    )

    if first_price == 0:
        five_day_return = 0.0
    else:
        five_day_return = (
            (
                current_price
                - first_price
            )
            / first_price
        ) * 100

    # Run the ML model exactly once.
    ml_prediction = None

    try:
        ml_prediction = predict_stock(
            ticker
        )

    except Exception as e:
        # The financial research system should still be
        # able to operate if the ML model fails.
        print(
            f"ML prediction failed for {ticker}: {e}"
        )

    return {
        "ticker": ticker,

        "market_data": {
            "ticker": ticker,
            "current_price": current_price,
            "5d_return": float(
                five_day_return
            ),
        },

        "ml_prediction": ml_prediction,
    }