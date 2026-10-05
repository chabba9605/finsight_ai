from components.state import FinSightState
from rag.retriever import get_financial_news


def retrieval_agent(
    state: FinSightState,
) -> FinSightState:
    """
    Retrieve recent financial/news information using Tavily.
    """

    ticker = state["ticker"]

    context, sources = get_financial_news(
        ticker
    )

    return {
        "retrieved_context": context,
        "sources": sources,
    }