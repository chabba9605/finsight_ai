from components.state import FinSightState
from rag.retriever import get_financial_news


def retrieval_agent(
    state: FinSightState,
) -> FinSightState:
    '''
    latest earnings financial results
    latest company news
    analyst outlook
    '''

    ticker = state["ticker"]

    context, sources = get_financial_news(
        ticker
    )

    return {
        "retrieved_context": context,
        "sources": sources,
    }