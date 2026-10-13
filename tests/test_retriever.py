
import importlib


def load_retriever(monkeypatch):
    # Use a dummy API key; no real API requests are made.
    monkeypatch.setenv("TAVILY_API_KEY", "test-api-key")
    return importlib.import_module("rag.retriever")


def test_duplicate_sources(monkeypatch):
    retriever = load_retriever(monkeypatch)

    mock_results = {
        "results": [
            {
                "title": "NVIDIA Earnings Report",
                "url": "https://example.com/nvda",
                "content": "NVIDIA released earnings.",
            },
            {
                "title": "NVIDIA Earnings Report",
                "url": "https://example.com/nvda?utm_source=test",
                "content": "Duplicate article.",
            },
        ]
    }

    monkeypatch.setattr(
        retriever.tavily,
        "search",
        lambda **kwargs: mock_results,
    )

    context, sources = retriever.get_financial_news("NVDA")

    assert len(sources) == 1
    assert "[Source 1]" in context
    assert "[Source 2]" not in context


def test_blocked_domains(monkeypatch):
    retriever = load_retriever(monkeypatch)

    mock_results = {
        "results": [
            {
                "title": "Reddit NVDA Discussion",
                "url": "https://www.reddit.com/r/stocks/test",
                "content": "Stock discussion.",
            },
            {
                "title": "NVIDIA News",
                "url": "https://example.com/news",
                "content": "Recent NVIDIA news.",
            },
        ]
    }

    monkeypatch.setattr(
        retriever.tavily,
        "search",
        lambda **kwargs: mock_results,
    )

    context, sources = retriever.get_financial_news("NVDA")

    assert len(sources) == 1
    assert sources[0] == "https://example.com/news"
    assert "Reddit" not in context


def test_no_usable_sources(monkeypatch):
    retriever = load_retriever(monkeypatch)

    monkeypatch.setattr(
        retriever.tavily,
        "search",
        lambda **kwargs: {"results": []},
    )

    context, sources = retriever.get_financial_news("NVDA")

    assert sources == []
    assert "No usable recent financial news" in context


def test_ticker_normalization(monkeypatch):
    retriever = load_retriever(monkeypatch)
    captured = {}

    def mock_search(**kwargs):
        captured.update(kwargs)
        return {"results": []}

    monkeypatch.setattr(
        retriever.tavily,
        "search",
        mock_search,
    )

    retriever.get_financial_news(" nvda ")

    assert "NVDA" in captured["query"]
