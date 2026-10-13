
import os
from datetime import datetime, timezone
from typing import List, Tuple
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in .env")

tavily = TavilyClient(api_key=TAVILY_API_KEY)


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())

    tracking_params = {"fbclid", "gclid", "mc_cid", "mc_eid"}

    query = urlencode([
        (key, value)
        for key, value in parse_qsl(parts.query)
        if not key.lower().startswith("utm_")
        and key.lower() not in tracking_params
    ])

    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path.rstrip("/") or "/",
        query,
        ""
    ))


def get_financial_news(ticker: str) -> Tuple[str, List[str]]:
    ticker = ticker.upper().strip()

    results = tavily.search(
        query=(
            f"{ticker} stock latest earnings financial news "
            "company developments analyst outlook"
        ),
        topic="news",
        days=7,
        max_results=20,
        search_depth="advanced",
    )

    articles = results.get("results", [])

    context_parts = []
    sources = []
    seen_urls = set()
    seen_titles = set()

    blocked_domains = {
        "youtube.com",
        "youtu.be",
        "stocktwits.com",
        "reddit.com",
        "facebook.com",
        "fb.com",
        "instagram.com",
        "tiktok.com",
        "x.com",
        "twitter.com",
    }

    for article in articles:
        url = (article.get("url") or "").strip()
        title = (article.get("title") or "").strip()
        content = (article.get("content") or "").strip()

        if not url or not title or not content:
            continue

        # Filter unwanted domains
        domain = urlsplit(url).netloc.lower()
        domain = domain.removeprefix("www.")

        if any(
            domain == blocked or domain.endswith("." + blocked)
            for blocked in blocked_domains
        ):
            continue

        # Remove duplicate URLs and titles
        normalized_url = normalize_url(url)
        normalized_title = " ".join(title.lower().split())

        if (
            normalized_url in seen_urls
            or normalized_title in seen_titles
        ):
            continue

        seen_urls.add(normalized_url)
        seen_titles.add(normalized_title)

        published_date = (
            article.get("published_date")
            or article.get("date")
            or "Not provided"
        )

        source_number = len(sources) + 1
        sources.append(url)

        context_parts.append(
            f"[Source {source_number}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Published: {published_date}\n"
            f"Content: {content}"
        )

        if len(sources) >= 5:
            break

    if not context_parts:
        return (
            "No usable recent financial news sources were found. "
            "Do not invent news or citations.",
            []
        )

    retrieved_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    return (
        f"News retrieved at: {retrieved_at}\n\n"
        + "\n\n".join(context_parts),
        sources
    )
