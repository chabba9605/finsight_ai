
"""Shared rate limiter for AI analysis endpoints."""

from slowapi import Limiter
from slowapi.util import get_remote_address


limiter = Limiter(
    key_func=get_remote_address,
)

analysis_limit = limiter.shared_limit(
    "5/minute",
    scope="finsight-ai-analysis",
)
