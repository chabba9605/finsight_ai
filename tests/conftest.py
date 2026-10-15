import pytest
from rate_limit import limiter


@pytest.fixture(autouse=True)
def reset_rate_limits():
    limiter._storage.reset()
    yield
    limiter._storage.reset()