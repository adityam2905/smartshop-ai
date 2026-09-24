"""
app.py behaviour: escaping of third-party listing data, and SerpAPI quota
protection.

Needs streamlit + torch (importing app.py loads the model stack), so it's
skipped under the lean requirements-test.txt; CI's test-full job runs it.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("stable_baselines3")

import app  # noqa: E402
from app import safe_text, safe_url  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_safe_text_escapes_markup():
    assert safe_text('<img src=x onerror="alert(1)">') == "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;"


@pytest.mark.parametrize("url", [
    "javascript:alert(1)",
    "JavaScript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "",
    "not a url",
])
def test_safe_url_rejects_non_http_links(url):
    assert safe_url(url) == ""


def test_safe_url_keeps_http_links_and_escapes_quotes():
    assert safe_url("https://www.amazon.com/dp/1") == "https://www.amazon.com/dp/1"
    assert safe_url('https://x.com/"onmouseover="alert(1)') == "https://x.com/&quot;onmouseover=&quot;alert(1)"


# ── SerpAPI quota protection ────────────────────────────────────────────────

@pytest.fixture
def serpapi_calls(monkeypatch):
    """Replaces the real SerpAPI fetch; returns the list of calls made."""
    calls = []
    outcome = {"reason": None}

    def fake_fetch(query, num_results, country):
        calls.append(query)
        return [{"title": query, "extracted_price": 10, "source": "Amazon"}], outcome["reason"]

    monkeypatch.setattr(app, "fetch_serpapi_results", fake_fetch)
    app._cached_serpapi_fetch.clear()
    yield calls, outcome
    app._cached_serpapi_fetch.clear()


def test_repeat_live_searches_are_served_from_cache(serpapi_calls):
    calls, _ = serpapi_calls
    for query in ("iPhone 15", "iPhone 15", "  iphone 15 "):
        results, reason = app.fetch_live_cached(query, 12, "in")
        assert reason is None and results
    assert len(calls) == 1


def test_failed_live_searches_are_not_cached(serpapi_calls):
    """One network blip mustn't pin demo data in place for the cache's 6-hour TTL."""
    calls, outcome = serpapi_calls
    outcome["reason"] = "SerpAPI request failed: timeout"
    assert app.fetch_live_cached("iPhone 15", 12, "in")[1] == outcome["reason"]
    outcome["reason"] = None
    assert app.fetch_live_cached("iPhone 15", 12, "in")[1] is None
    assert len(calls) == 2


def test_session_live_search_limit_falls_back_to_demo_data_with_a_reason():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=120).run()
    at.checkbox(key="use_mock").uncheck().run()
    at.session_state.live_queries_used = {(f"query {i}", "us") for i in range(app.MAX_LIVE_SEARCHES_PER_SESSION)}
    at.button(key="chip_2").click().run()

    assert at.session_state.search_source == "mock"
    assert "live searches" in at.session_state.fallback_reason
    assert any("Live search unavailable" in w.value for w in at.warning)
