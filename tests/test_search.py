"""The search pipeline: demo listings, live-search fallbacks (SerpAPI faked, no network)."""

import pytest

from smartshop import search as search_module
from smartshop.mock_data import match_mock_set
from smartshop.search import search_products, search_products_detailed


# ── search_products (mock mode) ──────────────────────────────────────────────

def test_search_products_mock_mode_returns_well_formed_features():
    results = search_products("Sony Headphones", use_mock=True)
    assert len(results) > 0
    for feat in results:
        assert 0.0 <= feat["normalized_price"] <= 2.0
        assert 0.0 <= feat["discount_percentage"] <= 1.0
        assert 0.0 <= feat["site_trust_score"] <= 1.0
        assert 0.0 <= feat["user_preference_score"] <= 1.0


@pytest.mark.parametrize("query, expected_set", [
    ("iPhone 15", "iPhone 15"),
    ("iphone 15 pro max", "iPhone 15"),
    ("MacBook Pro", "MacBook Pro"),
    ("cheap laptop", "MacBook Pro"),
    ("Gaming Chair", "Gaming Chair"),
    ("office chair", "Gaming Chair"),
    ("sony wh-1000xm5", "Sony Headphones"),
    ("running sneakers", "Nike Shoes"),
    ("garden hose", None),
])
def test_mock_queries_route_to_the_right_listing_set(query, expected_set):
    assert match_mock_set(query) == expected_set


@pytest.mark.parametrize("query", ["Sony Headphones", "Nike Shoes", "iPhone 15", "MacBook Pro", "Gaming Chair"])
def test_every_suggestion_chip_has_realistic_mock_listings(query):
    """Each chip in app.py should get its own listings, with both a trusted
    retailer and a listing the trust filter blocks, not the generic default."""
    search = search_products_detailed(query, use_mock=True)
    assert search["mock_matched"]
    trust = [f["site_trust_score"] for f in search["results"]]
    assert max(trust) >= 0.9
    assert min(trust) < 0.3


def test_unmatched_mock_query_is_reported_as_generic():
    search = search_products_detailed("garden hose", use_mock=True)
    assert search["source"] == "mock"
    assert not search["mock_matched"]
    assert search["fallback_reason"] is None     # mock was requested, not a failure


# ── Live search fallbacks (SerpAPI faked, no network) ────────────────────────

class _FakeGoogleSearch:
    response: dict = {}
    raises: Exception | None = None
    last_params: dict = {}

    def __init__(self, params):
        type(self).last_params = params

    def get_dict(self):
        if self.raises:
            raise self.raises
        return self.response


@pytest.fixture
def fake_serpapi(monkeypatch):
    monkeypatch.setattr(search_module, "SERPAPI_AVAILABLE", True)
    monkeypatch.setattr(search_module, "SERPAPI_KEY", "secret-key-123")
    monkeypatch.setattr(search_module, "GoogleSearch", _FakeGoogleSearch, raising=False)
    _FakeGoogleSearch.response, _FakeGoogleSearch.raises = {}, None
    return _FakeGoogleSearch


def test_missing_key_falls_back_with_a_reason(monkeypatch):
    monkeypatch.setattr(search_module, "SERPAPI_AVAILABLE", True)
    monkeypatch.setattr(search_module, "SERPAPI_KEY", "")
    search = search_products_detailed("iPhone 15")
    assert search["source"] == "mock"
    assert "SERPAPI_KEY" in search["fallback_reason"]


def test_serpapi_error_response_is_surfaced(fake_serpapi):
    """Quota exhaustion comes back as an 'error' field, not an exception."""
    fake_serpapi.response = {"error": "Your account has run out of searches."}
    search = search_products_detailed("iPhone 15")
    assert search["source"] == "mock"
    assert "run out of searches" in search["fallback_reason"]


def test_request_failure_reason_never_leaks_the_api_key(fake_serpapi):
    fake_serpapi.raises = ConnectionError("Max retries exceeded with url: /search?api_key=secret-key-123&q=x")
    search = search_products_detailed("iPhone 15")
    assert search["source"] == "mock"
    assert "secret-key-123" not in search["fallback_reason"]
    assert "***" in search["fallback_reason"]


def test_live_search_uses_the_selected_country_and_currency(fake_serpapi):
    fake_serpapi.response = {"shopping_results": [
        {"title": "Phone", "extracted_price": 59999.0, "link": "https://www.flipkart.com/p/1", "source": "Flipkart"},
    ]}
    search = search_products_detailed("iPhone 15", country="in")
    assert fake_serpapi.last_params["gl"] == "in"
    assert search["source"] == "live"
    assert search["fallback_reason"] is None
    assert search["results"][0]["currency"] == "₹"


def test_search_products_forwards_user_prefs(monkeypatch):
    prefs = {"Electronics": 0.77}
    results = search_products("Sony Headphones", use_mock=True, user_prefs=prefs)
    electronics_items = [r for r in results if r["category"] == "Electronics"]
    assert electronics_items
    for feat in electronics_items:
        assert feat["user_preference_score"] == pytest.approx(0.77)
