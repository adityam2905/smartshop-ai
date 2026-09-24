import pytest

import scraper
from scraper import (
    compute_domain_trust,
    extract_features,
    match_mock_set,
    search_products,
    search_products_detailed,
    update_user_preference,
    get_user_preference,
)


# ── Domain trust heuristic ──────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://www.amazon.com/dp/XXXX",
    "https://www.bestbuy.com/site/1",
    "https://www.walmart.com/ip/1",
])
def test_known_trusted_domains_score_high(url):
    assert compute_domain_trust(url) >= 0.9


@pytest.mark.parametrize("url", [
    "https://cheapbuy-store.xyz/deal",
    "https://bargain-hunt.gq/x",
    "https://discount-mega.ru/deal",
    "https://shop-fast-now.biz/x",
])
def test_scam_tld_domains_score_below_threshold(url):
    # 0.3 is the SCAM_TRUST_THRESHOLD used by ShoppingEnv / app.py
    assert compute_domain_trust(url) < 0.3


@pytest.mark.parametrize("url", [
    "https://www.amazon.in/dp/XXXX",
    "https://www.amazon.co.uk/dp/XXXX",
    "https://www.flipkart.com/p/1",
    "https://smile.amazon.com/dp/XXXX",     # real subdomain of a known retailer
])
def test_regional_and_subdomain_retailers_score_high(url):
    assert compute_domain_trust(url) >= 0.9


@pytest.mark.parametrize("url", [
    "https://cheap-amazon.com/x",           # used to score 0.999 via endswith("amazon.com")
    "https://notamazon.com/x",
    "https://amazon-deals.net/x",
    "https://amaz0n.com/x",                 # digit homoglyph
    "https://flipkartsale.shop/x",
    "https://bestbuy-outlet.com/x",
])
def test_brand_lookalike_domains_score_below_threshold(url):
    assert compute_domain_trust(url) < 0.3


@pytest.mark.parametrize("url", [
    "https://pineapple-store.com/x",        # contains "apple", but not as a brand token
    "https://wishlist-gifts.com/x",         # contains "wish"
    "https://nikesh-books.in/x",            # contains "nike"
])
def test_ordinary_words_containing_short_brand_names_are_not_flagged(url):
    assert compute_domain_trust(url) >= 0.45


def test_dot_net_scam_domain_is_not_reliably_flagged():
    """
    Documents a real gap rather than papering over it: data_generator.py's
    SCAM_DOMAINS list includes "ultra-deals99.net" and "deal-xpress.top",
    but SCAM_TLDS in this file does not include ".net" (only ".top" is
    covered). At training time, data_generator.py assigns scam rows a
    trust score directly (always < 0.28, via compute_site_trust_score),
    independent of this heuristic — so the DQN never sees this gap. Live
    inference goes through compute_domain_trust() instead, which CAN place
    a ".net" scam-styled domain at or above the 0.3 threshold depending on
    keyword-hit jitter, i.e. the live scam filter is not guaranteed to
    catch every domain shape the offline model was implicitly trained
    against. See README.md's "Limitations" section.
    """
    score = compute_domain_trust("https://ultra-deals99.net/product/1")
    assert 0.0 <= score <= 1.0  # sanity bound only — deliberately not asserting < 0.3


def test_domain_trust_is_deterministic_for_the_same_domain():
    url = "https://some-random-shop.com/item"
    assert compute_domain_trust(url) == compute_domain_trust(url)


def test_domain_trust_handles_missing_or_malformed_url():
    assert 0.0 <= compute_domain_trust("") <= 1.0
    assert 0.0 <= compute_domain_trust("not a url") <= 1.0


# ── Feature engineering ──────────────────────────────────────────────────────

def test_extract_features_computes_discount_from_old_price():
    item = {
        "title": "Widget",
        "extracted_price": 50.0,
        "old_price": 100.0,
        "link": "https://www.amazon.com/dp/1",
        "category": "Electronics",
    }
    feat = extract_features(item, market_avg_price=100.0)
    assert feat["discount_percentage"] == pytest.approx(0.5)
    assert feat["normalized_price"] == pytest.approx(0.5)
    assert 0.0 <= feat["site_trust_score"] <= 1.0


def test_extract_features_infers_discount_without_old_price():
    item = {"title": "Widget", "extracted_price": 50.0, "link": "https://www.amazon.com/dp/1"}
    feat = extract_features(item, market_avg_price=100.0)
    # No old_price given → discount inferred from normalized price.
    assert feat["discount_percentage"] == pytest.approx(0.5)


def test_extract_features_handles_string_prices():
    item = {"title": "Widget", "price": "$49.99", "link": "https://www.walmart.com/x"}
    feat = extract_features(item)
    assert feat["price"] == pytest.approx(49.99)


def test_extract_features_defaults_missing_price_to_zero():
    item = {"title": "Mystery Item", "link": "https://www.amazon.com/x"}
    feat = extract_features(item)
    assert feat["price"] == pytest.approx(0.0)


# ── User preference store (regression coverage for the multi-user bug) ──────

def test_user_preferences_do_not_leak_between_independent_stores():
    """
    Regression test: two independent preference dicts (standing in for two
    Streamlit sessions) must never see each other's updates. Prior to the
    fix, preferences lived in a single module-level dict shared by every
    user of a deployed app.
    """
    session_a_prefs: dict = {}
    session_b_prefs: dict = {}

    update_user_preference("Electronics", +0.3, session_a_prefs)
    update_user_preference("Electronics", -0.2, session_b_prefs)

    assert get_user_preference("Electronics", session_a_prefs) == pytest.approx(0.8)
    assert get_user_preference("Electronics", session_b_prefs) == pytest.approx(0.3)
    assert session_a_prefs != session_b_prefs


def test_update_user_preference_clips_to_unit_interval():
    prefs: dict = {}
    for _ in range(20):
        update_user_preference("Toys", +0.5, prefs)
    assert get_user_preference("Toys", prefs) == pytest.approx(1.0)

    for _ in range(20):
        update_user_preference("Toys", -0.5, prefs)
    assert get_user_preference("Toys", prefs) == pytest.approx(0.0)


def test_unknown_category_defaults_to_neutral_preference():
    assert get_user_preference("Some Category Nobody Rated", {}) == pytest.approx(0.5)


def test_extract_features_uses_the_prefs_dict_passed_in():
    prefs = {"Electronics": 0.9}
    item = {"title": "Widget", "price": 10.0, "category": "Electronics", "link": "https://amazon.com/1"}
    feat = extract_features(item, user_prefs=prefs)
    assert feat["user_preference_score"] == pytest.approx(0.9)

    # A second, untouched store must still see the neutral default.
    other_prefs: dict = {}
    feat2 = extract_features(item, user_prefs=other_prefs)
    assert feat2["user_preference_score"] == pytest.approx(0.5)


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
    monkeypatch.setattr(scraper, "SERPAPI_AVAILABLE", True)
    monkeypatch.setattr(scraper, "SERPAPI_KEY", "secret-key-123")
    monkeypatch.setattr(scraper, "GoogleSearch", _FakeGoogleSearch, raising=False)
    _FakeGoogleSearch.response, _FakeGoogleSearch.raises = {}, None
    return _FakeGoogleSearch


def test_missing_key_falls_back_with_a_reason(monkeypatch):
    monkeypatch.setattr(scraper, "SERPAPI_AVAILABLE", True)
    monkeypatch.setattr(scraper, "SERPAPI_KEY", "")
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
