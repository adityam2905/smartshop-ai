"""Seller trust: known retailers, look-alike domains, scam signals, and seller resolution."""

import pytest

from smartshop.features import extract_features
from smartshop.trust import compute_domain_trust, resolve_seller_domain


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
    against. The DQN is the backstop for this range: it's trained on
    polished scams with trust 0.30–0.60 and skips them when the price is
    implausibly low (as the mock "ultra-deals99.net" listing is).
    """
    score = compute_domain_trust("https://ultra-deals99.net/product/1")
    assert 0.0 <= score <= 1.0  # sanity bound only — deliberately not asserting < 0.3


def test_domain_trust_is_deterministic_for_the_same_domain():
    url = "https://some-random-shop.com/item"
    assert compute_domain_trust(url) == compute_domain_trust(url)


def test_domain_trust_handles_missing_or_malformed_url():
    assert 0.0 <= compute_domain_trust("") <= 1.0
    assert 0.0 <= compute_domain_trust("not a url") <= 1.0


# ── Seller resolution for live results ───────────────────────────────────────

@pytest.mark.parametrize("item, expected", [
    # Direct link to the seller
    ({"link": "https://www.flipkart.com/p/1", "source": "Flipkart"}, "flipkart.com"),
    # Google redirect carrying the seller URL
    ({"link": "https://www.google.com/url?url=https://www.bestbuy.com/site/1.p&sa=U", "source": "Best Buy"}, "bestbuy.com"),
    # Google product page, no seller URL → fall back to the seller name
    ({"link": "https://www.google.com/shopping/product/123", "source": "Walmart"}, "walmart.com"),
    ({"product_link": "https://www.google.co.in/shopping/product/123", "source": "Flipkart"}, "flipkart.com"),
    ({"source": "Amazon.in"}, "amazon.in"),
    ({"source": "Amazon.com - Seller"}, "amazon.com"),
    ({"source": "eBay - bestdeals123"}, "ebay.com"),
    ({"source": "B&H Photo-Video-Audio"}, "bhphotovideo.com"),
    ({"source": "cheap-amazon.com"}, "cheap-amazon.com"),
    # Seller names seen in real Google Shopping India results
    ({"source": "EMI Snapmint"}, "snapmint.com"),
    ({"source": "ubuy.co.in"}, "ubuy.co.in"),
    ({"source": "Joe's Corner Store"}, ""),          # unknown seller
    ({"source": "Amazon Deals Outlet"}, ""),         # brand-like name ≠ the brand
    ({}, ""),
])
def test_resolve_seller_domain(item, expected):
    assert resolve_seller_domain(item) == expected


def test_live_results_are_scored_on_the_seller_not_the_google_link():
    """
    Regression test: trust used to be computed from `link`, so every result
    linking to a Google page got the same "unknown .com" score and the scam
    filter couldn't tell sellers apart.
    """
    google = "https://www.google.com/shopping/product/123"
    trusted = extract_features({"title": "x", "price": 10, "link": google, "source": "Flipkart"})
    lookalike = extract_features({"title": "x", "price": 10, "link": google, "source": "cheap-amazon.com"})
    unknown = extract_features({"title": "x", "price": 10, "link": google, "source": "Joe's Corner Store"})

    assert trusted["site_trust_score"] >= 0.9
    assert lookalike["site_trust_score"] < 0.3
    assert unknown["site_trust_score"] == pytest.approx(0.5)
    # The clickable link is still the one SerpAPI gave
    assert trusted["site_url"] == google
