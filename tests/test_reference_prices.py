"""reference_prices.py with faked store lists — no network, no SerpAPI searches."""

import pytest

from reference_prices import _find_listing, _product_ids, fetch_reference_prices, reference_from_stores
from scraper import estimate_market_references

STORES = [
    {"name": "Amazon.in", "price": 29990.0, "link": "https://www.amazon.in/dp/1"},
    {"name": "Flipkart",  "price": 29990.0, "link": "https://www.flipkart.com/p/1"},
    {"name": "Myntra",    "price": 27990.0, "link": "https://www.myntra.com/1"},
    {"name": "cheapbuy-store.xyz", "price": 2999.0, "link": "https://cheapbuy-store.xyz/1"},
]


def test_reference_is_the_median_of_other_non_blocked_stores():
    ref, n = reference_from_stores(STORES, own_seller="AJIO.com")
    assert n == 3                                  # the .xyz scam store is ignored
    assert ref == pytest.approx(29990.0)


def test_a_listing_never_sets_its_own_reference():
    ref, n = reference_from_stores(STORES, own_seller="Myntra")
    assert n == 2 and ref == pytest.approx(29990.0)


def test_no_other_stores_means_no_reference():
    assert reference_from_stores(STORES[:1], own_seller="Amazon.in") == (None, 0)


def test_reference_price_takes_priority_over_in_search_estimates():
    raw = [
        {"title": "Sony WH-1000XM5", "extracted_price": 23664, "old_price": "₹34,990", "source": "AJIO.com"},
        {"title": "Sony WH-1000XM5", "extracted_price": 29680, "source": "Amazon.in"},
    ]
    refs = estimate_market_references(raw, reference_prices=[29990.0, None])
    assert refs[0] == (29990.0, "other stores, same product")    # not the ₹34,990 "MRP"
    assert refs[1][1] != "other stores, same product"


def test_app_lookups_are_capped_and_skip_blocked_sellers():
    calls = []

    def fake_fetch(token):
        calls.append(token)
        return STORES, None

    raw = [
        {"source": "cheapbuy-store.xyz", "link": "https://cheapbuy-store.xyz/a", "immersive_product_page_token": "t0"},
        {"source": "Flipkart", "immersive_product_page_token": "t1"},
        {"source": "Amazon.in", "immersive_product_page_token": "t2"},
        {"source": "Myntra", "immersive_product_page_token": "t3"},
        {"source": "Croma"},                                             # no token
    ]
    refs = fetch_reference_prices(raw, max_products=2, fetch=fake_fetch)
    assert calls == ["t1", "t2"]
    assert refs[0] is None and refs[1] is not None and refs[3] is None


def test_listing_is_found_again_by_any_shared_google_id():
    saved = {"title": "Sony WH-1000XM5", "source": "Amazon.in",
             "product_link": "https://www.google.com/search?prds=catalogid:111,productid:222,gpcid:333"}
    fresh = [
        {"title": "Other", "product_id": "999", "immersive_product_page_token": "x"},
        {"title": "Sony WH-1000XM5 Headphones", "product_id": "333", "immersive_product_page_token": "y"},
    ]
    assert _product_ids(saved) == {"111", "222", "333"}
    assert _find_listing(saved, "Amazon.in", fresh)["immersive_product_page_token"] == "y"
