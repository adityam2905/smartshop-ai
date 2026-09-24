"""The usual-price estimate: price parsing, same-product matching, market references."""

import numpy as np
import pytest

from smartshop.pricing import (
    _LURE_RATIO, estimate_market_prices, estimate_market_references, parse_old_price, same_product, title_profile,
)


@pytest.mark.parametrize("item, expected", [
    # Real SerpAPI India results: extracted_old_price is the percentage here
    ({"old_price": "32% off₹34,990", "extracted_old_price": 32}, 34990.0),
    ({"old_price": "Usually ₹44,410", "extracted_old_price": 44410}, 44410.0),
    ({"old_price": "Was ₹768"}, 768.0),
    ({"old_price": "₹1,10,990"}, 110990.0),          # Indian digit grouping
    ({"old_price": "$349.99"}, 349.99),
    ({"extracted_old_price": 100.0}, 100.0),
    ({"old_price": 70.0}, 70.0),
    ({}, 0.0),
])
def test_parse_old_price(item, expected):
    assert parse_old_price(item) == pytest.approx(expected)


# ── Market price reference ───────────────────────────────────────────────────

def test_scam_lures_do_not_drag_the_market_reference_down():
    """
    Regression test: the reference used to be the median of *all* prices, so
    a couple of scam lures made genuine retailers look overpriced.
    """
    raw = [
        {"extracted_price": 280, "source": "Amazon"},
        {"extracted_price": 300, "source": "Walmart"},
        {"extracted_price": 290, "source": "Best Buy"},
        {"extracted_price": 20,  "link": "https://cheapbuy-store.xyz/a"},
        {"extracted_price": 25,  "link": "https://discount-mega.ru/b"},
        {"extracted_price": 30,  "link": "https://bargain-hunt.gq/c"},
    ]
    market = estimate_market_prices(raw)
    assert market[0] == pytest.approx(295)          # the *other* trusted sellers: median(300, 290)
    assert market[3] == pytest.approx(290)          # lures are compared against trusted sellers only


def test_trusted_sellers_list_price_is_their_own_reference():
    raw = [
        {"extracted_price": 278, "extracted_old_price": 350, "source": "Amazon"},
        {"extracted_price": 80,  "old_price": 100, "source": "Walmart"},
        {"extracted_price": 40,  "old_price": 400, "link": "https://urbanstyle-outlet.com/x"},
    ]
    market = estimate_market_prices(raw)
    assert market[0] == pytest.approx(350)
    assert market[1] == pytest.approx(100)
    # An untrusted seller's "was" price is exactly what a scam would inflate
    assert market[2] == pytest.approx(np.median([278, 80]))


def test_market_reference_falls_back_to_other_sellers_of_the_same_product():
    raw = [{"extracted_price": p, "source": "Some Shop"} for p in (10, 20, 30)]
    assert estimate_market_prices(raw) == [25.0, 20.0, 15.0]


# ── Variant-aware market price ───────────────────────────────────────────────

@pytest.mark.parametrize("a, b, expected", [
    ("Apple iPhone 15", "iPhone 15 (128GB_Blue)", True),
    ("Apple iPhone 15", "Apple iPhone 15 Plus", False),                         # tier word
    ("Apple iPhone 15", "Refurbished Apple iPhone 15 by Cashify", False),       # condition
    ("Maybelline Fit Me Foundation 18 ml", "Maybelline Fit Me Foundation 30ml", False),  # volume
    ("Apple Watch Series 9 41mm GPS", "Apple Watch Series 9 45mm GPS", False),  # size
    ("iPhone 15 128GB", "iPhone 15 256 GB", False),                             # storage
    ("Sony WH-1000XM5 Headphones", "Sony Wireless Headphones with Mic-WH-1000XM5", True),  # model code
    ("Victus Laptop 15-fb3383AX", "Victus Laptop 15-fb3385AX", False),          # different code
    ("LEGO 42224 Technic Porsche", "LEGO 42213 Technic Ford Bronco", False),   # set numbers
    ("Prestige Svachh 3 Litre Pressure Cooker", "Prestige Svachh Aluminium Pressure Cooker 3L", True),
    ("Philips Steam Iron", "Ninja Foodi Air Fryer", False),                    # unrelated
    ("boAt Airdopes 141", "boAt Airdopes 131/138 TWS Earbuds", False),          # model number
    ("Apple Watch Series 11 GPS", "Apple Watch Series 9 GPS", False),           # generation
    ("Apple iPhone 15", "Apple iPhone 15 2023 edition", True),                  # extra number is fine
])
def test_same_product(a, b, expected):
    assert same_product(title_profile(a), title_profile(b)) is expected


def test_listings_are_compared_only_with_the_same_variant():
    """
    Regression test from the real-data evaluation: 18 ml bottles were compared
    against the median of the whole search (mostly 30 ml) and looked like
    50%-off deals.
    """
    raw = [
        {"title": "Maybelline Fit Me Foundation 30ml", "extracted_price": 376, "source": "Flipkart"},
        {"title": "Maybelline Fit Me Foundation 30ml", "extracted_price": 380, "source": "Amazon.in"},
        {"title": "Maybelline Fit Me Foundation 18 ml", "extracted_price": 181, "source": "Flipkart"},
        {"title": "Maybelline Fit Me Foundation 18 ml", "extracted_price": 185, "source": "Amazon.in"},
    ]
    refs = estimate_market_references(raw)
    assert refs[2][0] == pytest.approx(185)            # vs the other 18 ml, not the 30 ml
    assert refs[0][0] == pytest.approx(380)
    assert "similar trusted" in refs[2][1]


def test_listing_with_nothing_comparable_gets_no_price_signal():
    raw = [
        {"title": "HP Victus 15-fb3383AX", "extracted_price": 94999, "source": "Flipkart"},
        {"title": "HP Victus 15-fa2197TX", "extracted_price": 81495, "source": "Amazon.in"},
    ]
    refs = estimate_market_references(raw)
    assert refs[0] == (94999, "no comparable listing")   # → normalized_price 1.0


def test_a_replica_is_priced_against_the_product_it_claims_to_be():
    """The ₹1,849 'Rolex Submariner' from the real data matches real Submariners."""
    raw = [
        {"title": "Rolex Submariner 126610LN", "extracted_price": 1_736_000, "source": "Tata CLiQ"},
        {"title": "Rolex Submariner Date 126618LB", "extracted_price": 4_200_000, "source": "Tata CLiQ"},
        {"title": "Classy Rolex Submariner Watch For Men", "extracted_price": 1849, "source": "blinkartz"},
    ]
    ref, _ = estimate_market_references(raw)[2]
    assert 1849 / ref < 0.01


def test_a_lure_is_still_caught_with_nothing_comparable():
    raw = [
        {"title": "Sony Bravia 55 inch 4K TV", "extracted_price": 60000, "source": "Flipkart"},
        {"title": "Samsung Crystal 55 inch 4K TV", "extracted_price": 50000, "source": "Amazon.in"},
        {"title": "LG OLED evo C4", "extracted_price": 5000, "link": "https://techhaven-store.com/x"},
    ]
    ref, basis = estimate_market_references(raw)[2]
    assert basis == "far below everything in the search"
    assert 5000 / ref < _LURE_RATIO
