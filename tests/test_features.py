"""Turning a listing into model inputs, and per-session category preferences."""

import numpy as np
import pytest

from smartshop.features import extract_features, features_to_obs, get_user_preference, update_user_preference


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


def test_extracted_old_price_is_used_for_the_discount():
    feat = extract_features({"title": "x", "extracted_price": 60.0, "extracted_old_price": 100.0,
                             "source": "Walmart"}, market_avg_price=60.0)
    assert feat["discount_percentage"] == pytest.approx(0.4)



# ── Observation vector ───────────────────────────────────────────────────────

def test_features_to_obs_clips_out_of_range_values():
    obs = features_to_obs({
        "normalized_price": 5.0,      # should clip to 2.0
        "discount_percentage": -1.0,  # should clip to 0.0
        "site_trust_score": 2.0,      # should clip to 1.0
        "user_preference_score": 0.5,
    })
    assert obs == pytest.approx(np.array([2.0, 0.0, 1.0, 0.5]))


def test_features_to_obs_defaults_missing_keys():
    obs = features_to_obs({})
    assert obs == pytest.approx(np.array([1.0, 0.0, 0.5, 0.5]))
