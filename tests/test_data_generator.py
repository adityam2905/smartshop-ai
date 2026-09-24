import numpy as np
import pytest

from smartshop.data_generator import generate_dataset


@pytest.fixture(scope="module")
def df():
    return generate_dataset(n=4000)


def test_discount_alone_does_not_separate_scam_from_legit(df):
    """
    Regression test: legit and scam discounts used to be non-overlapping
    (5–40% vs 60–95%), so the agent learned "small discount = safe" and
    skipped real 65%+ deals from trusted retailers. The best discount-only
    threshold must now be little better than always guessing "legit".
    """
    base_rate = 1 - df["is_scam"].mean()
    best = max(
        ((df["discount_percentage"] >= t) == df["is_scam"]).mean()
        for t in np.linspace(0, 1, 201)
    )
    assert best < base_rate + 0.10


def test_legit_listings_include_big_discounts_and_scams_include_modest_ones(df):
    legit = df[~df["is_scam"]]
    scam = df[df["is_scam"]]
    assert (legit["discount_percentage"] > 0.65).mean() > 0.05
    assert (scam["discount_percentage"] < 0.40).mean() > 0.05


def test_legit_listings_cover_the_mid_trust_range_the_live_scorer_produces(df):
    # compute_domain_trust() scores unknown-but-clean shops around 0.45–0.65
    legit = df[~df["is_scam"]]
    assert ((legit["site_trust_score"] >= 0.40) & (legit["site_trust_score"] < 0.60)).mean() > 0.05


def test_trust_alone_no_longer_separates_the_classes(df):
    """
    Polished scam shops sit in the same trust range as small legit shops, so
    the trust < 0.3 hard rule misses some scams and the agent has to use price.
    """
    scam = df[df["is_scam"]]
    assert (scam["site_trust_score"] >= 0.3).mean() > 0.25
    best = max(
        ((df["site_trust_score"] < t) == df["is_scam"]).mean()
        for t in np.linspace(0, 1, 201)
    )
    assert best < 0.95


def test_mid_trust_scams_are_given_away_by_an_implausible_price(df):
    mid_trust = df[(df["site_trust_score"] >= 0.3) & (df["site_trust_score"] <= 0.6)]
    scam_price = mid_trust.loc[mid_trust["is_scam"], "normalized_price"]
    legit_price = mid_trust.loc[~mid_trust["is_scam"], "normalized_price"]
    assert scam_price.median() < 0.4 < legit_price.median()


def test_some_legit_listings_are_overpriced(df):
    """Needed for the agent to learn deal quality, not just scam avoidance."""
    legit = df[~df["is_scam"]]
    assert (legit["normalized_price"] > 1.0).mean() > 0.10
