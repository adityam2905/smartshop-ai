"""
The real-data collection / evaluation scripts, exercised on demo listings
standing in for saved SerpAPI results (no network, no torch).
"""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.collect_real_listings import KEPT_FIELDS, label_rows, slugify, trim_result
from experiments.evaluate_real import (
    featurize, load_labels, parse_label, policies, score_decisions, wilson_interval,
)
from smartshop.mock_data import fetch_mock_results


@pytest.fixture
def real_data(tmp_path):
    """A saved 'search' plus a labelling sheet, laid out like real_data/."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    query, country = "Sony Headphones", "in"
    results = [trim_result(r) for r in fetch_mock_results(query)]
    (raw_dir / f"{slugify(query, country)}.json").write_text(
        json.dumps({"query": query, "country": country, "results": results}), encoding="utf-8")

    rows = pd.DataFrame(label_rows(query, country, results, per_query=6))
    # Amazon, Best Buy, Walmart: good deals · ultra-deals99 / cheapbuy: scams ·
    # B&H: left blank to check it's excluded
    rows["legit"]     = ["1", "1", "yes", "0", "",  "0"]
    rows["good_deal"] = ["1", "1", "0",   "",  "",  ""]
    labels_csv = tmp_path / "labels.csv"
    rows.to_csv(labels_csv, index=False)
    return str(labels_csv), str(raw_dir)


# ── Collector ────────────────────────────────────────────────────────────────

def test_trim_result_drops_anything_that_could_carry_the_api_key():
    raw = {"title": "x", "extracted_price": 1.0, "serpapi_product_api": "https://serpapi.com/...&api_key=SECRET"}
    trimmed = trim_result(raw)
    assert set(trimmed) <= set(KEPT_FIELDS)
    assert "SECRET" not in json.dumps(trimmed)


def test_label_rows_leave_labels_blank_and_hide_model_outputs():
    rows = label_rows("iPhone 15", "in", fetch_mock_results("iPhone 15"), per_query=3)
    assert len(rows) == 3
    assert all(r["legit"] == "" and r["good_deal"] == "" for r in rows)
    assert not any(k in rows[0] for k in ("site_trust_score", "action", "normalized_price"))
    assert rows[1]["id"] == "in_iphone-15_1" and rows[1]["position"] == 1


# ── Label parsing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, expected", [("1", 1.0), ("yes", 1.0), (" Y ", 1.0), ("0", 0.0),
                                             ("no", 0.0), ("1.0", 1.0)])
def test_parse_label(value, expected):
    assert parse_label(value) == expected


@pytest.mark.parametrize("value", ["", "?", "maybe", "nan"])
def test_parse_label_unknown_is_nan(value):
    assert np.isnan(parse_label(value))


def test_load_labels_keeps_only_decidable_rows(real_data):
    labels_csv, _ = real_data
    df, skipped = load_labels(labels_csv)
    assert len(df) == 5                                  # the blank row is out
    assert skipped == {"total_rows": 6, "no_seller": 0, "unlabelled": 1, "legit_without_good_deal": 0}
    assert df["should_recommend"].tolist() == [True, True, False, False, False]
    assert df["is_scam"].tolist() == [False, False, False, True, True]


def test_featurize_matches_the_app_pipeline(real_data):
    labels_csv, raw_dir = real_data
    df = featurize(load_labels(labels_csv)[0], raw_dir=raw_dir)
    amazon = df.iloc[0]
    assert amazon["site_trust_score"] >= 0.9
    assert amazon["normalized_price"] == pytest.approx(278 / 349.99, abs=1e-3)   # own list price
    assert df.iloc[4]["site_trust_score"] < 0.3                                   # cheapbuy-store.xyz
    assert (df["user_preference_score"] == 0.5).all()                             # neutral


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_score_decisions():
    should = [1, 1, 0, 0, 0]
    rec    = [1, 0, 1, 0, 0]
    scam   = [0, 0, 1, 1, 0]
    m = score_decisions(should, rec, scam)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1 / 2)
    assert m["scams_recommended"] == 1 and m["n_scams"] == 2


def test_wilson_interval_brackets_the_estimate_and_widens_for_small_n():
    lo, hi = wilson_interval(45, 50)
    assert lo < 0.9 < hi
    small = wilson_interval(9, 10)
    assert (small[1] - small[0]) > (hi - lo)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_baseline_policies_on_demo_listings(real_data):
    labels_csv, raw_dir = real_data
    df = featurize(load_labels(labels_csv)[0], raw_dir=raw_dir)
    obs = [np.array([r[k] for k in ("normalized_price", "discount_percentage",
                                    "site_trust_score", "user_preference_score")]) for _, r in df.iterrows()]
    rule = policies()["Trust rule only"]
    # The trust rule blocks cheapbuy-store.xyz but not ultra-deals99.net (trust ~0.32)
    assert [rule(o) for o in obs] == [True, True, True, True, False]
