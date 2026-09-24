"""
Evaluate the deployed decision pipeline on real, hand-labelled listings.

Reads real_data/labels.csv (filled in by hand — see real_data/LABELLING.md)
and the raw SerpAPI results saved by collect_real_listings.py, rebuilds each
listing's features exactly as the app does (seller trust, market price from
the whole search's results, neutral preference), and scores:

  * the DQN as deployed (trust < 0.3 hard block, then the model), against
  * simple baselines: recommend everything / trust rule only / trust rule
    plus "recommend if below market".

The target for each listing is should_recommend = legit AND good_deal.
Everything runs offline from the saved files — no SerpAPI searches.

Usage:
    python -m experiments.evaluate_real
    python -m experiments.evaluate_real --show-errors 30
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from smartshop.config import MODEL_PATH, REAL_DATA_DIR
from smartshop.environment import ShoppingEnv
from smartshop.features import extract_features, features_to_obs
from smartshop.pricing import estimate_market_references
from smartshop.reference_prices import load_reference_cache

RAW_DIR    = REAL_DATA_DIR / "raw"
LABELS_CSV = REAL_DATA_DIR / "labels.csv"
SCAM_THRESHOLD = ShoppingEnv.SCAM_TRUST_THRESHOLD

_TRUE  = {"1", "1.0", "y", "yes", "true"}
_FALSE = {"0", "0.0", "n", "no", "false"}


def parse_label(value) -> float:
    """"1"/"yes" → 1.0, "0"/"no" → 0.0, anything else (blank, "?") → NaN."""
    text = str(value).strip().lower()
    if text in _TRUE:
        return 1.0
    if text in _FALSE:
        return 0.0
    return float("nan")


def load_labels(labels_csv: str = LABELS_CSV) -> tuple[pd.DataFrame, dict]:
    """
    Labelled rows with a decidable target, plus counts of what was left out.
    A row is usable when `legit` is 0, or `legit` is 1 and `good_deal` is set.
    """
    df = pd.read_csv(labels_csv, dtype=str).fillna("")
    df["legit"] = df["legit"].map(parse_label)
    df["good_deal"] = df["good_deal"].map(parse_label)

    # Google "compare prices" cards ("₹29,000+") have no seller or link —
    # there's no one to judge, so they're left out whatever their label.
    has_seller = df["seller"].str.strip() != ""
    usable = has_seller & ((df["legit"] == 0) | ((df["legit"] == 1) & df["good_deal"].notna()))
    skipped = {
        "total_rows": len(df),
        "no_seller": int((~has_seller).sum()),
        "unlabelled": int((has_seller & df["legit"].isna()).sum()),
        "legit_without_good_deal": int((has_seller & (df["legit"] == 1) & df["good_deal"].isna()).sum()),
    }
    out = df[usable].copy()
    out["is_scam"] = out["legit"] == 0
    out["should_recommend"] = (out["legit"] == 1) & (out["good_deal"] == 1)
    return out.reset_index(drop=True), skipped


def featurize(labels: pd.DataFrame, raw_dir: str = RAW_DIR, references: dict | None = None) -> pd.DataFrame:
    """
    Adds the four model features per row, computed the same way app.py does.
    `references` maps listing id → {"reference": price, …} from the cache
    built by experiments/fetch_reference_prices.py; listings without one fall back to the
    in-search estimate, exactly as the app does when a reference is missing.
    """
    references = references or {}
    feats = []
    for (query, country), group in labels.groupby(["query", "country"], sort=False):
        slug = group["id"].iloc[0].rsplit("_", 1)[0]
        with open(Path(raw_dir) / f"{slug}.json", encoding="utf-8") as f:
            results = json.load(f)["results"]
        ref_prices = [None] * len(results)
        for _, row in group.iterrows():
            ref_prices[int(row["position"])] = (references.get(row["id"]) or {}).get("reference")
        market = estimate_market_references(results, ref_prices)   # uses the whole search, as the app does
        for idx, row in group.iterrows():
            pos = int(row["position"])
            feat = extract_features(results[pos], market_avg_price=market[pos][0], user_prefs={})
            feat["market_basis"] = market[pos][1]
            feats.append((idx, feat))
    feats.sort(key=lambda t: t[0])
    out = labels.copy()
    for key in ShoppingEnv.STATE_COLS + ["market_basis"]:
        out[key] = [f[key] for _, f in feats]
    return out


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion k/n — honest error bars for small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_decisions(should_recommend, recommended, is_scam) -> dict:
    """Accuracy / precision / recall of Recommend decisions, plus scam handling."""
    y, r, s = (np.asarray(a, dtype=bool) for a in (should_recommend, recommended, is_scam))
    tp, n_rec, n_pos = int((y & r).sum()), int(r.sum()), int(y.sum())
    correct = int((y == r).sum())
    m = {
        "n": len(y),
        "accuracy": correct / max(len(y), 1),
        "accuracy_ci": wilson_interval(correct, len(y)),
        "precision": tp / n_rec if n_rec else float("nan"),
        "precision_ci": wilson_interval(tp, n_rec),
        "recall": tp / n_pos if n_pos else float("nan"),
        "recall_ci": wilson_interval(tp, n_pos),
        "n_recommended": n_rec,
        "n_scams": int(s.sum()),
        "scams_recommended": int((s & r).sum()),
    }
    p, rc = m["precision"], m["recall"]
    m["f1"] = 2 * p * rc / (p + rc) if p + rc > 0 else float("nan")
    return m


def policies(model=None) -> dict:
    """name → function(obs) -> recommend? The DQN entry mirrors app.run_agent_inference."""
    blocked = lambda o: o[2] < SCAM_THRESHOLD
    out = {
        "Recommend everything":          lambda o: True,
        "Trust rule only":               lambda o: not blocked(o),
        "Trust rule + below market":     lambda o: not blocked(o) and ShoppingEnv.deal_value(o[0], o[3]) > 0,
    }
    if model is not None:
        out["DQN (as deployed)"] = lambda o: not blocked(o) and int(model.predict(o, deterministic=True)[0]) == 1
    return out


def _pct(x: float) -> str:
    return "  n/a" if x != x else f"{x * 100:5.1f}%"


def _ci(ci: tuple[float, float]) -> str:
    return "" if ci[0] != ci[0] else f"[{ci[0] * 100:.0f}–{ci[1] * 100:.0f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on real, hand-labelled listings.")
    parser.add_argument("--labels", default=LABELS_CSV)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--show-errors", type=int, default=15,
                        help="How many of the DQN's wrong decisions to list")
    parser.add_argument("--no-references", action="store_true",
                        help="Ignore real_data/reference_prices.json (in-search estimates only)")
    args = parser.parse_args()

    labels, skipped = load_labels(args.labels)
    if labels.empty:
        sys.exit(f"No usable labels in {args.labels} yet — see real_data/LABELLING.md.")
    references = {} if args.no_references else load_reference_cache()
    data = featurize(labels, references=references)
    n_ref = int((data["market_basis"] == "other stores, same product").sum())
    obs = [features_to_obs(row) for _, row in data.iterrows()]

    from smartshop.agent import load_agent      # needs torch, so only imported when used
    model = load_agent(args.model)

    print("\n" + "=" * 96)
    print(f"  REAL-DATA EVALUATION — {len(data)} labelled listings from "
          f"{data['query'].nunique()} searches ({int(data['is_scam'].sum())} scams, "
          f"{int(data['should_recommend'].sum())} worth recommending)")
    print(f"  Market price: reference price (other stores) for {n_ref} listings, "
          f"in-search estimate for {len(data) - n_ref}")
    print(f"  Left out: {skipped['no_seller']} Google compare-price cards (no seller), "
          f"{skipped['unlabelled']} unlabelled, "
          f"{skipped['legit_without_good_deal']} legit without a good_deal label")
    print("=" * 96)
    print(f"  {'Policy':<28}{'Accuracy [95% CI]':>22}{'Precision':>18}{'Recall':>18}"
          f"{'Scams recommended':>20}")

    results = {}
    for name, policy in policies(model).items():
        rec = np.array([policy(o) for o in obs])
        m = score_decisions(data["should_recommend"], rec, data["is_scam"])
        results[name] = (m, rec)
        print(f"  {name:<28}{_pct(m['accuracy']):>8} {_ci(m['accuracy_ci']):>12}"
              f"{_pct(m['precision']):>10} {_ci(m['precision_ci']):>8}"
              f"{_pct(m['recall']):>10} {_ci(m['recall_ci']):>8}"
              f"{m['scams_recommended']:>11} / {m['n_scams']}")
    print("=" * 96)
    print("  Precision = of the listings recommended, how many were legit good deals.")
    print("  Recall    = of the legit good deals, how many were recommended.")

    _, dqn_rec = results["DQN (as deployed)"]
    wrong = data[dqn_rec != data["should_recommend"].to_numpy()]
    if len(wrong) and args.show_errors:
        print(f"\n  DQN mistakes ({len(wrong)}), first {min(len(wrong), args.show_errors)}:")
        for i, row in wrong.head(args.show_errors).iterrows():
            label = "scam" if row["is_scam"] else ("good deal" if row["should_recommend"] else "legit, not a deal")
            action = "recommended" if dqn_rec[i] else "skipped"
            print(f"    {action:<12} {label:<18} trust={row['site_trust_score']:.2f} "
                  f"price={row['normalized_price']:.2f}× ({row['market_basis'][:22]:<22}) "
                  f"{row['seller'][:16]:<16} {row['title'][:36]}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
