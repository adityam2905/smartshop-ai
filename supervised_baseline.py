"""
Supervised baseline for scam detection.

bandit_baseline.py asks whether the RL framing (contextual bandit vs. DQN)
is buying anything on the full *policy* task (Recommend/Skip, which mixes
"is this a scam" with "is this deal good enough to bother showing"). This
module asks a narrower, arguably more load-bearing question: on just the
safety-critical sub-task -- "is this listing a scam?" -- how does a plain
supervised classifier, trained directly on the `is_scam` label, compare to:

  (a) the hard-coded `site_trust_score < 0.3` rule that app.py actually
      applies in production (see run_agent_inference() in app.py), and
  (b) how well the DQN/bandit end up avoiding scams as a *side effect* of
      reward maximisation (their scam_avoid_rate / scam_slip_rate from
      evaluation.py).

Usage:
    python supervised_baseline.py
    python supervised_baseline.py --test-size 0.3 --seed 0
"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from shopping_env import ShoppingEnv

CSV_PATH = "product_listings.csv"
# Reuse the exact same 4 columns the RL agent observes, so this is a fair
# apples-to-apples comparison rather than a classifier given extra info.
FEATURES = ShoppingEnv.STATE_COLS
TRUST_RULE_THRESHOLD = ShoppingEnv.SCAM_TRUST_THRESHOLD  # 0.3, matches app.py's hard filter


def load_data(csv_path: str = CSV_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    return df[FEATURES], df["is_scam"].astype(int)


def score(name: str, y_true, y_pred) -> dict:
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return metrics


def print_report(metrics: dict) -> None:
    print(
        f"  {metrics['model']:<28}"
        f"acc={metrics['accuracy']:.3f}  "
        f"precision={metrics['precision']:.3f}  "
        f"recall={metrics['recall']:.3f}  "
        f"f1={metrics['f1']:.3f}   "
        f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised scam-detection baselines.")
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"'{args.csv}' not found — generating synthetic data first…")
        from data_generator import generate_dataset
        generate_dataset(n=5000).to_csv(args.csv, index=False)
        print(f"Generated '{args.csv}'.\n")

    X, y = load_data(args.csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    print("=" * 78)
    print("  SUPERVISED SCAM-DETECTION BASELINES")
    print(f"  {len(X_train)} train / {len(X_test)} test rows"
          f" ({y.mean() * 100:.1f}% scam overall)")
    print("=" * 78)

    results = []

    # 1. The hard rule app.py already ships with in production. Zero
    #    training, just a threshold on one of the four features.
    rule_pred = (X_test["site_trust_score"] < TRUST_RULE_THRESHOLD).astype(int)
    results.append(score(f"Hard rule (trust < {TRUST_RULE_THRESHOLD})", y_test, rule_pred))

    # 2. Logistic Regression — simple, interpretable, linear baseline.
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    results.append(score("Logistic Regression", y_test, logreg.predict(X_test)))

    # 3. Random Forest — a stronger, non-linear baseline.
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=args.seed)
    rf.fit(X_train, y_train)
    results.append(score("Random Forest", y_test, rf.predict(X_test)))

    for r in results:
        print_report(r)

    print("=" * 78)
    print(
        "\nNote on interpreting these numbers: site_trust_score alone nearly\n"
        "perfectly separates scam/legit in this synthetic dataset (by\n"
        "construction, data_generator.py gives scam rows trust in [0, 0.28]\n"
        "and legit rows trust in [0.55, 1.0]) -- so ALL three approaches score\n"
        "very high here, including the zero-parameter hard rule. That is\n"
        "expected, not evidence the learned models are doing something\n"
        "clever; see README.md's Limitations section (#3) on why the dataset\n"
        "is currently too easy to be a demanding benchmark. The confusion-\n"
        "matrix breakdown (false positives/negatives) is more informative\n"
        "here than the headline scores."
    )

    print("\nRandom Forest feature importances (which of the 4 features it leaned on):")
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {feat:<24}{imp:.3f}")


if __name__ == "__main__":
    main()
