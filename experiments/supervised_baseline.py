"""
Scam detection as a plain classification problem: how do Logistic Regression
and a Random Forest, trained on the same 4 features the DQN sees, compare with
the app's hard rule (block sellers with trust < 0.3)?

    python -m experiments.supervised_baseline
"""

import argparse
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

from smartshop.config import DATA_CSV
from smartshop.data_generator import ensure_dataset
from smartshop.environment import ShoppingEnv

# Reuse the exact same 4 columns the RL agent observes, so this is a fair
# apples-to-apples comparison rather than a classifier given extra info.
FEATURES = ShoppingEnv.STATE_COLS
TRUST_RULE_THRESHOLD = ShoppingEnv.SCAM_TRUST_THRESHOLD   # the app's hard block


def load_data(csv_path=DATA_CSV) -> Tuple[pd.DataFrame, pd.Series]:
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
    parser.add_argument("--csv", default=DATA_CSV)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dataset(args.csv)

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
        "\nNote on interpreting these numbers: the hard rule never flags a\n"
        "legit seller (precision 1.0) but misses the polished scam shops,\n"
        "whose trust (0.30-0.60) overlaps small legit shops -- they're only\n"
        "given away by a price far below market. The learned models catch\n"
        "them by combining trust with normalized_price, which is also what\n"
        "the DQN has to learn. The confusion matrix (FN = scams missed) is\n"
        "the number to watch."
    )

    print("\nRandom Forest feature importances (which of the 4 features it leaned on):")
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {feat:<24}{imp:.3f}")


if __name__ == "__main__":
    main()
