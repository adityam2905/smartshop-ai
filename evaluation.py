"""
Shared evaluation harness.

Both the DQN (train_agent.py) and the contextual-bandit baseline
(bandit_baseline.py) get scored by rolling the same kind of policy —
something that maps an observation to an action — through ShoppingEnv and
tallying the same metrics. Previously that rollout/scoring logic was
duplicated inline in train_agent.py; pulling it out here means both policies
are held to identical, single-source-of-truth scoring, and a new policy
(a supervised classifier, a rule-based one, etc.) can be benchmarked the
same way just by handing it a `predict_fn`.
"""

from typing import Callable, Dict

import numpy as np

from shopping_env import ShoppingEnv

# obs (shape (4,)) -> action (0 = Skip, 1 = Recommend)
PredictFn = Callable[[np.ndarray], int]


def evaluate_policy(
    predict_fn: PredictFn,
    csv_path: str = "product_listings.csv",
    n_episodes: int = 20,
    verbose: bool = True,
    label: str = "Policy",
) -> Dict[str, float]:
    """
    Rolls out `predict_fn` deterministically (no exploration) for
    `n_episodes` full passes over the dataset and returns a metrics dict.
    """
    env = ShoppingEnv(csv_path=csv_path, render_mode=None)

    metrics = {
        "scam_hits": 0, "scams_avoided": 0,
        "good_recs": 0, "deals_missed": 0,
        "episode_returns": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_return = 0.0
        done = False

        while not done:
            action = int(predict_fn(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

            reason = info.get("reward_reason", "")
            if "SCAM recommended" in reason:
                metrics["scam_hits"] += 1
            elif "Scam skipped" in reason:
                metrics["scams_avoided"] += 1
            elif "Missed legit" in reason:
                metrics["deals_missed"] += 1
            elif "Good recommendation" in reason:
                metrics["good_recs"] += 1

        metrics["episode_returns"].append(ep_return)

    env.close()

    total = sum([
        metrics["scam_hits"], metrics["scams_avoided"],
        metrics["good_recs"], metrics["deals_missed"],
    ])
    returns = metrics["episode_returns"]

    metrics["mean_return"]      = float(np.mean(returns))
    metrics["std_return"]       = float(np.std(returns))
    metrics["min_return"]       = float(np.min(returns))
    metrics["max_return"]       = float(np.max(returns))
    metrics["good_rec_rate"]    = metrics["good_recs"]     / max(total, 1)
    metrics["scam_avoid_rate"]  = metrics["scams_avoided"] / max(total, 1)
    metrics["scam_slip_rate"]   = metrics["scam_hits"]     / max(total, 1)
    metrics["deal_miss_rate"]   = metrics["deals_missed"]  / max(total, 1)

    if verbose:
        print("\n" + "=" * 65)
        print(f"  EVALUATION RESULTS — {label}")
        print("=" * 65)
        print(f"  Episodes evaluated      : {n_episodes}")
        print(f"  Mean episode return     : {metrics['mean_return']:>8.1f}")
        print(f"  Std  episode return     : {metrics['std_return']:>8.1f}")
        print(f"  Min / Max episode return: {metrics['min_return']:>8.1f} / {metrics['max_return']:>8.1f}")
        print(f"  ---")
        print(f"  Good recommendations    : {metrics['good_recs']:>6,}  ({metrics['good_rec_rate']*100:5.1f}%)")
        print(f"  Scams correctly avoided : {metrics['scams_avoided']:>6,}  ({metrics['scam_avoid_rate']*100:5.1f}%)")
        print(f"  Scam slips (BAD!)       : {metrics['scam_hits']:>6,}  ({metrics['scam_slip_rate']*100:5.1f}%)")
        print(f"  Legit deals missed      : {metrics['deals_missed']:>6,}  ({metrics['deal_miss_rate']*100:5.1f}%)")
        print("=" * 65)

    return metrics
