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

    # reward_reason prefix (see ShoppingEnv._compute_reward) → metric
    outcomes = {
        "SCAM recommended":            "scam_hits",      # wrong
        "Scam skipped":                "scams_avoided",  # right
        "Good recommendation":         "good_recs",      # right
        "Poor-value recommendation":   "poor_recs",      # wrong
        "Missed good deal":            "deals_missed",   # wrong
        "Poor deal skipped":           "poor_skipped",   # right
    }
    metrics = {key: 0 for key in outcomes.values()}
    metrics["episode_returns"] = []
    oracle_returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_return = oracle_return = 0.0
        done = False

        while not done:
            # What a policy that knows the label would have earned here
            is_scam = env._get_info()["is_scam"]
            oracle_return += max(env._compute_reward(a, is_scam)[0] for a in (0, 1))

            action = int(predict_fn(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated

            reason = info.get("reward_reason", "")
            for prefix, key in outcomes.items():
                if reason.startswith(prefix):
                    metrics[key] += 1
                    break

        metrics["episode_returns"].append(ep_return)
        oracle_returns.append(oracle_return)

    env.close()

    total = sum(metrics[key] for key in outcomes.values())
    returns = metrics["episode_returns"]
    rate = lambda key: metrics[key] / max(total, 1)

    metrics["mean_return"]      = float(np.mean(returns))
    metrics["std_return"]       = float(np.std(returns))
    metrics["min_return"]       = float(np.min(returns))
    metrics["max_return"]       = float(np.max(returns))
    metrics["pct_of_oracle"]    = float(np.sum(returns) / max(np.sum(oracle_returns), 1e-9))
    metrics["accuracy"]         = (metrics["good_recs"] + metrics["scams_avoided"] + metrics["poor_skipped"]) / max(total, 1)
    metrics["good_rec_rate"]    = rate("good_recs")
    metrics["scam_avoid_rate"]  = rate("scams_avoided")
    metrics["scam_slip_rate"]   = rate("scam_hits")
    metrics["deal_miss_rate"]   = rate("deals_missed")
    metrics["poor_rec_rate"]    = rate("poor_recs")
    metrics["poor_skip_rate"]   = rate("poor_skipped")
    # Of all scams, how many got recommended — the safety-critical number
    metrics["scam_recall_miss"] = metrics["scam_hits"] / max(metrics["scam_hits"] + metrics["scams_avoided"], 1)

    if verbose:
        print("\n" + "=" * 65)
        print(f"  EVALUATION RESULTS — {label}")
        print("=" * 65)
        print(f"  Episodes evaluated      : {n_episodes}")
        print(f"  Mean episode return     : {metrics['mean_return']:>8.1f}")
        print(f"  Std  episode return     : {metrics['std_return']:>8.1f}")
        print(f"  Min / Max episode return: {metrics['min_return']:>8.1f} / {metrics['max_return']:>8.1f}")
        print(f"  Return vs. oracle       : {metrics['pct_of_oracle']*100:>7.1f}%")
        print(f"  Decision accuracy       : {metrics['accuracy']*100:>7.1f}%")
        print(f"  ---")
        print(f"  Good recommendations    : {metrics['good_recs']:>6,}  ({metrics['good_rec_rate']*100:5.1f}%)")
        print(f"  Poor deals skipped      : {metrics['poor_skipped']:>6,}  ({metrics['poor_skip_rate']*100:5.1f}%)")
        print(f"  Scams correctly avoided : {metrics['scams_avoided']:>6,}  ({metrics['scam_avoid_rate']*100:5.1f}%)")
        print(f"  Scam slips (BAD!)       : {metrics['scam_hits']:>6,}  ({metrics['scam_slip_rate']*100:5.1f}%)"
              f"  — {metrics['scam_recall_miss']*100:.1f}% of all scams")
        print(f"  Poor-value recommended  : {metrics['poor_recs']:>6,}  ({metrics['poor_rec_rate']*100:5.1f}%)")
        print(f"  Good deals missed       : {metrics['deals_missed']:>6,}  ({metrics['deal_miss_rate']*100:5.1f}%)")
        print("=" * 65)

    return metrics
