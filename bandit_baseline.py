"""
Contextual-bandit baseline for the Deal Hunter problem.

Why this file exists
---------------------
ShoppingEnv is framed as a sequential Gymnasium environment and trained
with a DQN, but look closely at shopping_env.step(): the reward for a given
product depends only on that product's own features and the action taken
on it, and the *next* observation (the next product in the shuffled order)
does not depend on the action just taken. There is no credit assignment
across time — every "episode" is really a sequence of independent one-shot
decisions with a context (the 4 product features) attached to each one.

That is a contextual bandit problem, not a true MDP. Full DQN machinery
(a replay buffer, a target network, a discount factor) isn't obviously
buying anything over a much simpler per-decision learner, so this module
implements one and benchmarks it against the trained DQN using the exact
same evaluation harness (evaluation.evaluate_policy) — see README.md's
"Bandit vs. DQN" section for how to read the results.

Usage:
    python bandit_baseline.py                       # train + evaluate the bandit alone
    python bandit_baseline.py --compare-dqn          # also compare against dqn_shopping_agent.zip
    python bandit_baseline.py --train-episodes 100 --episodes 30 --compare-dqn
"""

import argparse
import os
from typing import Optional

import numpy as np

from shopping_env import ShoppingEnv
from evaluation import evaluate_policy

CSV_PATH = "product_listings.csv"


class LinearEpsilonGreedyBandit:
    """
    A minimal contextual bandit: one linear scoring function
    Q(s, a) = w_a . [s, 1] per action, updated online via stochastic
    gradient descent toward the *single-step* reward actually observed.

    No bootstrapping, no discounting, no replay buffer, no target network —
    that's the point. If the problem really is a bandit (see module
    docstring), this is the "right-sized" tool, and how close it gets to
    the DQN's performance is itself informative.
    """

    def __init__(
        self,
        n_actions: int = 2,
        n_features: int = 4,
        lr: float = 0.05,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20_000,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(epsilon_decay_steps, 1)
        self.rng = np.random.default_rng(seed)
        # +1 feature slot for a bias term appended to every context vector.
        self.weights = np.zeros((n_actions, n_features + 1), dtype=np.float64)
        self.t = 0

    @staticmethod
    def _featurize(obs: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float64), [1.0]])

    def _epsilon(self) -> float:
        frac = min(self.t / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        x = self._featurize(obs)
        q_values = self.weights @ x
        if not deterministic and self.rng.random() < self._epsilon():
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(q_values))

    def update(self, obs: np.ndarray, action: int, reward: float) -> None:
        """Single SGD step toward the observed reward for the chosen arm."""
        x = self._featurize(obs)
        pred = self.weights[action] @ x
        error = reward - pred
        self.weights[action] += self.lr * error * x
        self.t += 1


def train_bandit(
    csv_path: str = CSV_PATH,
    n_episodes: int = 60,
    lr: float = 0.05,
    seed: int = 0,
) -> LinearEpsilonGreedyBandit:
    """Trains a fresh bandit by walking the dataset `n_episodes` times."""
    env = ShoppingEnv(csv_path=csv_path, render_mode=None)
    bandit = LinearEpsilonGreedyBandit(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        lr=lr,
        epsilon_decay_steps=max(n_episodes * env.n_products, 1),
        seed=seed,
    )

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            action = bandit.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            bandit.update(obs, action, reward)
            obs = next_obs
            done = terminated or truncated

    env.close()
    return bandit


def _print_comparison(bandit_metrics: dict, dqn_metrics: dict) -> None:
    print("\n" + "=" * 65)
    print("  LINEAR BANDIT vs DQN")
    print("=" * 65)
    print(f"  {'Metric':<28}{'Bandit':>15}{'DQN':>15}")
    rows = [
        ("mean_return",     "Mean episode return",            "{:>15.1f}"),
        ("good_rec_rate",   "Good rec rate",                  "{:>15.1%}"),
        ("scam_avoid_rate", "Scam avoid rate",                "{:>15.1%}"),
        ("scam_slip_rate",  "Scam slip rate (lower=better)",  "{:>15.1%}"),
        ("deal_miss_rate",  "Deal miss rate",                 "{:>15.1%}"),
    ]
    for key, name, fmt in rows:
        print(f"  {name:<28}{fmt.format(bandit_metrics[key])}{fmt.format(dqn_metrics[key])}")
    print("=" * 65)
    print(
        "\nA bandit landing close to the DQN on these numbers is evidence\n"
        "that the DQN's extra machinery isn't earning its complexity for\n"
        "this particular formulation of the problem — worth calling out\n"
        "explicitly rather than letting the DQN framing go unquestioned."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the contextual-bandit baseline, optionally against the DQN."
    )
    parser.add_argument("--train-episodes", type=int, default=60,
                        help="Bandit training episodes (default: 60)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Evaluation episodes for both policies (default: 20)")
    parser.add_argument("--lr", type=float, default=0.05, help="Bandit learning rate")
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--compare-dqn", action="store_true",
                        help="Also load dqn_shopping_agent.zip and print a side-by-side comparison")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"'{args.csv}' not found — generating synthetic data first…")
        from data_generator import generate_dataset
        generate_dataset(n=5000).to_csv(args.csv, index=False)
        print(f"Generated '{args.csv}'.\n")

    print(f"Training linear epsilon-greedy bandit for {args.train_episodes} episodes…")
    bandit = train_bandit(csv_path=args.csv, n_episodes=args.train_episodes, lr=args.lr)

    bandit_metrics = evaluate_policy(
        lambda obs: bandit.act(obs, deterministic=True),
        csv_path=args.csv,
        n_episodes=args.episodes,
        label="Linear Bandit",
    )

    if args.compare_dqn:
        model_path = "dqn_shopping_agent.zip"
        if not os.path.exists(model_path):
            print(f"\n(no {model_path} found — run train_agent.py first to enable --compare-dqn)")
            return
        from train_agent import load_agent
        model = load_agent(model_path)
        dqn_metrics = evaluate_policy(
            lambda obs: int(model.predict(obs, deterministic=True)[0]),
            csv_path=args.csv,
            n_episodes=args.episodes,
            label="DQN",
        )
        _print_comparison(bandit_metrics, dqn_metrics)


if __name__ == "__main__":
    main()
