"""
A simpler learning model to compare the DQN against: a linear contextual bandit.

Each listing's reward depends only on that listing and the action, and
skipping one doesn't change the next — so this is a contextual bandit, not a
true sequential problem. If a one-layer bandit (no replay buffer, target
network or discounting) did as well as the DQN, the extra machinery wouldn't
be earning its keep. Both are scored with the same smartshop.evaluation.

    python -m experiments.bandit_baseline
    python -m experiments.bandit_baseline --compare-dqn
"""

import argparse
from typing import Optional

import numpy as np

from smartshop.config import DATA_CSV, MODEL_PATH
from smartshop.data_generator import ensure_dataset
from smartshop.environment import ShoppingEnv
from smartshop.evaluation import evaluate_policy


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
    csv_path=DATA_CSV,
    n_episodes: int = 60,
    lr: float = 0.05,
    seed: int = 0,
) -> LinearEpsilonGreedyBandit:
    """Trains a fresh bandit by walking the dataset `n_episodes` times."""
    env = ShoppingEnv(csv_path=csv_path)
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
        ("pct_of_oracle",    "Return vs. oracle",              "{:>15.1%}"),
        ("accuracy",         "Decision accuracy",              "{:>15.1%}"),
        ("scam_recall_miss", "Scams recommended (of scams)",   "{:>15.1%}"),
        ("poor_rec_rate",    "Poor-value recommended",         "{:>15.1%}"),
        ("deal_miss_rate",   "Good deals missed",              "{:>15.1%}"),
    ]
    for key, name, fmt in rows:
        print(f"  {name:<30}{fmt.format(bandit_metrics[key])}{fmt.format(dqn_metrics[key])}")
    print("=" * 65)
    print(
        "\nThe task is a contextual bandit (no action affects the next\n"
        "listing), so any gap here is about model capacity — a linear score\n"
        "vs. an MLP — not about the DQN's temporal-credit machinery."
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
    parser.add_argument("--csv", default=DATA_CSV)
    parser.add_argument("--compare-dqn", action="store_true",
                        help="Also score the trained DQN and print a side-by-side comparison")
    args = parser.parse_args()
    ensure_dataset(args.csv)

    print(f"Training linear epsilon-greedy bandit for {args.train_episodes} episodes…")
    bandit = train_bandit(csv_path=args.csv, n_episodes=args.train_episodes, lr=args.lr)

    bandit_metrics = evaluate_policy(
        lambda obs: bandit.act(obs, deterministic=True),
        csv_path=args.csv,
        n_episodes=args.episodes,
        label="Linear Bandit",
    )

    if args.compare_dqn:
        if not MODEL_PATH.exists():
            print(f"\n(no model at {MODEL_PATH} — run `python -m smartshop.train` first)")
            return
        from smartshop.agent import load_agent      # needs torch, so only imported when used
        model = load_agent()
        dqn_metrics = evaluate_policy(
            lambda obs: int(model.predict(obs, deterministic=True)[0]),
            csv_path=args.csv,
            n_episodes=args.episodes,
            label="DQN",
        )
        _print_comparison(bandit_metrics, dqn_metrics)


if __name__ == "__main__":
    main()
