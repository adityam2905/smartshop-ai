"""
Phase 2: Custom Gymnasium Environment
ShoppingEnv — wraps the synthetic product dataset as an RL environment.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class ShoppingEnv(gym.Env):
    """
    A custom Gymnasium environment for the E-Commerce Deal Hunter agent.

    Observation (state):  4 continuous values —
        [normalized_price, discount_percentage, site_trust_score, user_preference_score]

    Actions:
        0 → Skip    (do not show this product)
        1 → Recommend (show this product to the user)

    Reward function:
        Recommend + Scam  (trust < 0.3) : -100  catastrophic penalty
        Recommend + Legit              : (discount_pct * 20) + user_feedback_score
        Skip      + Scam               :  +10   correctly avoided a trap
        Skip      + Legit              :   -5   missed a good deal
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Column names in the CSV that map to the 4 state features
    STATE_COLS = [
        "normalized_price",
        "discount_percentage",
        "site_trust_score",
        "user_preference_score",
    ]
    SCAM_TRUST_THRESHOLD = 0.3

    def __init__(
        self,
        csv_path: str = "product_listings.csv",
        render_mode: Optional[str] = None,
        user_feedback_score: float = 0.0,   # injected externally for online learning
    ):
        super().__init__()

        self.render_mode = render_mode
        self.user_feedback_score = user_feedback_score   # updated by app.py

        # ── Load dataset ──────────────────────────────────────────────────────
        self.df = pd.read_csv(csv_path)
        self._validate_columns()
        self.n_products = len(self.df)
        self._feature_matrix = self.df[self.STATE_COLS].to_numpy(dtype=np.float32)
        self._records = self.df.to_dict("records")
        self._order = np.arange(self.n_products)

        # ── Spaces ────────────────────────────────────────────────────────────
        # Observation: 4 floats, each in [0, 1] except normalized_price → [0, 2]
        low  = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action: 0 = Skip, 1 = Recommend
        self.action_space = spaces.Discrete(2)

        # ── Internal state ────────────────────────────────────────────────────
        self._current_index: int = 0
        self._current_obs: Optional[np.ndarray] = None
        self._episode_rewards: list = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate_columns(self) -> None:
        missing = [c for c in self.STATE_COLS + ["is_scam"] if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

    def _get_obs(self) -> np.ndarray:
        row_index = self._order[self._current_index]
        return self._feature_matrix[row_index]

    def _get_info(self) -> Dict[str, Any]:
        row = self._records[self._order[self._current_index]]
        return {
            "product_name":        row.get("product_name", "Unknown"),
            "category":            row.get("category", "Unknown"),
            "price":               float(row.get("price", 0.0)),
            "market_avg_price":    float(row.get("market_avg_price", 0.0)),
            "discount_percentage": float(row["discount_percentage"]),
            "site_trust_score":    float(row["site_trust_score"]),
            "is_scam":             bool(row["is_scam"]),
            "site_url":            row.get("site_url", ""),
            "index":               self._current_index,
        }

    def _compute_reward(self, action: int, is_scam: bool) -> Tuple[float, str]:
        """
        Returns (reward, reason_string).
        """
        row = self._records[self._order[self._current_index]]
        discount = float(row["discount_percentage"])
        trust = float(row["site_trust_score"])

        if action == 1:   # Recommend
            if is_scam:
                reward = -100.0
                reason = "SCAM recommended! penalty -100"
            else:                                          # legit deal
                reward = (discount * 20.0) + self.user_feedback_score
                reason = (
                    f"Good recommendation: discount={discount:.2f}, "
                    f"feedback={self.user_feedback_score:.1f} → reward={reward:.2f}"
                )
        else:              # Skip
            if is_scam:          # correctly avoided scam
                reward = +10.0
                reason = "Scam skipped correctly! → reward +10"
            else:                                          # missed a legit deal
                reward = -5.0
                reason = (
                    f"Missed legit deal: discount={discount:.2f}, "
                    f"trust={trust:.2f} → penalty -5"
                )

        return float(reward), reason

    # ── Core Gym API ──────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Shuffle dataset at the start of each episode so the agent sees
        # products in a different order → better generalisation
        rng = np.random.default_rng(seed)
        self._order = rng.permutation(self.n_products)
        self._current_index = 0
        self._episode_rewards = []

        obs = self._get_obs()
        self._current_obs = obs
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human(obs, info)

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        info = self._get_info()
        is_scam = info["is_scam"]

        reward, reason = self._compute_reward(action, is_scam)
        self._episode_rewards.append(reward)

        # Advance to next product
        self._current_index += 1
        terminated = self._current_index >= self.n_products
        truncated  = False

        if not terminated:
            obs = self._get_obs()
        else:
            # Return a zero observation on termination (Gym convention)
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        self._current_obs = obs
        info["action"]        = action
        info["reward"]        = reward
        info["reward_reason"] = reason
        info["episode_return"] = sum(self._episode_rewards)

        if self.render_mode == "human":
            self._render_human(obs, info, action=action, reward=reward, reason=reason)

        return obs, reward, terminated, truncated, info

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_human(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
        action: Optional[int] = None,
        reward: Optional[float] = None,
        reason: str = "",
    ) -> None:
        action_str = {0: "SKIP", 1: "RECOMMEND"}.get(action, "—")
        scam_tag   = "🚨 SCAM" if info["is_scam"] else "✅ LEGIT"
        reward_str = f"{reward:.2f}" if reward is not None else "—"
        print(
            f"[{info['index']:>4}] {info['product_name'][:40]:<40} "
            f"| {scam_tag} "
            f"| trust={info['site_trust_score']:.2f} "
            f"| disc={info['discount_percentage']:.2f} "
            f"| action={action_str} "
            f"| reward={reward_str:>7} "
            f"| {reason}"
        )

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            info = self._get_info()
            return (
                f"Index={info['index']} | Product={info['product_name']} | "
                f"Trust={info['site_trust_score']:.2f} | "
                f"Discount={info['discount_percentage']:.2f} | "
                f"Scam={info['is_scam']}"
            )
        return None

    def close(self) -> None:
        pass

    # ── Extra: inject live user feedback (used in app.py online loop) ─────────

    def set_user_feedback(self, score: float) -> None:
        """
        Called by app.py to inject real-time user feedback (+20 Like / -20 Dislike).
        This updates the bonus component of the Recommend-Legit reward.
        """
        self.user_feedback_score = float(score)

    # ── Extra: build a state array from a raw dict (used in scraper.py) ───────

    @staticmethod
    def features_to_obs(features: Dict[str, float]) -> np.ndarray:
        """
        Convert a feature dict produced by scraper.extract_features() into
        the 4-float observation array expected by the DQN model.
        """
        obs = np.array(
            [
                np.clip(features.get("normalized_price", 1.0),        0.0, 2.0),
                np.clip(features.get("discount_percentage", 0.0),     0.0, 1.0),
                np.clip(features.get("site_trust_score", 0.5),        0.0, 1.0),
                np.clip(features.get("user_preference_score", 0.5),   0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return obs


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        # _render_human() prints emoji (✅/🚨); on Windows the default
        # console codepage (cp1252) can't encode them and this script would
        # crash with a UnicodeEncodeError otherwise.
        sys.stdout.reconfigure(encoding="utf-8")

    # Generate a tiny dataset on-the-fly if the CSV doesn't exist yet
    if not os.path.exists("product_listings.csv"):
        print("product_listings.csv not found — generating synthetic data first...")
        from data_generator import generate_dataset
        generate_dataset(n=5000).to_csv("product_listings.csv", index=False)

    env = ShoppingEnv(csv_path="product_listings.csv", render_mode="human")

    print("=" * 100)
    print("Smoke test: 10 random steps")
    print("=" * 100)

    obs, info = env.reset(seed=0)

    total_reward = 0.0
    for step_i in range(10):
        action = env.action_space.sample()   # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break

    print("=" * 100)
    print(f"Total reward over {step_i + 1} steps: {total_reward:.1f}")

    # Verify spaces
    print("\nObservation space:", env.observation_space)
    print("Action space:     ", env.action_space)
    print("Sample obs:       ", obs)

    env.close()
    print("\n✅ ShoppingEnv passed smoke test.")
