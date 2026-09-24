"""
The training environment: the model sees one listing at a time and decides
whether to recommend it.

Observation — 4 numbers:
    [normalized_price, discount_percentage, site_trust_score, user_preference_score]
Actions:
    0 = Skip, 1 = Recommend
Reward:
    Recommend a scam                    −100
    Skip a scam                          +10
    Recommend a legit listing            deal_value(): positive only at ≥ 10% off
    Skip a legit listing                   0

The value of a deal comes from the *real* price vs the market, not the
seller's claimed discount (which can be inflated). An earlier reward paid for
recommending any legit listing, so the model learned to recommend everything
that wasn't a scam and never had to judge whether a deal was any good.
"""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .config import DATA_CSV, DEAL_THRESHOLD, SCAM_TRUST_THRESHOLD


class ShoppingEnv(gym.Env):
    STATE_COLS = ["normalized_price", "discount_percentage", "site_trust_score", "user_preference_score"]
    SCAM_TRUST_THRESHOLD = SCAM_TRUST_THRESHOLD
    DEAL_THRESHOLD = DEAL_THRESHOLD
    PRICE_WEIGHT = 20.0
    PREFERENCE_WEIGHT = 10.0

    @classmethod
    def deal_value(cls, normalized_price: float, user_preference: float) -> float:
        """
        Reward for recommending a legit listing (before live 👍/👎 feedback).
        Negative above DEAL_THRESHOLD whatever the preference: taste can make
        the model pickier about a deal, but never lowers the bar.
        """
        price_value = cls.PRICE_WEIGHT * (cls.DEAL_THRESHOLD - normalized_price)
        if normalized_price > cls.DEAL_THRESHOLD:
            return min(price_value, -1.0)
        return price_value + cls.PREFERENCE_WEIGHT * (user_preference - 0.5)

    def __init__(self, csv_path=DATA_CSV, render_mode: Optional[str] = None, user_feedback_score: float = 0.0):
        super().__init__()
        self.render_mode = render_mode              # accepted for the Gym API; nothing is rendered
        self.user_feedback_score = user_feedback_score

        self.df = pd.read_csv(csv_path)
        missing = [c for c in self.STATE_COLS + ["is_scam"] if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        self.n_products = len(self.df)
        self._features = self.df[self.STATE_COLS].to_numpy(dtype=np.float32)
        self._records = self.df.to_dict("records")
        self._order = np.arange(self.n_products)
        self._current_index = 0

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32),   # price ratio can reach 2×
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

    def set_user_feedback(self, score: float) -> None:
        """A bonus on the Recommend reward (+20 Like / −20 Dislike)."""
        self.user_feedback_score = float(score)

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # A new random order each episode, so the model can't memorise it
        self._order = np.random.default_rng(seed).permutation(self.n_products)
        self._current_index = 0
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        info = self._get_info()
        reward, reason = self._compute_reward(action, info["is_scam"])

        self._current_index += 1
        terminated = self._current_index >= self.n_products
        obs = np.zeros(4, dtype=np.float32) if terminated else self._get_obs()

        info.update(action=action, reward=reward, reward_reason=reason)
        return obs, reward, terminated, False, info

    # ── Internals ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return self._features[self._order[self._current_index]]

    def _get_info(self) -> dict[str, Any]:
        row = self._records[self._order[self._current_index]]
        return {
            "product_name":        row.get("product_name", "Unknown"),
            "discount_percentage": float(row["discount_percentage"]),
            "site_trust_score":    float(row["site_trust_score"]),
            "is_scam":             bool(row["is_scam"]),
            "index":               self._current_index,
        }

    def _compute_reward(self, action: int, is_scam: bool) -> tuple[float, str]:
        """(reward, reason). evaluation.py and train.py count outcomes by the reason's prefix."""
        row = self._records[self._order[self._current_index]]
        price_ratio = float(row["normalized_price"])
        value = self.deal_value(price_ratio, float(row["user_preference_score"])) + self.user_feedback_score

        if action == 1:
            if is_scam:
                return -100.0, "SCAM recommended! penalty -100"
            if value > 0:
                return float(value), f"Good recommendation: price={price_ratio:.2f}× market → reward {value:.2f}"
            return float(value), f"Poor-value recommendation: price={price_ratio:.2f}× market → reward {value:.2f}"
        if is_scam:
            return 10.0, "Scam skipped correctly! → reward +10"
        if value > 0:
            return 0.0, f"Missed good deal: price={price_ratio:.2f}× market (was worth {value:.2f})"
        return 0.0, f"Poor deal skipped correctly: price={price_ratio:.2f}× market"
