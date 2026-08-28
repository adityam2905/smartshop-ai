import numpy as np
import pandas as pd
import pytest

from shopping_env import ShoppingEnv


def test_observation_and_action_spaces(tiny_csv):
    env = ShoppingEnv(csv_path=tiny_csv)
    assert env.action_space.n == 2
    assert env.observation_space.shape == (4,)
    assert (env.observation_space.low == np.array([0.0, 0.0, 0.0, 0.0])).all()
    assert (env.observation_space.high == np.array([2.0, 1.0, 1.0, 1.0])).all()


def test_missing_required_column_raises(tmp_path):
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"product_name": ["x"]}).to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        ShoppingEnv(csv_path=str(bad_path))


def test_reward_function_matrix(tiny_csv):
    """
    Exercises all four cells of the reward table described in the README
    and shopping_env.py's docstring: Recommend/Skip crossed with Scam/Legit.
    """
    env = ShoppingEnv(csv_path=tiny_csv)
    env.reset(seed=0)
    # Bypass the per-episode shuffle so CSV row i is at index i.
    env._order = np.arange(env.n_products)

    for idx, row in env.df.iterrows():
        env._current_index = idx
        is_scam = bool(row["is_scam"])
        discount = float(row["discount_percentage"])

        reward_recommend, reason_recommend = env._compute_reward(action=1, is_scam=is_scam)
        reward_skip, reason_skip = env._compute_reward(action=0, is_scam=is_scam)

        if is_scam:
            assert reward_recommend == pytest.approx(-100.0)
            assert "SCAM" in reason_recommend
            assert reward_skip == pytest.approx(10.0)
            assert "skipped" in reason_skip.lower()
        else:
            # user_feedback_score defaults to 0.0 for a fresh env
            assert reward_recommend == pytest.approx(discount * 20.0)
            assert reward_skip == pytest.approx(-5.0)
            assert "missed" in reason_skip.lower()


def test_user_feedback_score_bonuses_legit_recommend_reward(tiny_csv):
    env = ShoppingEnv(csv_path=tiny_csv)
    env.reset(seed=0)
    env._order = np.arange(env.n_products)
    env.set_user_feedback(20.0)

    legit_rows = env.df.index[~env.df["is_scam"]]
    assert len(legit_rows) > 0
    idx = legit_rows[0]
    env._current_index = idx
    discount = float(env.df.loc[idx, "discount_percentage"])

    reward, _ = env._compute_reward(action=1, is_scam=False)
    assert reward == pytest.approx(discount * 20.0 + 20.0)


def test_episode_terminates_after_all_products(tiny_csv):
    env = ShoppingEnv(csv_path=tiny_csv)
    obs, info = env.reset(seed=0)
    terminated = False
    steps = 0
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        assert steps <= env.n_products  # safety net against an infinite loop
    assert steps == env.n_products


def test_reset_shuffles_deterministically_per_seed(tiny_csv):
    env = ShoppingEnv(csv_path=tiny_csv)
    env.reset(seed=42)
    order_a = env._order.copy()
    env.reset(seed=42)
    order_b = env._order.copy()
    assert (order_a == order_b).all()


def test_features_to_obs_clips_out_of_range_values():
    obs = ShoppingEnv.features_to_obs({
        "normalized_price": 5.0,      # should clip to 2.0
        "discount_percentage": -1.0,  # should clip to 0.0
        "site_trust_score": 2.0,      # should clip to 1.0
        "user_preference_score": 0.5,
    })
    assert obs == pytest.approx(np.array([2.0, 0.0, 1.0, 0.5]))


def test_features_to_obs_defaults_missing_keys():
    obs = ShoppingEnv.features_to_obs({})
    assert obs == pytest.approx(np.array([1.0, 0.0, 0.5, 0.5]))
