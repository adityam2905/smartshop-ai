import numpy as np
import pytest

from bandit_baseline import LinearEpsilonGreedyBandit, train_bandit
from evaluation import evaluate_policy


def test_epsilon_decays_from_start_to_floor():
    bandit = LinearEpsilonGreedyBandit(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=10)
    assert bandit._epsilon() == pytest.approx(1.0)
    bandit.t = 10
    assert bandit._epsilon() == pytest.approx(0.1)
    bandit.t = 1000  # past the decay window — should clamp, not overshoot
    assert bandit._epsilon() == pytest.approx(0.1)


def test_update_moves_the_chosen_arm_toward_the_observed_reward():
    bandit = LinearEpsilonGreedyBandit(n_actions=2, n_features=4, lr=0.5)
    obs = np.array([1.0, 0.5, 0.9, 0.5], dtype=np.float32)

    before = bandit.weights[1] @ bandit._featurize(obs)
    bandit.update(obs, action=1, reward=10.0)
    after = bandit.weights[1] @ bandit._featurize(obs)

    assert after > before  # moved toward the positive reward
    # The untouched arm's weights must be unaffected by an update to arm 1.
    assert (bandit.weights[0] == 0).all()


def test_deterministic_act_ignores_epsilon():
    bandit = LinearEpsilonGreedyBandit(epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=1)
    bandit.weights[1] = np.array([0.0, 0.0, 0.0, 0.0, 5.0])  # arm 1 always wins via the bias term
    obs = np.zeros(4, dtype=np.float32)
    for _ in range(20):
        assert bandit.act(obs, deterministic=True) == 1


def test_bandit_learns_to_avoid_the_catastrophic_scam_penalty(tiny_csv):
    """
    On the fixture's 2-row dataset, recommending the scam item costs -100
    while skipping it earns +10 — a large enough gap that a bandit trained
    for a modest number of episodes should reliably learn to skip it.
    """
    bandit = train_bandit(csv_path=tiny_csv, n_episodes=200, lr=0.1, seed=0)
    metrics = evaluate_policy(
        lambda obs: bandit.act(obs, deterministic=True),
        csv_path=tiny_csv,
        n_episodes=10,
        verbose=False,
    )
    assert metrics["scam_slip_rate"] < 0.5
