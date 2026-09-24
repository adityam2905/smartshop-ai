"""
Behavioural tests for the committed DQN and the app's online-learning loop.

Needs the full requirements.txt (torch + stable-baselines3); skipped under
the lean requirements-test.txt used in CI.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from train_agent import fine_tune_on_feedback, load_agent  # noqa: E402

MODEL_ZIP = Path(__file__).resolve().parent.parent / "dqn_shopping_agent.zip"

RECOMMEND, SKIP = 1, 0

# [normalized_price, discount, trust, user_pref]
TRUSTED_BIG_DISCOUNT   = np.array([0.30, 0.70, 1.00, 0.5], dtype=np.float32)   # real Amazon, 70% off
TRUSTED_SMALL_DISCOUNT = np.array([0.80, 0.20, 0.95, 0.5], dtype=np.float32)
SMALL_SHOP_DISCOUNT    = np.array([0.75, 0.25, 0.55, 0.5], dtype=np.float32)   # unknown-but-clean shop
SCAM_BIG_DISCOUNT      = np.array([0.30, 0.70, 0.10, 0.5], dtype=np.float32)
SCAM_SMALL_DISCOUNT    = np.array([0.80, 0.20, 0.10, 0.5], dtype=np.float32)   # "believable" scam


@pytest.fixture(scope="module")
def pretrained():
    return load_agent(str(MODEL_ZIP))


@pytest.fixture
def model(pretrained):
    return copy.deepcopy(pretrained)


def act(model, obs) -> int:
    return int(model.predict(obs, deterministic=True)[0])


def q(model, obs) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model.q_net(torch.as_tensor(obs[None], device=model.device)).cpu().numpy()[0]


def dislike(obs):
    return {"obs": obs, "action": RECOMMEND, "reward": -20.0}


# ── Pretrained policy: keyed on trust, not on discount size ─────────────────

@pytest.mark.parametrize("obs", [TRUSTED_BIG_DISCOUNT, TRUSTED_SMALL_DISCOUNT, SMALL_SHOP_DISCOUNT])
def test_pretrained_model_recommends_legit_listings_at_any_discount(pretrained, obs):
    assert act(pretrained, obs) == RECOMMEND


@pytest.mark.parametrize("obs", [SCAM_BIG_DISCOUNT, SCAM_SMALL_DISCOUNT])
def test_pretrained_model_skips_low_trust_listings_at_any_discount(pretrained, obs):
    assert act(pretrained, obs) == SKIP


# ── Online fine-tuning ──────────────────────────────────────────────────────

def test_first_fine_tune_trains_immediately_without_crashing(model, pretrained):
    """Used to train nothing for 126 clicks, then crash at click 129."""
    before = q(model, TRUSTED_SMALL_DISCOUNT)[RECOMMEND]
    steps = fine_tune_on_feedback(model, pretrained, [dislike(TRUSTED_SMALL_DISCOUNT)] * 3, gradient_steps=50)
    assert steps == 50
    assert q(model, TRUSTED_SMALL_DISCOUNT)[RECOMMEND] < before


def test_repeated_dislikes_flip_that_item_but_do_not_collapse_the_policy(model, pretrained):
    """
    Used to: 128 Dislikes made the model skip every listing. Now the
    disliked listing should be skipped while unrelated legit deals are
    still recommended and scams are still skipped.
    """
    feedback = []
    for _ in range(128 // 3):
        feedback += [dislike(TRUSTED_SMALL_DISCOUNT)] * 3
        fine_tune_on_feedback(model, pretrained, feedback, gradient_steps=50)

    assert act(model, TRUSTED_SMALL_DISCOUNT) == SKIP
    assert act(model, TRUSTED_BIG_DISCOUNT) == RECOMMEND
    assert act(model, SMALL_SHOP_DISCOUNT) == RECOMMEND
    assert act(model, SCAM_BIG_DISCOUNT) == SKIP


def test_fine_tuning_never_mutates_the_teacher(model, pretrained):
    """app.py passes the process-wide cached model as the teacher."""
    before = {k: v.clone() for k, v in pretrained.q_net.state_dict().items()}
    fine_tune_on_feedback(model, pretrained, [dislike(TRUSTED_SMALL_DISCOUNT)] * 3, gradient_steps=20)
    after = pretrained.q_net.state_dict()
    assert all((before[k] == after[k]).all() for k in before)


def test_no_feedback_means_no_training(model, pretrained):
    assert fine_tune_on_feedback(model, pretrained, [], gradient_steps=50) == 0
