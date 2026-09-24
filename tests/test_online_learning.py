"""
Behavioural tests for the committed DQN and the app's online-learning loop.

Needs torch + stable-baselines3: skipped in CI's lean job, run in its test-full job.
"""

import copy

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from smartshop.agent import fine_tune_on_feedback, load_agent  # noqa: E402

RECOMMEND, SKIP = 1, 0

# [normalized_price, discount, trust, user_pref]
TRUSTED_BIG_DISCOUNT   = np.array([0.30, 0.70, 1.00, 0.5], dtype=np.float32)   # real Amazon, 70% off
TRUSTED_SMALL_DISCOUNT = np.array([0.75, 0.25, 0.95, 0.5], dtype=np.float32)   # 25% off: clears the 0.90× bar
SMALL_SHOP_DISCOUNT    = np.array([0.72, 0.28, 0.60, 0.5], dtype=np.float32)   # unknown-but-clean shop
TRUSTED_NOT_ENOUGH_OFF = np.array([0.95, 0.05, 0.95, 0.5], dtype=np.float32)   # 5% off: below the bar
MID_TIER_DEAL          = np.array([0.45, 0.55, 0.80, 0.5], dtype=np.float32)   # known mid-tier shop, 55% off
SCAM_BIG_DISCOUNT      = np.array([0.30, 0.70, 0.10, 0.5], dtype=np.float32)
SCAM_SMALL_DISCOUNT    = np.array([0.80, 0.20, 0.10, 0.5], dtype=np.float32)   # "believable" scam
# Middling trust (passes the trust < 0.3 hard rule) but priced at a quarter
# of market — the refurbished-iPhone-at-₹18,499 pattern seen in live results
POLISHED_SCAM          = np.array([0.25, 0.75, 0.55, 0.5], dtype=np.float32)
TRUSTED_OVERPRICED     = np.array([1.30, 0.05, 0.95, 0.5], dtype=np.float32)
MODEST_DEAL_LOVED      = np.array([0.84, 0.16, 0.95, 0.9], dtype=np.float32)
MODEST_DEAL_UNWANTED   = np.array([0.84, 0.16, 0.95, 0.1], dtype=np.float32)
AT_MARKET_LOVED        = np.array([1.00, 0.00, 0.95, 1.0], dtype=np.float32)


@pytest.fixture(scope="module")
def pretrained():
    return load_agent()


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


def test_pretrained_model_skips_too_good_to_be_true_prices_from_mid_trust_sellers(pretrained):
    """The case the trust < 0.3 hard rule can't catch."""
    assert act(pretrained, POLISHED_SCAM) == SKIP


def test_pretrained_model_skips_overpriced_listings_even_from_trusted_sellers(pretrained):
    assert act(pretrained, TRUSTED_OVERPRICED) == SKIP


def test_pretrained_model_uses_the_preference_score(pretrained):
    """Same modest deal: recommended in a loved category, not in an unwanted one."""
    assert act(pretrained, MODEST_DEAL_LOVED) == RECOMMEND
    assert act(pretrained, MODEST_DEAL_UNWANTED) == SKIP


@pytest.mark.parametrize("obs", [TRUSTED_NOT_ENOUGH_OFF, AT_MARKET_LOVED])
def test_pretrained_model_needs_at_least_10_percent_off(pretrained, obs):
    """Trusted seller, but not ≤ 0.90× market → skip, even in a loved category."""
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
    # Unrelated listings keep their decisions. (A near-identical listing —
    # e.g. a small shop at 0.72× vs the disliked 0.75× — may flip too; that's
    # the dislike generalising, not the policy collapsing.)
    assert act(model, TRUSTED_BIG_DISCOUNT) == RECOMMEND
    assert act(model, MID_TIER_DEAL) == RECOMMEND
    assert act(model, SCAM_BIG_DISCOUNT) == SKIP
    assert act(model, POLISHED_SCAM) == SKIP


def test_fine_tuning_never_mutates_the_teacher(model, pretrained):
    """app.py passes the process-wide cached model as the teacher."""
    before = {k: v.clone() for k, v in pretrained.q_net.state_dict().items()}
    fine_tune_on_feedback(model, pretrained, [dislike(TRUSTED_SMALL_DISCOUNT)] * 3, gradient_steps=20)
    after = pretrained.q_net.state_dict()
    assert all((before[k] == after[k]).all() for k in before)


def test_no_feedback_means_no_training(model, pretrained):
    assert fine_tune_on_feedback(model, pretrained, [], gradient_steps=50) == 0
