"""
Checks the committed dqn_shopping_agent.zip against agent_config.py without
needing torch / stable-baselines3 (the zip's `data` member is plain JSON).

Regression test: the live demo once shipped a model trained with a different
learning rate, gamma, batch size and network size than train_agent.py and
the README documented.
"""

import json
import zipfile
from pathlib import Path

import pytest

from agent_config import DQN_HYPERPARAMS

MODEL_ZIP = Path(__file__).resolve().parent.parent / "dqn_shopping_agent.zip"

CHECKED_KEYS = [
    "learning_rate", "buffer_size", "learning_starts", "batch_size", "tau",
    "gamma", "gradient_steps", "target_update_interval", "exploration_fraction",
    "exploration_initial_eps", "exploration_final_eps", "seed",
]


@pytest.fixture(scope="module")
def saved_params() -> dict:
    with zipfile.ZipFile(MODEL_ZIP) as zf:
        return json.loads(zf.read("data"))


@pytest.mark.parametrize("key", CHECKED_KEYS)
def test_committed_model_matches_training_config(saved_params, key):
    assert saved_params[key] == pytest.approx(DQN_HYPERPARAMS[key]), (
        f"dqn_shopping_agent.zip was trained with {key}={saved_params[key]}, "
        f"but agent_config.py says {DQN_HYPERPARAMS[key]} — retrain with "
        "`python train_agent.py` and commit the new zip."
    )


def test_committed_model_matches_network_architecture(saved_params):
    assert saved_params["policy_kwargs"]["net_arch"] == DQN_HYPERPARAMS["policy_kwargs"]["net_arch"]
