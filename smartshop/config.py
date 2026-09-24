"""
Project-wide settings: file paths, decision thresholds, and the DQN's
training hyperparameters.

Deliberately free of torch / stable-baselines3 imports, so the lean CI job
can check the committed model against these settings without installing them.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
MODEL_PATH     = ROOT / "models" / "dqn_shopping_agent.zip"
DATA_CSV       = ROOT / "data" / "product_listings.csv"      # generated, not committed
REAL_DATA_DIR  = ROOT / "real_data"
CHECKPOINT_DIR = ROOT / "checkpoints"
BEST_MODEL_DIR = ROOT / "best_model"
LOG_DIR        = "logs/"      # relative on purpose: SB3 stores it inside the saved model

# ── Decision thresholds ──────────────────────────────────────────────────────
# Sellers below this trust are blocked outright, before the model is asked.
SCAM_TRUST_THRESHOLD = 0.3
# Recommend only at ≤ 90% of the usual price (≥ 10% off). Chosen from a
# 5/10/15/20% sweep on the labelled real listings (see README).
DEAL_THRESHOLD = 0.90

# ── DQN hyperparameters ──────────────────────────────────────────────────────
# tests/test_model_artifact.py fails if the committed model wasn't trained with
# these — the live demo once shipped a model trained with different settings.
DQN_HYPERPARAMS = dict(
    policy                 = "MlpPolicy",
    learning_rate          = 5e-4,
    buffer_size            = 100_000,
    learning_starts        = 2_000,     # random steps before learning starts
    batch_size             = 64,
    tau                    = 1.0,       # hard target-network update (classic DQN)
    gamma                  = 0.97,
    train_freq             = 1,
    gradient_steps         = 1,
    target_update_interval = 1_000,
    exploration_fraction   = 0.2,       # share of training spent reducing random actions
    exploration_initial_eps= 1.0,
    exploration_final_eps  = 0.02,
    policy_kwargs          = dict(net_arch=[128, 128]),
    verbose                = 0,
    tensorboard_log        = LOG_DIR,
    device                 = "auto",
    seed                   = 0,         # fixes the random start; see train.py on exact reproducibility
)
