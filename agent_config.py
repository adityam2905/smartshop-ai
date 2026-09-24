"""
DQN hyperparameters — the single source of truth for train_agent.py, the
README, and tests/test_model_artifact.py.

Kept in its own module (no torch / stable-baselines3 import) so the lean CI
test suite can check that the committed dqn_shopping_agent.zip was actually
trained with these settings. The live demo previously shipped a model
trained with a different LR, gamma, batch size and network size than the
ones documented here, and nothing caught it.
"""

LOG_DIR = "logs/"

DQN_HYPERPARAMS = dict(
    policy               = "MlpPolicy",
    learning_rate        = 5e-4,        # Adam LR
    buffer_size          = 100_000,     # replay buffer capacity
    learning_starts      = 2_000,       # steps before first gradient update
    batch_size           = 64,          # mini-batch size for each update
    tau                  = 1.0,         # hard target-network update (classic DQN)
    gamma                = 0.97,        # discount factor — balanced future vs immediate
    train_freq           = 1,           # update every environment step
    gradient_steps       = 1,
    target_update_interval = 1_000,     # sync target network every 1000 steps
    exploration_fraction   = 0.2,       # fraction of training spent decaying ε
    exploration_initial_eps= 1.0,       # start fully random
    exploration_final_eps  = 0.02,      # end with 2% random actions
    policy_kwargs        = dict(
        net_arch=[128, 128],            # compact MLP for low-dimensional state
    ),
    verbose              = 0,           # suppress SB3 internal logs (we use our callback)
    tensorboard_log      = LOG_DIR,
    device               = "auto",
)
