"""
Phase 3: Pre-Training the DQN Brain
Trains a Deep Q-Network agent on the ShoppingEnv for 50,000–100,000 timesteps.
Saves the final model to dqn_shopping_agent.zip.

Usage:
    python train_agent.py                        # default 100,000 steps
    python train_agent.py --timesteps 50000      # quick run
    python train_agent.py --timesteps 100000 --eval   # train + final evaluation
"""

import argparse
import os
import time
import numpy as np
import torch

# ── Stable Baselines 3 ────────────────────────────────────────────────────────
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

# ── Local ─────────────────────────────────────────────────────────────────────
from shopping_env import ShoppingEnv
from evaluation import evaluate_policy

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH  = "dqn_shopping_agent"          # SB3 appends .zip automatically
CHECKPOINT_DIR   = "checkpoints/"
LOG_DIR          = "logs/"
CSV_PATH         = "product_listings.csv"
EVAL_EPISODES    = 20
N_TRAIN_ENVS     = 4


# ─────────────────────────────────────────────────────────────────────────────
# Custom callback: prints a live training summary every N steps
# ─────────────────────────────────────────────────────────────────────────────
class TrainingMonitorCallback(BaseCallback):
    """
    Logs reward stats and scam-hit / deal-miss rates to the console
    at regular intervals so you can watch the agent improve in real time.
    """

    def __init__(self, log_interval: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval   = log_interval
        self.episode_rewards: list[float] = []
        self.scam_hits      = 0     # times agent recommended a scam
        self.deals_missed   = 0     # times agent skipped a legit deal
        self.scams_avoided  = 0     # times agent correctly skipped a scam
        self.good_recs      = 0     # times agent recommended a legit deal
        self._last_log_step = 0

    def _on_step(self) -> bool:
        # Harvest info dicts from the vectorised environment
        for info in self.locals.get("infos", []):
            reward = info.get("reward")
            reason = info.get("reward_reason", "")
            if reward is None:
                continue
            self.episode_rewards.append(reward)
            if   "SCAM recommended" in reason:  self.scam_hits     += 1
            elif "Scam skipped"     in reason:  self.scams_avoided += 1
            elif "Missed legit"     in reason:  self.deals_missed  += 1
            elif "Good recommendation" in reason: self.good_recs   += 1

        if (self.num_timesteps - self._last_log_step) >= self.log_interval:
            self._last_log_step = self.num_timesteps
            if self.episode_rewards:
                mean_r  = np.mean(self.episode_rewards[-500:])
                total   = self.scam_hits + self.scams_avoided + self.deals_missed + self.good_recs
                avoid_r = self.scams_avoided / max(total, 1) * 100
                scam_r  = self.scam_hits     / max(total, 1) * 100
                print(
                    f"  Step {self.num_timesteps:>7,} | "
                    f"Mean reward (last 500): {mean_r:>7.2f} | "
                    f"Scams avoided: {avoid_r:5.1f}% | "
                    f"Scam slips: {scam_r:5.1f}% | "
                    f"Good recs: {self.good_recs:>5,} | "
                    f"Missed deals: {self.deals_missed:>5,}"
                )
        return True   # returning False would stop training


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(csv_path: str = CSV_PATH) -> ShoppingEnv:
    """Wrapped in Monitor so SB3 can track episode stats."""
    env = ShoppingEnv(csv_path=csv_path, render_mode=None)
    return Monitor(env)


# ─────────────────────────────────────────────────────────────────────────────
# DQN hyperparameters (tuned for this environment)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model: DQN, csv_path: str, n_episodes: int = EVAL_EPISODES) -> dict:
    """
    Rolls out the trained DQN policy (no exploration) and returns a stats
    dict. Thin wrapper around evaluation.evaluate_policy() so the DQN and
    the contextual-bandit baseline (bandit_baseline.py) are scored with
    exactly the same rollout/metric logic — see evaluation.py for why that
    matters for a fair comparison.
    """
    predict_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])
    return evaluate_policy(predict_fn, csv_path=csv_path, n_episodes=n_episodes, label="DQN")


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(timesteps: int = 100_000, run_eval: bool = True) -> DQN:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))

    # Make sure the dataset exists
    if not os.path.exists(CSV_PATH):
        print(f"'{CSV_PATH}' not found — generating synthetic data first…")
        from data_generator import generate_dataset
        generate_dataset(n=5000).to_csv(CSV_PATH, index=False)
        print(f"Generated '{CSV_PATH}'.\n")

    print("=" * 65)
    print("  E-COMMERCE DEAL HUNTER — DQN TRAINING")
    print("=" * 65)
    print(f"  CSV path      : {CSV_PATH}")
    print(f"  Total timesteps: {timesteps:,}")
    print(f"  Network arch  : 128 → 128")
    print(f"  Gamma (discount): {DQN_HYPERPARAMS['gamma']}")
    print(f"  Replay buffer : {DQN_HYPERPARAMS['buffer_size']:,}")
    print(f"  Exploration   : {DQN_HYPERPARAMS['exploration_initial_eps']} → "
          f"{DQN_HYPERPARAMS['exploration_final_eps']} over "
          f"{DQN_HYPERPARAMS['exploration_fraction']*100:.0f}% of training")
    print("=" * 65)

    # ── Build environment ─────────────────────────────────────────────────────
    train_env = make_vec_env(make_env, n_envs=N_TRAIN_ENVS)

    # ── Instantiate DQN ───────────────────────────────────────────────────────
    model = DQN(env=train_env, **DQN_HYPERPARAMS)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    monitor_cb = TrainingMonitorCallback(log_interval=5_000)

    checkpoint_cb = CheckpointCallback(
        save_freq      = 10_000,
        save_path      = CHECKPOINT_DIR,
        name_prefix    = "dqn_shopping",
        save_replay_buffer = False,
        verbose        = 0,
    )

    # EvalCallback requires a separate env instance to avoid contamination
    eval_env = Monitor(ShoppingEnv(csv_path=CSV_PATH, render_mode=None))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = "./best_model/",
        log_path             = LOG_DIR,
        eval_freq            = 10_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining started…\n")
    t0 = time.time()

    model.learn(
        total_timesteps  = timesteps,
        callback         = [monitor_cb, checkpoint_cb, eval_cb],
        log_interval     = 1,          # SB3 internal logging (we suppress via verbose=0)
        progress_bar     = True,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/60:.1f} min).")

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved → {MODEL_SAVE_PATH}.zip")

    # ── Optional evaluation ───────────────────────────────────────────────────
    if run_eval:
        run_evaluation(model, CSV_PATH, n_episodes=EVAL_EPISODES)

    train_env.close()
    eval_env.close()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Loading helper (used by app.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_agent(model_path: str = f"{MODEL_SAVE_PATH}.zip") -> DQN:
    """
    Load a previously saved DQN model.
    Call this from app.py instead of re-training every session.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Run train_agent.py first."
        )
    model = DQN.load(model_path)
    print(f"Loaded DQN model from '{model_path}'.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tune on a small experience buffer (used by app.py online loop)
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune_on_feedback(
    model: DQN,
    experience_buffer: list[dict],
    gradient_steps: int = 50,
) -> DQN:
    """
    Accepts a list of experience dicts from the Streamlit feedback loop and
    injects them directly into the DQN's replay buffer, then runs a small
    number of gradient updates so the model adapts to user preferences.

    Each dict in experience_buffer must have:
        {
            "obs":      np.ndarray shape (4,),   # state before action
            "action":   int,                     # always 1 (Recommend)
            "reward":   float,                   # +20 Like / -20 Dislike
            "next_obs": np.ndarray shape (4,),   # state of next product (or zeros)
            "done":     bool,
        }
    """
    if not experience_buffer:
        return model

    buf = model.replay_buffer

    for exp in experience_buffer:
        obs      = np.array(exp["obs"],      dtype=np.float32).reshape(1, -1)
        next_obs = np.array(exp["next_obs"], dtype=np.float32).reshape(1, -1)
        action   = np.array([exp["action"]], dtype=np.int64)
        reward   = np.array([exp["reward"]], dtype=np.float32)
        done     = np.array([exp["done"]],   dtype=np.float32)

        buf.add(obs, next_obs, action, reward, done, [{}])

    # Only run gradient steps if the buffer has enough data
    if buf.size() >= model.batch_size:
        model.train(gradient_steps=gradient_steps, batch_size=model.batch_size)
        print(f"Fine-tuned on {len(experience_buffer)} feedback samples "
              f"({gradient_steps} gradient steps).")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DQN Deal Hunter agent.")
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps (default: 100,000)"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run a final evaluation after training"
    )
    args = parser.parse_args()

    train(timesteps=args.timesteps, run_eval=args.eval)
