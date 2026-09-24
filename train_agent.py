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
import importlib.util
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
from agent_config import DQN_HYPERPARAMS, LOG_DIR

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH  = "dqn_shopping_agent"          # SB3 appends .zip automatically
CHECKPOINT_DIR   = "checkpoints/"
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


# DQN hyperparameters live in agent_config.py (torch-free, so CI can check
# the committed model against them) — imported above.


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
    print(f"  Network arch  : {' → '.join(map(str, DQN_HYPERPARAMS['policy_kwargs']['net_arch']))}")
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
        # SB3's progress bar needs `rich`, which only comes with
        # stable-baselines3[extra] (deliberately not installed — see
        # requirements.txt). Without this guard training dies on startup.
        progress_bar     = importlib.util.find_spec("rich") is not None,
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

N_ANCHOR_STATES   = 2_048
ANCHOR_BATCH_SIZE = 64


def make_anchor_states(model: DQN, n: int = N_ANCHOR_STATES, seed: int = 0) -> np.ndarray:
    """States spread uniformly over the observation space, used to pin the
    fine-tuned Q-function to the pretrained one everywhere it wasn't rated."""
    space = model.observation_space
    rng = np.random.default_rng(seed)
    return rng.uniform(space.low, space.high, size=(n, space.shape[0])).astype(np.float32)


def fine_tune_on_feedback(
    model: DQN,
    teacher: DQN,
    feedback: list[dict],
    gradient_steps: int = 50,
    anchor_states: np.ndarray | None = None,
    seed: int = 0,
) -> int:
    """
    Adapts `model`'s Q-network to Like/Dislike feedback in place and returns
    the number of gradient steps taken (0 if there was nothing to learn).

    `teacher` is the untouched pretrained model; it is only read, never
    trained, so it's safe to pass the process-wide cached model from app.py.

    Each dict in `feedback` must have:
        {
            "obs":    np.ndarray shape (4,),   # state that was rated
            "action": int,                     # always 1 (Recommend)
            "reward": float,                   # +20 Like / -20 Dislike
        }

    Why this doesn't go through SB3's replay buffer + model.train():
      * A loaded DQN starts with an empty replay buffer (SB3 doesn't save
        it), so nothing trained until the buffer reached batch_size — ~128
        clicks — while the UI claimed "Agent retrained!".
      * model.train() on a loaded model then crashed: its logger is only
        created inside learn().
      * Training on feedback alone, with a target of `reward` for a
        terminal transition, dragged Q(s, Recommend) toward -20 for every
        state; a few dozen Dislikes made the agent skip everything.

    Instead, each gradient step minimises two terms:
      1. Feedback: Q(s, a) → Q_teacher(s, a) + reward. In ShoppingEnv, user
         feedback is a bonus added to the Recommend reward, so this is the
         pretrained value shifted by exactly that bonus. The target is
         bounded, so repeating the same Dislike can't push past it.
      2. Anchor: Q(s', ·) → Q_teacher(s', ·) on states sampled across the
         observation space, so everything that wasn't rated keeps its
         pretrained behaviour (no catastrophic forgetting).
    """
    if not feedback:
        return 0

    device = model.device
    if anchor_states is None:
        anchor_states = make_anchor_states(model)

    obs_fb  = torch.as_tensor(np.stack([f["obs"] for f in feedback]), dtype=torch.float32, device=device)
    act_fb  = torch.as_tensor([int(f["action"]) for f in feedback], dtype=torch.long, device=device)
    rew_fb  = torch.as_tensor([float(f["reward"]) for f in feedback], dtype=torch.float32, device=device)
    anchors = torch.as_tensor(anchor_states, dtype=torch.float32, device=device)

    with torch.no_grad():
        target_fb = teacher.q_net(obs_fb).clone()
        target_fb[torch.arange(len(feedback), device=device), act_fb] += rew_fb
        target_anchor = teacher.q_net(anchors)

    q_net     = model.q_net
    optimizer = model.policy.optimizer
    rng       = np.random.default_rng(seed)

    model.policy.set_training_mode(True)
    for _ in range(gradient_steps):
        idx  = torch.as_tensor(rng.integers(0, len(anchors), size=ANCHOR_BATCH_SIZE), device=device)
        loss = (
            torch.nn.functional.smooth_l1_loss(q_net(obs_fb), target_fb)
            + torch.nn.functional.smooth_l1_loss(q_net(anchors[idx]), target_anchor[idx])
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), model.max_grad_norm)
        optimizer.step()
    model.policy.set_training_mode(False)

    # Keep the target network consistent with the online one
    model.q_net_target.load_state_dict(q_net.state_dict())
    return gradient_steps


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        # train() prints progress lines containing "→"; on Windows the
        # default console codepage (cp1252) can't encode that character and
        # this script would crash with a UnicodeEncodeError otherwise.
        sys.stdout.reconfigure(encoding="utf-8")

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
