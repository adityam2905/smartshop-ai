"""
Train the DQN on the synthetic listings and save it to models/.

    python -m smartshop.train                 # 100,000 steps (~2 min on CPU)
    python -m smartshop.train --eval          # …then score it
    python -m smartshop.train --timesteps 20000

Training is seeded (config.DQN_HYPERPARAMS["seed"]): short runs are
bit-for-bit repeatable, but a full run can end slightly differently each time
(multi-threaded CPU maths isn't exactly repeatable), so a retrained model
behaves like the committed one without being identical to it.
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from .config import BEST_MODEL_DIR, CHECKPOINT_DIR, DATA_CSV, DQN_HYPERPARAMS, LOG_DIR, MODEL_PATH
from .data_generator import ensure_dataset
from .environment import ShoppingEnv
from .evaluation import evaluate_policy

N_TRAIN_ENVS = 4
EVAL_EPISODES = 20


class ProgressPrinter(BaseCallback):
    """Prints how often the agent is avoiding scams and finding deals, every few thousand steps."""

    def __init__(self, every: int = 5_000):
        super().__init__()
        self.every = every
        self.rewards: list[float] = []
        self.counts = {"SCAM recommended": 0, "Scam skipped": 0, "Missed good deal": 0, "Good recommendation": 0}
        self._last = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "reward" not in info:
                continue
            self.rewards.append(info["reward"])
            for prefix in self.counts:
                if info["reward_reason"].startswith(prefix):
                    self.counts[prefix] += 1

        if self.num_timesteps - self._last >= self.every and self.rewards:
            self._last = self.num_timesteps
            c = self.counts
            scams = max(c["SCAM recommended"] + c["Scam skipped"], 1)
            print(f"  Step {self.num_timesteps:>7,} | mean reward (last 500) {np.mean(self.rewards[-500:]):>6.2f} | "
                  f"scams recommended {c['SCAM recommended'] / scams:5.1%} | "
                  f"good recs {c['Good recommendation']:>6,} | missed deals {c['Missed good deal']:>5,}")
        return True


def make_env():
    return Monitor(ShoppingEnv(csv_path=DATA_CSV))


def train(timesteps: int = 100_000, run_eval: bool = True) -> DQN:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))

    ensure_dataset(DATA_CSV)
    print(f"Training a DQN for {timesteps:,} steps on {DATA_CSV.name}…\n")
    train_env = make_vec_env(make_env, n_envs=N_TRAIN_ENVS, seed=DQN_HYPERPARAMS["seed"])
    model = DQN(env=train_env, **DQN_HYPERPARAMS)

    eval_env = Monitor(ShoppingEnv(csv_path=DATA_CSV))
    callbacks = [
        ProgressPrinter(),
        CheckpointCallback(save_freq=10_000, save_path=str(CHECKPOINT_DIR), name_prefix="dqn_shopping"),
        EvalCallback(eval_env, best_model_save_path=str(BEST_MODEL_DIR), log_path=LOG_DIR,
                     eval_freq=10_000, n_eval_episodes=5, deterministic=True, verbose=0),
    ]

    start = time.time()
    # The progress bar needs `rich`, which only comes with stable-baselines3[extra]
    # (deliberately not installed); without this check training crashes on startup.
    model.learn(total_timesteps=timesteps, callback=callbacks,
                progress_bar=importlib.util.find_spec("rich") is not None)
    print(f"\nTrained in {time.time() - start:.0f}s.")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"Saved to {MODEL_PATH}")

    if run_eval:
        evaluate_policy(lambda obs: int(model.predict(obs, deterministic=True)[0]),
                        csv_path=DATA_CSV, n_episodes=EVAL_EPISODES, label="DQN")

    train_env.close()
    eval_env.close()
    return model


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train the SmartShop DQN.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval", action="store_true", help="Score the model after training")
    args = parser.parse_args()
    train(timesteps=args.timesteps, run_eval=args.eval)


if __name__ == "__main__":
    main()
