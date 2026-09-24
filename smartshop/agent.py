"""
The trained model at run time: loading it, and adapting it to a user's 👍 / 👎.
"""

import numpy as np
import torch
from stable_baselines3 import DQN

from .config import MODEL_PATH

N_ANCHOR_STATES = 2_048
ANCHOR_BATCH_SIZE = 64


def load_agent(model_path=MODEL_PATH) -> DQN:
    """Load the saved DQN (train one with `python -m smartshop.train`)."""
    try:
        return DQN.load(str(model_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"No trained model at {model_path} — run `python -m smartshop.train`.") from None


def make_anchor_states(model: DQN, n: int = N_ANCHOR_STATES, seed: int = 0) -> np.ndarray:
    """States spread evenly over the observation space, used to keep the
    fine-tuned model equal to the pretrained one everywhere that wasn't rated."""
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
    Adapt `model` to the user's ratings, in place. Returns the number of
    gradient steps taken (0 if there was no feedback).

    `feedback` items are {"obs": 4 floats, "action": 1, "reward": +20 / −20}.
    `teacher` is the untouched pretrained model — only read, so it's safe to
    pass the model shared across all sessions.

    Each step pulls on two things:
      1. Feedback: Q(s, a) → teacher's Q(s, a) + reward, for each rated listing.
         The target is fixed, so repeating a Dislike can't push past it.
      2. Anchor: Q → teacher's Q on states across the whole observation space,
         so nothing that wasn't rated changes.

    (SB3's own replay buffer + model.train() is not used: a loaded model starts
    with an empty buffer, crashes in train() because it has no logger, and
    training on feedback alone made the model forget everything else.)
    """
    if not feedback:
        return 0

    device = model.device
    if anchor_states is None:
        anchor_states = make_anchor_states(model)

    obs = torch.as_tensor(np.stack([f["obs"] for f in feedback]), dtype=torch.float32, device=device)
    actions = torch.as_tensor([int(f["action"]) for f in feedback], dtype=torch.long, device=device)
    rewards = torch.as_tensor([float(f["reward"]) for f in feedback], dtype=torch.float32, device=device)
    anchors = torch.as_tensor(anchor_states, dtype=torch.float32, device=device)

    with torch.no_grad():
        feedback_target = teacher.q_net(obs).clone()
        feedback_target[torch.arange(len(feedback), device=device), actions] += rewards
        anchor_target = teacher.q_net(anchors)

    q_net, optimizer = model.q_net, model.policy.optimizer
    rng = np.random.default_rng(seed)

    model.policy.set_training_mode(True)
    for _ in range(gradient_steps):
        idx = torch.as_tensor(rng.integers(0, len(anchors), size=ANCHOR_BATCH_SIZE), device=device)
        loss = (
            torch.nn.functional.smooth_l1_loss(q_net(obs), feedback_target)
            + torch.nn.functional.smooth_l1_loss(q_net(anchors[idx]), anchor_target[idx])
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), model.max_grad_norm)
        optimizer.step()
    model.policy.set_training_mode(False)

    model.q_net_target.load_state_dict(q_net.state_dict())   # keep the target network in sync
    return gradient_steps
