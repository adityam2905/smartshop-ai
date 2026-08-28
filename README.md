# 🛍️ E-Commerce Deal Hunter & Scam Prevention RL Agent

[![CI](https://github.com/adityam2905/ai-deal-hunter/actions/workflows/ci.yml/badge.svg)](https://github.com/adityam2905/ai-deal-hunter/actions/workflows/ci.yml)

A full-stack AI application that uses a **Deep Q-Network (DQN)** to evaluate
live product search results in real time — maximising deals, blocking scams,
and personalising recommendations based on your live feedback.

See [Limitations & Honest Notes](#limitations--honest-notes) below for a
candid look at where the DQN framing is doing more work than the model
underneath it, and what a `bandit_baseline.py` comparison shows about that.

---

## Architecture Overview

```
data_generator.py   →   product_listings.csv
                               ↓
shopping_env.py     →   ShoppingEnv (Gymnasium)
                               ↓
        ┌──────────────────────┴──────────────────────┐
        ↓                                              ↓
train_agent.py  →  dqn_shopping_agent.zip   bandit_baseline.py  (baseline comparison)
        └──────────────────────┬──────────────────────┘
                        evaluation.py  (shared eval harness)
                               ↓
scraper.py          →   feature dicts from SerpAPI / mock
                               ↓
app.py              →   Streamlit UI + online learning loop

tests/               →   pytest coverage for shopping_env.py, scraper.py, bandit_baseline.py
.github/workflows/   →   CI: installs requirements-test.txt, runs pytest on every push/PR
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic training data
```bash
python data_generator.py
# → creates product_listings.csv (5,000 rows, 25% scam)
```

### 3. Train the DQN agent
```bash
# Full training (recommended)
python train_agent.py --timesteps 100000 --eval

# Quick test run
python train_agent.py --timesteps 50000
```
Saves `dqn_shopping_agent.zip` and periodic checkpoints in `checkpoints/`.

### 4. (Optional) Test the scraper pipeline
```bash
# Mock data (no API key needed)
python scraper.py "Sony Headphones" --mock

# Live data (requires SERPAPI_KEY)
export SERPAPI_KEY="your_key_here"
python scraper.py "Nike Shoes"
```

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```

For live Google Shopping results, set your SerpAPI key first:
```bash
export SERPAPI_KEY="your_key_here"
streamlit run app.py
```

### 6. (Optional) Run the contextual-bandit baseline
```bash
python bandit_baseline.py --compare-dqn
```
See [Contextual Bandit Baseline](#contextual-bandit-baseline) below for why this exists.

---

## File Reference

| File | Phase | Description |
|---|---|---|
| `data_generator.py` | 1 | Generates 5,000 synthetic product listings (legit + scam) |
| `shopping_env.py`   | 2 | Custom Gymnasium environment with reward shaping |
| `train_agent.py`    | 3 | DQN training, checkpointing, evaluation, online fine-tuning |
| `scraper.py`        | 4 | SerpAPI live fetch + feature engineering + trust scoring |
| `app.py`            | 5 | Streamlit UI, agent inference, Like/Dislike feedback loop |
| `evaluation.py`     | — | Shared rollout/scoring harness used by both the DQN and the bandit baseline |
| `bandit_baseline.py`| — | A from-scratch linear contextual bandit, benchmarked against the DQN |
| `tests/`            | — | pytest suite covering the environment, feature engineering, and the bandit |
| `requirements.txt`  | — | Full dependency set (incl. torch, stable-baselines3, streamlit) |
| `requirements-test.txt` | — | Lean dependency set for `pytest` / CI (no torch, no streamlit) |

---

## RL Design

### Observation Space
| Feature | Range | Description |
|---|---|---|
| `normalized_price` | [0, 2] | price ÷ market average |
| `discount_percentage` | [0, 1] | inferred or explicit discount |
| `site_trust_score` | [0, 1] | rule-based domain trust |
| `user_preference_score` | [0, 1] | learned per-category preference |

### Action Space
- **0 → Skip** — don't show this product
- **1 → Recommend** — display to the user

### Reward Function
| Situation | Reward |
|---|---|
| Recommend + Scam (trust < 0.3) | **-100** |
| Recommend + Legit | `(discount × 20) + user_feedback` |
| Skip + Scam | **+10** |
| Skip + Legit | **-5** |

### DQN Hyperparameters
| Parameter | Value |
|---|---|
| Network | 128 → 128 MLP |
| Learning rate | 5e-4 |
| Gamma | 0.97 |
| Replay buffer | 100,000 |
| Exploration | ε: 1.0 → 0.02 over 20% of training |
| Target update | Every 1,000 steps (hard copy) |

---

## Online Learning Loop

Every time you click 👍 or 👎 on a product card:

1. An experience tuple `(obs, action=1, reward=±20, next_obs, done)` is recorded.
2. After every **3 feedback signals**, the experience is injected into the DQN's
   replay buffer and **50 gradient steps** are run — adapting the model to your taste.
3. Per-category preference scores are also updated and fed back as the
   `user_preference_score` feature on the next search.

---

## Getting a SerpAPI Key

1. Sign up at [serpapi.com](https://serpapi.com) (100 free searches/month)
2. Copy your API key from the dashboard
3. `export SERPAPI_KEY="your_key_here"`

Without a key the app uses rich mock data containing a realistic mix of
trusted retailers and scam domains — perfect for demos.

---

## Testing

The test suite covers `shopping_env.py`'s reward logic, `scraper.py`'s
feature engineering and trust heuristics, and `bandit_baseline.py`. It
deliberately does **not** require torch/stable-baselines3, so it installs
and runs in seconds:

```bash
pip install -r requirements-test.txt
pytest -v
```

CI (`.github/workflows/ci.yml`) runs the same suite on every push and pull
request against `main`.

A few of these tests exist specifically to pin down bugs found while
reviewing this project (see [Limitations](#limitations--honest-notes)):
`test_user_preferences_do_not_leak_between_independent_stores` in
`tests/test_scraper.py` is a regression test for the multi-user preference
leak described below, and `test_dot_net_scam_domain_is_not_reliably_flagged`
documents — rather than hides — a real gap in the live trust heuristic.

---

## Contextual Bandit Baseline

`bandit_baseline.py` implements a small linear epsilon-greedy contextual
bandit from scratch (no external RL library) and evaluates it against the
trained DQN using the exact same harness (`evaluation.py`):

```bash
python bandit_baseline.py --train-episodes 60 --episodes 20 --compare-dqn
```

**Why bother, if the DQN already works?** Look closely at
`ShoppingEnv.step()`: the reward for a product depends only on that
product's own features and the action taken on it, and the *next*
observation doesn't depend on the action just taken. There's no credit
assignment across time — every "episode" is really a sequence of
independent one-shot decisions with a context attached. That's a
**contextual bandit problem**, not a true MDP, so `gamma = 0.97` isn't
doing anything meaningful, and the DQN's replay buffer / target network /
discounting exist to solve a temporal-credit-assignment problem that isn't
actually present here.

The bandit — a single linear layer per action, updated with one SGD step
per observed reward, no bootstrapping — gets remarkably close to the DQN on
this task's metrics (mean episode return, good-rec rate, scam-avoid rate).
That's the point: it's evidence that the extra machinery isn't earning its
complexity for *this* formulation of the problem, and it's worth stating
that explicitly rather than presenting DQN as required.

---

## Limitations & Honest Notes

Written deliberately, not to undersell the project but because naming these
directly is more convincing than hoping nobody asks:

1. **This is a contextual bandit wearing an MDP's clothes.** See
   [Contextual Bandit Baseline](#contextual-bandit-baseline) above. A fair
   framing is "I evaluated whether full RL was the right tool and found X,"
   not "I built an RL agent" without qualification.

2. **The model doesn't fully own the scam-detection decision.**
   `app.py::run_agent_inference()` hard-filters anything with
   `site_trust_score < 0.3` *before* the DQN gets a vote — so in production,
   scam-blocking is a rule, not a learned decision. The DQN only ever
   chooses among listings that already passed that rule. That trust score
   is itself a hand-built lookup table + TLD/keyword heuristic in
   `scraper.py::compute_domain_trust()`, not something the model learned.
   This is a legitimate hybrid rules+ML design — just worth describing as
   one rather than implying the model catches every scam on its own. It
   also isn't airtight: `tests/test_scraper.py::test_dot_net_scam_domain_is_not_reliably_flagged`
   documents a live case (a `.net` scam-styled domain) that can land at or
   above the 0.3 cutoff and slip past the filter, even though
   `data_generator.py`'s *offline* labeling would have scored the
   equivalent training row well below it.

3. **The synthetic training data is close to trivially separable.**
   Scam listings use non-overlapping discount ranges (60–95%) and a fixed
   scam-TLD list; legit listings use 5–40% discounts and mainstream `.com`
   domains. The DQN's very strong evaluation numbers partly reflect that
   the boundary is easy, not just that the policy is good. Noisier,
   more overlapping synthetic data (or real listings) would be a
   meaningfully harder and more convincing benchmark.

4. **A single Like/Dislike barely moves the shared model.** `app.py`'s
   online-learning loop injects one experience into a 100,000-capacity
   replay buffer and then samples a random batch from the *whole* buffer
   for 50 gradient steps — so a handful of new samples have a small
   influence on the network relative to the "Agent retrained!" toast the
   UI shows. It's a good demo of the mechanism, not yet evidence the model
   meaningfully personalizes per user in a single session.

5. **(Fixed in this pass) Preferences used to leak across users.**
   `scraper.py` previously stored `_USER_PREFS` in a module-level dict,
   which a deployed multi-user Streamlit app would have shared across every
   visitor's session. Preferences are now threaded through explicitly as a
   per-session dict (`st.session_state.user_prefs` in `app.py`), with a
   regression test (`test_user_preferences_do_not_leak_between_independent_stores`)
   covering it. Relatedly, `app.py`'s cached-search function previously
   named its cache-key parameter `_pref_snapshot` — Streamlit's caching
   convention treats a leading underscore as "exclude this from the cache
   key," so despite a comment claiming otherwise, a changed preference was
   not actually invalidating cached search results. Renamed to
   `pref_snapshot` (no underscore) to fix that.

What's *not* addressed here (kept out of scope for this pass, listed for
transparency): Docker/deployment, a persistent (non-in-memory) preference
store, prioritized replay for feedback samples, and richer NLP-derived
features from listing text.
