# 🛍️ SmartShop: E-Commerce Deal Hunter & Scam Prevention RL Agent

[![CI](https://github.com/adityam2905/smartshop-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/adityam2905/smartshop-ai/actions/workflows/ci.yml)

**🔗 [Live demo](https://smartshop-ai.streamlit.app/)** — Streamlit Community Cloud
*(free-tier apps sleep when idle — give it a minute to wake up)*

A Streamlit app where a **Deep Q-Network (DQN)** screens product search
results: it recommends good deals, blocks scam sites, and adapts to your
👍 / 👎 feedback during the session.

See [Limitations](#limitations) for where the RL framing is doing more work
than the problem needs.

---

## Results

The committed model, evaluated on its training data and on two freshly
generated datasets it never saw (5,000 listings each):

| Metric | DQN | Linear bandit baseline |
|---|---|---|
| Scams recommended | **0%** | 0% |
| Legit deals missed | **0%** | 15.7% |

The decision is driven by **site trust**, not discount size: trusted and
small legit shops are recommended at any discount (5–95%), low-trust sites
are skipped at any discount. Look-alike domains such as `cheap-amazon.com`
are blocked.

These scores are perfect partly because trust separates scam from legit
perfectly in the synthetic data — see [Limitations](#limitations).

---

## Quick Start

```bash
pip install -r requirements.txt

python data_generator.py                           # 5,000 synthetic listings, 25% scam
python train_agent.py --timesteps 100000 --eval    # saves dqn_shopping_agent.zip (~2 min on CPU)
streamlit run app.py                               # uses mock data unless SERPAPI_KEY is set

# Optional
python scraper.py "Sony Headphones" --mock         # inspect engineered features
python bandit_baseline.py --compare-dqn            # baselines, see below
python supervised_baseline.py
```

**Live vs. demo data.** For live Google Shopping results, get a key at
[serpapi.com](https://serpapi.com) (100 free searches/month) and
`export SERPAPI_KEY="..."`. Pick the country (US, India, UK, Canada,
Australia) in the sidebar; prices show in that country's currency, and
`SERPAPI_COUNTRY=in` sets the default. Without a key, the app uses built-in
demo listings for headphones, shoes, iPhones, MacBooks and gaming chairs, each
a mix of real retailers and scam sites. If a live search fails (no key, quota
used up, network error), the app says why and shows demo data instead.

### Tests

```bash
pip install -r requirements-test.txt && pytest -v    # lean: no torch, runs in CI
pip install -r requirements.txt && pytest -v         # also runs model + online-learning tests
```

`tests/test_online_learning.py` needs torch, so CI skips it — run it locally
after retraining. `tests/test_model_artifact.py` runs in CI and fails if the
committed model wasn't trained with the settings in `agent_config.py`.

---

## How It Works

| File | Role |
|---|---|
| `data_generator.py` | Synthetic listings (legit + scam) → `product_listings.csv` |
| `shopping_env.py` | Gymnasium environment: observations, actions, rewards |
| `agent_config.py` | DQN hyperparameters (torch-free, so CI can check the committed model) |
| `train_agent.py` | DQN training, evaluation, and online fine-tuning |
| `dqn_shopping_agent.zip` | Trained model, committed so the deploy works without training |
| `scraper.py` | SerpAPI / mock fetch, feature engineering, domain trust scoring |
| `app.py` | Streamlit UI, inference, Like/Dislike loop |
| `evaluation.py` | Shared scoring harness for the DQN and the bandit |
| `bandit_baseline.py`, `supervised_baseline.py` | Baselines (see below) |

### RL design

**Observation:** `normalized_price` [0, 2] · `discount_percentage` [0, 1] ·
`site_trust_score` [0, 1] · `user_preference_score` [0, 1]
**Actions:** `0` Skip · `1` Recommend

| Situation | Reward |
|---|---|
| Recommend + Scam | **-100** |
| Recommend + Legit | `discount × 20 + user_feedback` |
| Skip + Scam | **+10** |
| Skip + Legit | **-5** |

**Hyperparameters:** 128→128 MLP · LR 5e-4 · γ 0.97 · batch 64 · replay
buffer 100k · ε 1.0 → 0.02 over 20% of training · target update every 1,000 steps.

**Training data:** discount ranges overlap on purpose — 25% of legit listings
are 40–85% clearance deals and 35% of scams use a believable 10–50% discount —
so the agent must learn from trust rather than discount size. 15% of legit
listings come from small shops with trust 0.40–0.70, matching what the live
scorer gives unknown domains.

### Domain trust

`scraper.py::compute_domain_trust()` scores each listing's URL:

1. **Known retailers** (Amazon incl. regional sites like `amazon.in`,
   Flipkart, Walmart, Best Buy, …) — matched on the exact registrable domain,
   so `smile.amazon.com` counts but `cheap-amazon.com` doesn't.
2. **Look-alikes** — a domain using a known brand name without being that
   brand (`cheap-amazon.com`, `amaz0n.com`, `flipkartsale.shop`) scores ~0.10.
3. **Everything else** — heuristics on TLD (`.xyz`, `.tk`, … score low) and
   scammy keywords (`deal`, `cheap`, `mega`, …).

`app.py` hard-blocks anything with trust < 0.3 before the DQN is consulted.

### Online learning

Every 3 Likes/Dislikes, the session's model runs 50 gradient steps on two
objectives:

- **Feedback:** move `Q(s, Recommend)` to the pretrained value ± 20 for each
  rated listing. This is bounded, so repeated Dislikes can't push past it.
- **Anchor:** keep `Q` equal to the pretrained model's on states sampled
  across the whole observation space, so unrated listings don't drift.

It trains on all of the session's feedback. Each browser session fine-tunes
its own copy; the shared pretrained model is only read. Likes/Dislikes also
shift a per-category preference score used as a feature on the next search.

---

## Baselines

**Contextual bandit** (`bandit_baseline.py`). Each reward depends only on
the current listing, and the next listing doesn't depend on the action, so
this is a contextual bandit, not a true MDP — γ, the replay buffer and the
target network have no temporal credit to assign. A from-scratch linear
bandit never recommends a scam but misses 15.7% of legit deals. That gap is
model capacity (linear vs. MLP), not RL; an MLP bandit is the fair next
comparison.

**Supervised classifiers** (`supervised_baseline.py`). On the plain
"is this a scam?" label, Logistic Regression (0.998 F1), Random Forest and
the `trust < 0.3` rule all score ~1.000 — trust separates the classes by
construction, so a classifier has nothing to add.

---

## Deployment

Hosted on [Streamlit Community Cloud](https://share.streamlit.io) from `main`
with main file `app.py`. No secrets are needed; add `SERPAPI_KEY` under
**Settings → Secrets** for live results.

- **Updating the model:** retrain, commit `dqn_shopping_agent.zip`, push,
  then **Reboot app** from the dashboard. The model is held in
  `st.cache_resource`, so a running app keeps the old one until it restarts.
- **Install size:** `requirements.txt` uses plain `stable-baselines3`, not
  `[extra]`, which pulls in opencv/pygame/Atari and roughly doubles install
  time. See the file for a CPU-only torch tip.

---

## Limitations

1. **It's a contextual bandit framed as an MDP** — see [Baselines](#baselines).
2. **Scam blocking is mostly a rule.** The trust < 0.3 hard filter runs
   before the DQN, and trust comes from a hand-built heuristic. It has known
   gaps: a scam-styled `.net` domain can score just above 0.3
   (`test_dot_net_scam_domain_is_not_reliably_flagged`).
3. **The synthetic data is still easy.** Trust separates scam (< 0.28) from
   legit (≥ 0.40) perfectly, which is why every model scores ~100%. Real
   listings, or scams with mid-range trust, would be a real benchmark.
4. **Preferences are in-memory** and reset when the session ends.
5. **Live search shares one quota.** Every visitor to the public demo uses
   the same 100 searches/month; once they run out, everyone gets demo data
   (with a warning) until the quota resets.

### Fixed bugs

- **Online learning never worked:** it trained nothing for 126 clicks (empty
  replay buffer after loading), crashed at click 129 (loaded SB3 models have
  no logger), and ~128 Dislikes made the model skip everything. Replaced with
  the approach [above](#online-learning).
- **Stale model:** the deployed model was trained with different LR, γ, batch
  size and network size than documented. Retrained; now checked in CI.
- **Discount shortcut:** non-overlapping discount ranges taught the agent
  "small discount = safe", so it skipped real 75%+ deals and recommended
  low-trust scams at 5–20% off.
- **Look-alike domains:** `endswith("amazon.com")` gave `cheap-amazon.com`
  full trust, while `amazon.in` and Flipkart were unknown.
- **Shared state across users:** one visitor's feedback retrained the model
  everyone used, and preferences leaked between sessions.
- **Stale search cache:** preference changes didn't invalidate cached results.
- **UI:** quick-search chips crashed the app, and the "Force mock data"
  checkbox reset itself on every rerun.
- **Training crash:** `train_agent.py` required `rich` for its progress bar,
  which isn't installed without `[extra]`.
