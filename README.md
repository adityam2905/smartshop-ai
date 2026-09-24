# 🛍️ SmartShop: E-Commerce Deal Hunter & Scam Prevention RL Agent

[![CI](https://github.com/adityam2905/smartshop-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/adityam2905/smartshop-ai/actions/workflows/ci.yml)

**🔗 [Live demo](https://smartshop-ai.streamlit.app/)** — deployed on Streamlit Community Cloud
*(free-tier apps sleep after inactivity — give it a minute to wake up)*

A full-stack app that uses a **Deep Q-Network (DQN)** to evaluate live product
search results in real time: maximising deals, blocking scams, and
personalising recommendations from your Like/Dislike feedback.

Read [Limitations & Honest Notes](#limitations--honest-notes) for a candid
look at where the DQN framing does more work than the model underneath it,
and for the multi-session bugs found and fixed while reviewing this repo.

---

## Architecture

```
data_generator.py   →   product_listings.csv
                               ↓
shopping_env.py     →   ShoppingEnv (Gymnasium)
                               ↓
        ┌──────────────────────┼──────────────────────┐
        ↓                      ↓                      ↓
train_agent.py  →  bandit_baseline.py    supervised_baseline.py
dqn_shopping_agent.zip   (bandit baseline)   (LogReg / RandomForest
        │                      │              scam classifiers)
        └──────────────────────┼──────────────────────┘
                        evaluation.py  (shared eval harness, DQN + bandit)
                               ↓
scraper.py          →   feature dicts from SerpAPI / mock
                               ↓
app.py               →   Streamlit UI + online learning loop  →  live demo

tests/                →   pytest coverage for env, scraper, both baselines
.github/workflows/    →   CI: pytest + smoke-tests both baseline scripts
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data (5,000 rows, 25% scam)
python data_generator.py

# 3. Train the DQN agent — saves dqn_shopping_agent.zip + checkpoints/
python train_agent.py --timesteps 100000 --eval     # full run
python train_agent.py --timesteps 50000             # quick test

# 4. (Optional) Test the scraper
python scraper.py "Sony Headphones" --mock           # no API key needed
export SERPAPI_KEY="your_key_here" && python scraper.py "Nike Shoes"

# 5. Launch the app
export SERPAPI_KEY="your_key_here"   # optional — falls back to mock data
streamlit run app.py

# 6. (Optional) Baselines
python bandit_baseline.py --compare-dqn
python supervised_baseline.py
```

See [Contextual Bandit Baseline](#contextual-bandit-baseline) and
[Supervised Baseline](#supervised-baseline) for why steps 6 exist.

---

## File Reference

| File | Phase | Description |
|---|---|---|
| `data_generator.py` | 1 | Generates 5,000 synthetic product listings (legit + scam) |
| `shopping_env.py`   | 2 | Custom Gymnasium environment with reward shaping |
| `train_agent.py`    | 3 | DQN training, checkpointing, evaluation, online fine-tuning |
| `agent_config.py`   | 3 | DQN hyperparameters (torch-free, so CI can check the committed model against them) |
| `scraper.py`        | 4 | SerpAPI live fetch + feature engineering + trust scoring |
| `app.py`            | 5 | Streamlit UI, agent inference, Like/Dislike feedback loop |
| `dqn_shopping_agent.zip` | — | Pretrained DQN weights, checked into the repo so a fresh deploy works without retraining (see [Deployment](#deployment)) |
| `evaluation.py`     | — | Shared rollout/scoring harness used by both the DQN and the bandit baseline |
| `bandit_baseline.py`| — | A from-scratch linear contextual bandit, benchmarked against the DQN |
| `supervised_baseline.py` | — | Logistic Regression / Random Forest scam classifiers, benchmarked against the app's hard rule |
| `tests/`            | — | pytest suite covering the environment, feature engineering, and both baselines |
| `requirements.txt`  | — | Full dependency set (torch, stable-baselines3, streamlit — trimmed, see file comments) |
| `requirements-test.txt` | — | Lean dependency set for `pytest` / CI (no torch, no streamlit) |

---

## RL Design

**Observation** — 4 continuous features:

| Feature | Range | Description |
|---|---|---|
| `normalized_price` | [0, 2] | price ÷ market average |
| `discount_percentage` | [0, 1] | inferred or explicit discount |
| `site_trust_score` | [0, 1] | rule-based domain trust |
| `user_preference_score` | [0, 1] | learned per-category preference |

**Action** — `0` Skip · `1` Recommend

**Reward:**

| Situation | Reward |
|---|---|
| Recommend + Scam (trust < 0.3) | **-100** |
| Recommend + Legit | `(discount × 20) + user_feedback` |
| Skip + Scam | **+10** |
| Skip + Legit | **-5** |

**DQN hyperparameters:** 128→128 MLP · LR 5e-4 · γ 0.97 · batch 64 · replay buffer 100k ·
ε 1.0 → 0.02 over 20% of training · hard target update every 1,000 steps.
Defined once in `agent_config.py`; `tests/test_model_artifact.py` fails if the
committed `dqn_shopping_agent.zip` was trained with anything else.

---

## Online Learning Loop

Every 👍 / 👎 on a product card:

1. Records a feedback sample `(obs, action=1, reward=±20)`.
2. Every **3 feedback signals**, runs **50 gradient steps** on the Q-network
   (`train_agent.py::fine_tune_on_feedback`). Each step minimises two terms:
   - **Feedback:** `Q(s, Recommend) → Q_pretrained(s, Recommend) ± 20`. This
     is the environment's own definition of feedback (a bonus on the
     Recommend reward), and it is bounded, so repeating a Dislike can't push
     past it.
   - **Anchor:** `Q(s', ·) → Q_pretrained(s', ·)` on states sampled across the
     whole observation space, so nothing the user didn't rate drifts away
     from the pretrained policy.
   It trains on the whole session's feedback, not just the latest three.
3. Updates the per-category preference score, which feeds back as the
   `user_preference_score` feature on the next search.

The pretrained model is only read here, never trained, so it's shared
safely across sessions.

Each browser session fine-tunes its own private copy of the model — see
[Limitations #4](#limitations--honest-notes) for why that isolation matters
and how it's enforced.

---

## Getting a SerpAPI Key

1. Sign up at [serpapi.com](https://serpapi.com) (100 free searches/month)
2. Copy your API key from the dashboard
3. `export SERPAPI_KEY="your_key_here"`

Without a key the app uses rich mock data — a realistic mix of trusted
retailers and scam domains, ideal for demos.

---

## Testing

```bash
pip install -r requirements-test.txt
pytest -v
```

Covers `shopping_env.py`'s reward logic, `scraper.py`'s feature engineering
and trust heuristics, and both baselines — no torch/stable-baselines3
required, so it installs and runs in seconds. CI runs the same suite on
every push/PR against `main`. `tests/test_online_learning.py` checks the
committed model's behaviour and the Like/Dislike fine-tuning loop. It needs
the full `requirements.txt`, so it's skipped under `requirements-test.txt`;
run it locally after retraining.

A few tests pin down bugs found while reviewing this project:
`test_user_preferences_do_not_leak_between_independent_stores` regression-tests
the multi-session preference leak in [#5](#limitations--honest-notes), and
`test_dot_net_scam_domain_is_not_reliably_flagged` documents — rather than
hides — a real gap in the live trust heuristic ([#2](#limitations--honest-notes)).

---

## Contextual Bandit Baseline

`bandit_baseline.py` implements a small linear epsilon-greedy contextual
bandit from scratch (no external RL library) and evaluates it against the
trained DQN using the same harness (`evaluation.py`):

```bash
python bandit_baseline.py --train-episodes 60 --episodes 20 --compare-dqn
```

**Why, if the DQN already works?** In `ShoppingEnv.step()`, the reward for
a product depends only on that product's own features and the action taken
— the *next* observation doesn't depend on the action just taken. There's
no credit assignment across time: every "episode" is really a sequence of
independent one-shot decisions. That's a **contextual bandit problem**, not
a true MDP, so `gamma = 0.97` isn't doing anything meaningful, and the
DQN's replay buffer / target network exist to solve a temporal-credit
problem that isn't actually present here.

The bandit is one linear layer per action, with one SGD step per reward and
no bootstrapping. It never recommends a scam, but it misses 15.7% of legit
deals where the DQN misses none (20 eval episodes, `--train-episodes 60`).
That gap comes from the model's capacity, not from RL. Discount no longer
predicts scams (see [Limitations #3](#limitations--honest-notes)), and a
linear score can't express "recommend if trusted, *whatever* the discount"
as cleanly as a 2-layer MLP can. A non-linear bandit (same MLP, no
bootstrapping) is the fair next comparison. Until then, the replay buffer,
target network and γ still have no temporal-credit problem to solve here.

---

## Supervised Baseline

The bandit comparison is about the *policy* task (Recommend vs. Skip, which
mixes "is this a scam" with "is this deal good enough"). `supervised_baseline.py`
asks the narrower, safety-critical question: on the binary "is this a
scam?" label, how does a plain classifier compare to the hard-coded
`site_trust_score < 0.3` rule `app.py` ships with?

```bash
python supervised_baseline.py
```

Trains Logistic Regression and Random Forest on the same 4 features the RL
agent sees and reports precision/recall/F1 next to the hard rule. All three
still land at ~1.000 on the current dataset (Logistic Regression 0.998 F1).
That's a **data-quality** finding, not a model-quality one (see
[Limitations #3](#limitations--honest-notes)): `site_trust_score` is still a
perfect separator by construction, so there's no signal left for a
classifier to add over the hard rule. Discount and price are no longer
separators on their own.

---

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) (free tier):

1. Push this repo to GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub →
   authorize repo access.
3. **New app** → this repo, branch `main`, main file `app.py` → deploy.
4. No secrets required — without `SERPAPI_KEY` the app falls back to the
   mock dataset (no API quota to run out, guaranteed mix of legit + scam
   listings). Add `SERPAPI_KEY` under **Settings → Secrets** for live
   Google Shopping results.

Two things that make this work on the free tier:

- **The trained model is checked into the repo** (`dqn_shopping_agent.zip`,
  ~300KB) — `app.py` hard-fails without it, and a fresh deploy only has
  what's in git. A larger model would need Git LFS or a model registry.
- **`requirements.txt` skips `stable-baselines3[extra]`** — it pulls in
  opencv-python, pygame, and Atari packages this project never touches,
  roughly doubling install time for nothing. See the file's comments for a
  CPU-only torch wheel trick if a build is still slow.

Live at **https://smartshop-ai.streamlit.app/**.

---

## Limitations & Honest Notes

Named directly because that's more convincing than hoping nobody asks:

1. **This is a contextual bandit wearing an MDP's clothes.** See
   [Contextual Bandit Baseline](#contextual-bandit-baseline). Fair framing:
   "I evaluated whether full RL was the right tool and found X," not "I
   built an RL agent" unqualified.

2. **The model doesn't fully own scam detection.**
   `app.py::run_agent_inference()` hard-filters `site_trust_score < 0.3`
   *before* the DQN gets a vote — scam-blocking is a rule, not a learned
   decision, and that trust score is itself a hand-built heuristic in
   `scraper.py::compute_domain_trust()`. A legitimate hybrid rules+ML
   design, but not airtight: `test_dot_net_scam_domain_is_not_reliably_flagged`
   documents a live case (a `.net` scam-styled domain) that can land at or
   above the cutoff and slip past the filter.

   **(Fixed)** Known-domain lookup used `domain.endswith("amazon.com")`, so
   `cheap-amazon.com` got Amazon's full trust (0.999). Lookup now matches
   the registrable domain exactly, so real subdomains like `smile.amazon.com`
   still count. Any other domain that contains a known retailer's name
   (`cheap-amazon.com`, `amaz0n.com`, `flipkartsale.shop`) is treated as a
   look-alike and scored below the scam threshold. Regional storefronts
   (`amazon.in`, `amazon.co.uk`, …) and major Indian retailers (Flipkart,
   Myntra, Croma, …) are now in the trust table too.

3. **(Partly fixed) The synthetic training data was separable on the wrong
   feature.** Scam discounts used to be 60–95% and legit discounts 5–40%, so
   the agent learned "small discount = good" instead of "trusted site =
   good". It skipped real Amazon listings at 75%+ off and recommended
   low-trust scams offering 5–20% off. Now 25% of legit listings are 40–85%
   clearance deals, and 35% of scams use a believable 10–50% discount.
   Discount alone predicts scams only slightly better than always guessing
   "legit" (`tests/test_data_generator.py`). 15% of legit listings also come
   from small shops with trust 0.40–0.70, the range the live scorer gives
   unknown domains. The retrained agent's decision now flips on trust, at
   any discount (`tests/test_online_learning.py`).

   Still true: trust itself separates the classes perfectly by
   construction (scam < 0.28 ≤ legit), which is why the DQN evaluates at
   0 scam slips / 0 missed deals. Real listings, or scams with
   mid-range trust, would be a meaningfully harder benchmark.

4. **(Fixed) Online learning didn't work.** The old loop pushed feedback
   into SB3's replay buffer and called `model.train()`. A loaded model
   starts with an empty buffer (SB3 doesn't save it), so the first 126
   clicks trained nothing while the UI still said "Agent retrained!".
   Click 129 then crashed, because a loaded model has no logger until
   `learn()` runs. And the training target (`reward = -20`, terminal) had
   nothing holding the rest of the Q-function in place, so ~128 Dislikes
   made the agent skip every listing. See
   [Online Learning Loop](#online-learning-loop) for the replacement.
   `tests/test_online_learning.py` covers all three failures (it needs
   torch, so it runs locally, not in the lean CI job).

   Related bug, also **fixed**: `app.py` loads the DQN through
   `st.cache_resource`, which shares one model object across *every*
   visitor's session — by design, that's the point of `cache_resource`.
   But `handle_feedback()` fine-tunes by mutating that model in place
   (`replay_buffer.add()` + `model.train()`), so before this fix, one
   user's feedback silently retrained the model everyone else's session
   used for inference. `app.py::main()` now deep-copies the cached model
   into `st.session_state.model` once per session before any fine-tuning
   happens, so each session only ever mutates its own private copy.

5. **(Fixed) Preferences used to leak across users.** `scraper.py`
   previously stored preferences in a module-level dict shared by every
   visitor. Now threaded through explicitly as a per-session dict
   (`st.session_state.user_prefs`), with a regression test
   (`test_user_preferences_do_not_leak_between_independent_stores`).
   Relatedly, `app.py`'s cached-search function previously named its
   cache-key parameter `_pref_snapshot` — Streamlit's caching convention
   treats a leading underscore as "exclude from the cache key," so a
   changed preference wasn't actually invalidating cached results despite
   a comment claiming otherwise. Renamed to `pref_snapshot` to fix that.

6. **A supervised classifier doesn't beat the hard rule here**, and that's
   informative, not a null result. See
   [Supervised Baseline](#supervised-baseline): all three approaches score
   ~1.000 on the current dataset, the trust-separability point from #3 seen
   from a different angle.

7. **(Fixed) The live demo ran a stale model.** The committed
   `dqn_shopping_agent.zip` had been trained with LR 1e-3, γ 0.95, batch 128
   and a 128→128→64 network, none of which matched `train_agent.py` or this
   README. It has been retrained with the documented settings, and
   `tests/test_model_artifact.py` (torch-free, runs in CI) now fails if the
   two drift apart again. The retrain also exposed that `train_agent.py`
   crashed on startup: SB3's progress bar needs `rich`, which went away when
   `[extra]` was dropped from `requirements.txt`. The progress bar is now
   used only when `rich` is installed.

**Also fixed while reviewing this repo:** the search bar crashed
(`StreamlitAPIException`) whenever a quick-search chip was clicked, because
`app.py` reassigned `st.session_state.search_query` *after* the
`text_input` widget bound to that key had already rendered in the same run
— Streamlit disallows that. The chip buttons now set the query via an
`on_click` callback instead, which runs before the widget re-renders. The
sidebar's "Force mock data" checkbox had a similar self-defeating bug: it
was reset from the `SERPAPI_KEY` env var on every rerun *before* reading
the checkbox, silently reverting any click when a key was set — the
checkbox is now bound directly to session state via `key="use_mock"` and is
the sole source of truth after its initial default.

**Out of scope for this pass** (listed for transparency): a persistent
(non-in-memory) preference store, prioritized replay for feedback samples,
and richer NLP-derived features from listing text. The live demo means the
fixes above are no longer hypothetical — they're live for anyone who opens
the demo link.
