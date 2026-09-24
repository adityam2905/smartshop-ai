# 🛍️ SmartShop: E-Commerce Deal Hunter & Scam Prevention RL Agent

[![CI](https://github.com/adityam2905/smartshop-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/adityam2905/smartshop-ai/actions/workflows/ci.yml)

**🔗 [Live demo](https://smartshop-ai.streamlit.app/)** — Streamlit Community Cloud
*(free-tier apps sleep when idle — give it a minute to wake up)*

A Streamlit app where a **Deep Q-Network (DQN)** screens live Google Shopping
results. For each listing it weighs the seller's trustworthiness, the real
price against the market, and your taste, then recommends it or skips it:
real deals from trusted shops are shown, overpriced listings are skipped, and
scams — including polished shops whose only tell is a price that's too good to
be true — are blocked. It adapts to your 👍 / 👎 feedback during the session.

See [Limitations](#limitations) for where the RL framing is doing more work
than the problem needs.

---

## Results

Evaluated on two freshly generated datasets the model never saw (5,000
listings each). "Return vs. oracle" is the reward earned relative to a policy
that knows every listing's true label.

| Policy | Decision accuracy | Return vs. oracle | Scams recommended | Good deals missed |
|---|---|---|---|---|
| Hard rule: block trust < 0.3, recommend the rest | 72% | negative | 44% | 0% |
| Hard rule + recommend only below-market prices | 89% | negative | 44% | 0% |
| Linear contextual bandit | 58% | 61% | 0% | 38% |
| **DQN** | **98%** | **98%** | **0.1–0.2%** | **1%** |

The rules recommend almost half the scams (the polished ones pass the trust
check), which is why their return is negative despite decent accuracy. The
linear bandit avoids scams by being so cautious it misses 38% of good deals.
The DQN gets both right because it learns to *combine* trust with price: a
mid-trust shop is fine at 0.9× market and a scam at 0.25×.

On real results — a live Google Shopping India search for "iPhone 15" — it
recommended Flipkart at 0.73× market and a Cashify refurbished unit at 0.54×,
skipped imported listings above market, and skipped an unknown seller's
refurbished iPhone at ₹18,499 (0.23× market).

---

## Quick Start

```bash
pip install -r requirements.txt

python data_generator.py                           # 5,000 synthetic listings, 25% scam
python train_agent.py --timesteps 100000 --eval    # saves dqn_shopping_agent.zip (~1 min on CPU)
streamlit run app.py                               # uses demo data unless SERPAPI_KEY is set

# Optional
python scraper.py "Sony Headphones" --mock         # features, seller and trust per listing
python scraper.py "iPhone 15" --country in         # same, live (needs SERPAPI_KEY)
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
pip install -r requirements-test.txt && pytest -v    # lean: no torch
pip install -r requirements.txt && pytest -v         # everything (132 tests)
```

CI runs both: a fast job without torch, and a `test-full` job with CPU-only
torch that also runs the model-behaviour, online-learning and `app.py` tests.
`tests/test_model_artifact.py` fails if the committed model wasn't trained
with the settings in `agent_config.py`. Live search is tested against a faked
SerpAPI, so no key or network is needed.

---

## How It Works

| File | Role |
|---|---|
| `data_generator.py` | Synthetic listings (legit + scam) → `product_listings.csv` |
| `shopping_env.py` | Gymnasium environment: observations, actions, rewards |
| `agent_config.py` | DQN hyperparameters (torch-free, so CI can check the committed model) |
| `train_agent.py` | DQN training, evaluation, and online fine-tuning |
| `dqn_shopping_agent.zip` | Trained model, committed so the deploy works without training |
| `scraper.py` | SerpAPI / demo fetch, seller resolution, trust scoring, market prices |
| `app.py` | Streamlit UI, inference, Like/Dislike loop, quota protection |
| `evaluation.py` | Shared scoring harness for the DQN, bandit and rules |
| `bandit_baseline.py`, `supervised_baseline.py` | Baselines (see below) |

### RL design

**Observation:**

| Feature | Meaning |
|---|---|
| `normalized_price` [0, 2] | real price ÷ market price — values the deal, and flags lures |
| `discount_percentage` [0, 1] | the discount the seller *claims* (can be inflated) |
| `site_trust_score` [0, 1] | how trustworthy the seller is |
| `user_preference_score` [0, 1] | how much you like this category |

**Actions:** `0` Skip · `1` Recommend

| Situation | Reward |
|---|---|
| Recommend + Scam | **-100** |
| Recommend + Legit | `20 × (1 − normalized_price) + 10 × (preference − 0.5)` |
| Skip + Scam | **+10** |
| Skip + Legit | 0 |

Recommending a legit listing pays off only if it's actually below market or
in a category you like — an overpriced listing from a trusted retailer costs
reward, so the agent has to judge deal quality, not just avoid scams.

**Hyperparameters:** 128→128 MLP · LR 5e-4 · γ 0.97 · batch 64 · replay
buffer 100k · ε 1.0 → 0.02 over 20% of training · target update every 1,000 steps.

**Training data** is built so no single feature decides the answer:

- **Legit sellers** range from big retailers (trust ~0.9) through mid-tier
  shops to small independents (trust 0.40–0.70), and list clearance deals,
  normal prices and overpriced items.
- **Scams** are either obvious (trust < 0.28) or **polished shops** with
  middling trust (0.30–0.60) — the same range as small legit shops — whose
  price is far below market.
- The **claimed discount** overlaps between classes, so it can't be used as
  a shortcut.

### Seller trust

Live Google Shopping results link to a Google page rather than the shop, so
`resolve_seller_domain()` unwraps Google redirects and otherwise maps the
seller name ("Flipkart", "Amazon.in", "EMI Snapmint", "eBay - seller123") to a
domain. Unknown sellers get a neutral 0.5. `compute_domain_trust()` then
scores the domain:

1. **Known retailers** — major US/UK retailers, Amazon's regional sites, and
   Indian retailers (Flipkart, Croma, Reliance Digital, Vijay Sales, Cashify,
   …). Matched on the exact registrable domain, so `smile.amazon.com` counts
   but `cheap-amazon.com` doesn't.
2. **Look-alikes** — a domain using a known brand name without being that
   brand (`cheap-amazon.com`, `amaz0n.com`, `flipkartsale.shop`) scores ~0.10.
3. **Everything else** — heuristics on TLD (`.xyz`, `.tk`, … score low) and
   scammy keywords (`deal`, `cheap`, `mega`, …).

`app.py` hard-blocks trust < 0.3 before the DQN is consulted; the DQN handles
everything above that. Listing titles, seller names and links come from
third-party shops, so they're HTML-escaped and only `http(s)` links render.

### Market price

`normalized_price` needs a market price to compare against
(`estimate_market_prices()`): a trusted seller's own list price when it gives
one, otherwise the median price among **trusted** sellers in the results, and
the median of everything only as a last resort. Scam lures are excluded on
purpose — they'd drag the median down and make real retailers look overpriced.

### Online learning

Every 3 Likes/Dislikes, the session's model runs 50 gradient steps on two
objectives:

- **Feedback:** move `Q(s, Recommend)` to the pretrained value ± 20 for each
  rated listing. This is bounded, so repeated Dislikes can't push past it.
- **Anchor:** keep `Q` equal to the pretrained model's on states sampled
  across the whole observation space, so unrated listings don't drift.

It trains on all of the session's feedback. Each browser session fine-tunes
its own copy; the shared pretrained model is only read. Likes/Dislikes also
shift the per-category preference score, which the model uses as a feature.

---

## Baselines

**Contextual bandit** (`bandit_baseline.py`). Each reward depends only on
the current listing, and the next listing doesn't depend on the action, so
this is a contextual bandit, not a true MDP — γ, the replay buffer and the
target network have no temporal credit to assign. The from-scratch linear
bandit can't represent "mid trust is fine unless the price is implausible"
with a single linear score per action, so it settles for skipping anything
borderline. The DQN's edge is the MLP's capacity, not RL; an MLP bandit is the
fair next comparison.

**Supervised classifiers** (`supervised_baseline.py`), on the plain "is this
a scam?" label:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Hard rule (trust < 0.3) | 1.000 | 0.574 | 0.729 |
| Logistic Regression | 0.984 | 0.968 | 0.976 |
| Random Forest | 0.997 | 1.000 | 0.998 |

The rule never flags a legit seller but misses the polished scams. Random
Forest leans on trust (0.62 importance) and price (0.30) — the same
combination the DQN has to learn.

---

## Deployment

Hosted on [Streamlit Community Cloud](https://share.streamlit.io) from `main`
with main file `app.py`. No secrets are needed; add `SERPAPI_KEY` (and
optionally `SERPAPI_COUNTRY = "in"`) under **Settings → Secrets** for live
results.

- **SerpAPI quota:** raw live results are cached for 6 hours across all
  visitors (failures aren't cached), and each session gets 15 distinct live
  searches before falling back to demo data with a notice.
- **Updating the model:** retrain, commit `dqn_shopping_agent.zip`, push,
  then **Reboot app** from the dashboard. The model is held in
  `st.cache_resource`, so a running app keeps the old one until it restarts.
- **Install size:** `requirements.txt` uses plain `stable-baselines3`, not
  `[extra]`, which pulls in opencv/pygame/Atari and roughly doubles install
  time. See the file for a CPU-only torch tip.

---

## Limitations

1. **It's a contextual bandit framed as an MDP** — see [Baselines](#baselines).
2. **The training data is still synthetic.** It's designed to be realistic —
   overlapping trust, overlapping claimed discounts, overpriced legit
   listings — but the model has only been checked on a handful of real
   searches, not a labelled set of real listings. The 98% is on data drawn
   from the same generator.
3. **"Market price" is estimated from one search's results.** A search that
   mixes very different products (a phone and its case) gives a poor
   reference, and a search with few trusted sellers falls back to the median
   of everything.
4. **Trust is a hand-built heuristic.** An unknown seller gets 0.5 however
   reputable it is, and a scam on a clean-looking domain needs an implausible
   price to be caught.
5. **Preferences are in-memory** and reset when the session ends.
6. **Live search shares one quota** of 100 searches/month across all visitors;
   caching and the per-session limit slow that down but don't remove it.

### Fixed bugs

- **Online learning never worked:** it trained nothing for 126 clicks (empty
  replay buffer after loading), crashed at click 129 (loaded SB3 models have
  no logger), and ~128 Dislikes made the model skip everything. Replaced with
  the approach [above](#online-learning).
- **Stale model:** the deployed model was trained with different LR, γ, batch
  size and network size than documented. Retrained; now checked in CI.
- **The agent didn't judge deals:** the old reward paid for recommending any
  legit listing and penalised skipping it, so the policy was "recommend
  everything that isn't a scam", matching the trust rule; the preference
  feature was ignored.
- **Discount shortcut:** non-overlapping discount ranges taught the agent
  "small discount = safe", so it skipped real 75%+ deals and recommended
  low-trust scams at 5–20% off.
- **Look-alike domains:** `endswith("amazon.com")` gave `cheap-amazon.com`
  full trust, while `amazon.in` and Flipkart were unknown.
- **Live trust was meaningless:** it was scored on the result's link, which
  for Google Shopping is a Google page, so every live seller got the same score.
- **Skewed market price:** the plain median of all results was dragged down by
  scam lures, making genuine retailers look overpriced.
- **Quota drain:** search results were cached for 5 minutes and keyed on the
  user's preferences, so every Like/Dislike cost another SerpAPI search.
- **Silent fallback to demo data:** failed live searches (including SerpAPI's
  "out of searches" response) quietly showed mock listings.
- **Unescaped listing HTML:** third-party titles and links were inserted into
  the page as raw HTML.
- **Shared state across users:** one visitor's feedback retrained the model
  everyone used, and preferences leaked between sessions.
- **UI:** quick-search chips crashed the app, and the "Force mock data"
  checkbox reset itself on every rerun.
- **Training crash:** `train_agent.py` required `rich` for its progress bar,
  which isn't installed without `[extra]`.
