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
| Hard rule: block trust < 0.3, recommend the rest | 59% | negative | 44% | 0% |
| Hard rule + recommend only at ≤ 0.90× market | 89% | negative | 44% | 0% |
| Linear contextual bandit | 63% | 62% | 0% | 31% |
| **DQN** | **95%** | **98%** | **0.1%** | **1%** |

The rules recommend almost half the scams (the polished ones pass the trust
check), which is why their return is negative despite decent accuracy. The
linear bandit avoids scams by being so cautious it misses 31% of good deals.
The DQN gets both right because it learns to *combine* trust with price: a
mid-trust shop is fine at 0.8× market and a scam at 0.25×.

### On real listings

`evaluate_real.py` runs the deployed pipeline on **101 hand-labelled real
Google Shopping India listings** from 14 searches (`real_data/`, collected
September 2026). Each listing was labelled — before the model was run on it —
for whether the seller is trustworthy and whether the price beats the usual
price for that exact variant. Labels were drafted with AI assistance (web
checks of each unfamiliar seller, reasons in the `notes` column) and reviewed
by the author.

Only 19 of the 101 listings are good deals, so **accuracy is misleading
here — recommending nothing scores 81%**. Precision, recall and F1 are the
numbers that matter.

**Deployed model** (recommends only at ≤ 0.90× market, i.e. ≥ 10% off):

| Policy | Accuracy [95% CI] | Precision | Recall | F1 | Bad sellers recommended |
|---|---|---|---|---|---|
| Recommend nothing | 81% | — | 0% | 0% | 0 / 5 |
| Recommend everything / trust rule | 19% [12–28] | 19% | 100% | 32% | 5 / 5 |
| Trust rule + ≤ 0.90× market | 74% [65–82] | 38% | 58% | 46% | 3 / 5 |
| Trust rule + ≤ 0.90× market, with reference prices | 78% [69–85] | 45% | 74% | 56% | 3 / 5 |
| **DQN, as deployed** (in-search price estimates) | **77% [68–84]** | **41%** | **47%** | **44%** | **0 / 5** |
| DQN + reference prices (60 of 101 listings) | 70% [61–78] | 28% | 37% | 32% | 0 / 5 |

**Discount threshold sweep** — one DQN retrained per threshold, in-search
price estimates:

| Recommend at ≥ … off | DQN precision | DQN recall | DQN F1 | Price rule F1 | Bad sellers (DQN / rule) |
|---|---|---|---|---|---|
| 5% | 25% | 53% | 34% | 47% | 1 / 4 |
| 10% (sweep run) | 41% | 58% | 48% | 46% | 1 / 3 |
| **10% (deployed, seed 0)** | **41%** | **47%** | **44%** | 46% | **0** / 3 |
| 15% | 40% | 42% | 41% | 50% | 1 / 3 |
| 20% | 58% | 37% | 45% | 48% | 0 / 3 |

- **10% vs 20% is a trade-off, not a win.** A higher bar makes the DQN more
  precise and less complete; 10% was chosen to surface more deals (47% of
  them vs 37%) at the cost of precision (41% vs 58%).
- **Training noise is as large as the threshold effect.** The two 10% rows
  are the same threshold and data, different training runs, and differ by
  4 F1 points — about the spread across thresholds. Training is now seeded,
  so `train_agent.py` reproduces the deployed model exactly.
- **The price rule finds deals about as well but recommends most of the bad
  sellers** — the DQN's value is its caution about sellers.

**How it got here** (each step measured on the same labels):

| Change | DQN precision | DQN recall | DQN F1 |
|---|---|---|---|
| First real-data run | 30% | 63% | 41% |
| Compare each listing only with the same product | 25% | 53% | 34% |
| Recommend only at ≤ 0.80× market (retrained) | 58% | 37% | 45% |
| **Recommend only at ≤ 0.90× market** (retrained, deployed) | **41%** | **47%** | **44%** |

- **The big win was the reward.** The old reward broke even at the normal
  price, so the agent recommended trusted sellers at no discount (29 of 40
  recommendations were no real deal). Requiring a real discount fixed that.
- **Reference prices** (the median price of the same product at other
  stores) help the price rule (F1 46% → 56%) but hurt the DQN (44% → 32%) —
  probably because prices measured against other stores are distributed
  differently from the synthetic data it was trained on (untested).
- **Most remaining misses are unknown sellers with big discounts:** 8 of the
  10 missed deals. The agent recommends unknown (trust 0.5) sellers at 10–30%
  off but not deeper — a deep discount from an unknown seller is exactly the
  polished-scam pattern it was trained to avoid (e.g. StockX at 0.45×).
- **Scams:** 0 of 5 untrustworthy sellers recommended and 0 legit sellers
  blocked, with or without reference prices.

These 101 listings were used to diagnose and tune the steps above, so they
are a development set and these numbers are optimistic; a fresh set of
searches is the fair test.

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
python evaluate_real.py                            # real labelled listings (offline)
python collect_real_listings.py                    # fetch more (spends SerpAPI searches)
```

**Live vs. demo data.** For live Google Shopping results, get a key at
[serpapi.com](https://serpapi.com) (250 free searches/month) and
`export SERPAPI_KEY="..."`. Pick the country (US, India, UK, Canada,
Australia) in the sidebar; prices show in that country's currency, and
`SERPAPI_COUNTRY=in` sets the default. Without a key, the app uses built-in
demo listings for headphones, shoes, iPhones, MacBooks and gaming chairs, each
a mix of real retailers and scam sites. If a live search fails (no key, quota
used up, network error), the app says why and shows demo data instead.

### Tests

```bash
pip install -r requirements-test.txt && pytest -v    # lean: no torch
pip install -r requirements.txt && pytest -v         # everything (188 tests)
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
| `collect_real_listings.py`, `evaluate_real.py`, `real_data/` | Real-listing collection, labels, and evaluation |
| `reference_prices.py` | Usual price of a product across stores (SerpAPI product pages) |

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
| Recommend + Legit, price ≤ 0.90× market | `20 × (0.90 − normalized_price) + 10 × (preference − 0.5)` |
| Recommend + Legit, price > 0.90× market | negative (≤ −1), whatever the preference |
| Skip + Scam | **+10** |
| Skip + Legit | 0 |

A listing is only worth recommending at **10% or more below the market
price** (`ShoppingEnv.DEAL_THRESHOLD`, chosen from a 5–20% sweep on real
listings); preference can make the agent pickier about a deal but never
lowers that bar. (The reward used to break even at 1.0×, so the agent
recommended trusted sellers at their normal price — 29 of its 40
recommendations on real listings were no real deal.)

**Hyperparameters:** 128→128 MLP · LR 5e-4 · γ 0.97 · batch 64 · replay
buffer 100k · ε 1.0 → 0.02 over 20% of training · target update every 1,000 steps ·
seed 0 (training is reproducible).

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
(`estimate_market_references()`), in order of preference:

0. **A reference price** (`reference_prices.py`): the median price of this
   exact product across the *other* stores on Google's product page (SerpAPI
   `google_immersive_product`), excluding hard-blocked stores. The only source
   that actually knows the usual price — but it costs one SerpAPI search per
   product, so the app uses it only when `SERPAPI_REFERENCE_PRICES=1`, for up
   to 5 listings per search, cached for 24 hours.
1. A trusted seller's own list price ("was ₹34,990").
2. The median price of **the same product** from trusted sellers, then from
   any seller that isn't hard-blocked. Two listings count as the same product
   only if their condition (new vs refurbished/used), tier words (Pro, Max,
   Plus, Ultra, Elite…), variant attributes (128GB, 30 ml, 45 mm), model codes
   (RB3025, 15-fb3383AX, LEGO 42172) and model numbers ("Airdopes 141",
   "Series 11") agree.
3. Nothing comparable → no price signal (1.0×), unless the price is below 40%
   of the search's trusted median — suspicious whatever the product.

Scam lures never set the reference for anyone else. Each listing records
which basis it used (`market_basis`), and `evaluate_real.py` prints it next to
every mistake.

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
- **Reference prices** (optional, off by default): add
  `SERPAPI_REFERENCE_PRICES = "1"` to Secrets. Each new search then costs up
  to 6 SerpAPI searches instead of 1 — and with the current model they
  *lower* its F1 on real listings (see [On real listings](#on-real-listings)),
  so leave them off unless the model is retrained for them.
- **Updating the model:** retrain, commit `dqn_shopping_agent.zip`, push,
  then **Reboot app** from the dashboard. The model is held in
  `st.cache_resource`, so a running app keeps the old one until it restarts.
- **Install size:** `requirements.txt` uses plain `stable-baselines3`, not
  `[extra]`, which pulls in opencv/pygame/Atari and roughly doubles install
  time. See the file for a CPU-only torch tip.

---

## Limitations

1. **It's a contextual bandit framed as an MDP** — see [Baselines](#baselines).
2. **Strong on synthetic data, modest on real data.** 95% accuracy on
   synthetic data; on 101 real labelled listings, F1 44% for finding deals
   (41% precision, 47% recall) — no better than a simple price rule, though
   far safer about sellers (see [On real listings](#on-real-listings)). The
   real set is small, has only 5 bad sellers, and was used for development;
   differences of a few points are within training-run noise.
3. **It misses deep discounts from unknown sellers.** It recommends only at
   ≥ 10% off, and treats a big discount from a seller the trust scorer
   doesn't know as a likely scam — 8 of its 10 missed real deals. About half
   of real good deals get shown, and 4 in 10 recommendations are real deals.
4. **Trust is a hand-built heuristic.** Unknown sellers — including many
   genuine brand stores — get 0.5, and unknown `.in`/`.com` shops can score
   above 0.6 whatever their reputation. A proper seller-reputation source
   would help both recall and scam detection.
5. **Reference prices cost quota.** One SerpAPI search per product, so they're
   off by default in the app and cover ~60% of listings when on (the rest
   aren't sold by any other store Google lists).
6. **Preferences are in-memory** and reset when the session ends.
7. **Live search shares one quota** of 250 searches/month across all visitors;
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
- **Misread list prices:** for "32% off₹34,990", SerpAPI's numeric old price
  is 32 (the percentage), so real list prices were discarded. Found in the
  real-data evaluation.
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
