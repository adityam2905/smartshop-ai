# 🛍️ SmartShop: AI Deal Hunter & Scam Blocker

[![CI](https://github.com/adityam2905/smartshop-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/adityam2905/smartshop-ai/actions/workflows/ci.yml)

**🔗 [Live demo](https://smartshop-ai.streamlit.app/)** *(free hosting — the app may take a minute to wake up)*

Search for a product and SmartShop checks every Google Shopping result. It
shows you real deals from trustworthy shops, hides overpriced listings, and
blocks scam sites. Tell it 👍 or 👎 and it adjusts to your taste.

The decisions are made by a **Deep Q-Network (DQN)**, a reinforcement
learning model, trained on synthetic listings and tested on real ones.

---

## How it works

For each search result, the app works out four things:

1. **How trustworthy the seller is** (0–1), from a list of known retailers and
   warning signs such as scam domains (`.xyz`, `.tk`) or fake brand names
   (`cheap-amazon.com`).
2. **How the price compares to the usual price** for that exact product.
3. **The discount the seller claims** (which can be exaggerated).
4. **How much you like that category** (updated by your 👍 / 👎).

The model then decides **Recommend** or **Skip**. It recommends a listing only
if the seller looks trustworthy and the price is **at least 10% below the usual
price**. Obvious scam sites (trust below 0.3) are blocked before the model
even looks at them.

---

## Results

### On synthetic data

Tested on 10,000 generated listings the model never saw:

| Approach | Correct decisions | Scams recommended | Good deals missed |
|---|---|---|---|
| Simple rule: block low-trust sites, recommend the rest | 59% | 44% | 0% |
| Simple rule + only recommend at ≥ 10% off | 89% | 44% | 0% |
| Simple learning model (linear bandit) | 63% | 0% | 31% |
| **DQN** | **95%** | **0.1%** | **1%** |

Simple rules let through "polished" scam shops, which look trustworthy and
are only given away by a price that's too good to be true. The DQN learns to
weigh trust and price together.

### On real listings

To check it on real data, I labelled **101 real Google Shopping India
listings** from 14 searches (`real_data/`). For each one I recorded whether
the seller is trustworthy and whether the price beats the usual price. The
labels were drafted with AI help (each unfamiliar seller was checked online,
with the reason noted) and then reviewed by me.

Only 19 of the 101 are good deals, so "percent correct" is misleading here:
recommending nothing at all would score 81%. These numbers matter more:

- **Precision:** of the listings it recommends, how many are real good deals.
- **Recall:** of the real good deals, how many it recommends.
- **F1:** a single score that balances the two.

| Approach | Precision | Recall | F1 | Bad sellers recommended |
|---|---|---|---|---|
| Recommend everything from non-scam sites | 19% | 100% | 32% | 5 of 5 |
| Simple rule: ≥ 10% off | 38% | 58% | 46% | 3 of 5 |
| **DQN (deployed)** | **41%** | **47%** | **44%** | **0 of 5** |

**In short:** the DQN finds deals about as well as a simple price rule, but it
is much safer: it recommended none of the 5 untrustworthy sellers, and it
never blocked a real shop.

**Honest caveats:**

- The real test set is small (19 good deals, 5 bad sellers), so small
  differences aren't meaningful.
- I used these listings to find and fix problems, so the numbers are
  probably a bit optimistic. New, unseen searches would be the fair test.
- Retraining the model gives slightly different results each time; F1 can
  move by about 4 points.

<details>
<summary><b>How the real-data results changed as I improved the model</b></summary>

| Change | Precision | Recall | F1 |
|---|---|---|---|
| First test on real data | 30% | 63% | 41% |
| Compare each listing only with the same product | 25% | 53% | 34% |
| Only recommend at ≥ 20% off | 58% | 37% | 45% |
| **Only recommend at ≥ 10% off (deployed)** | **41%** | **47%** | **44%** |

The biggest fix was requiring a real discount. Before that, the model
recommended trusted shops even at their normal price.

I also tested different minimum discounts (a separately trained model for each):

| Minimum discount | Precision | Recall | F1 |
|---|---|---|---|
| 5% | 25% | 53% | 34% |
| 10% | 41% | 47–58% | 44–48% |
| 15% | 40% | 42% | 41% |
| 20% | 58% | 37% | 45% |

A higher minimum gives fewer but more reliable recommendations. I chose 10%
to show more deals.

</details>

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py              # the trained model is included, so this just works
```

The app works without any setup using built-in demo listings. For **live
Google Shopping results**, get a free key at [serpapi.com](https://serpapi.com)
(250 searches/month) and set it:

```bash
export SERPAPI_KEY="your_key"
export SERPAPI_COUNTRY="in"      # optional: default country (us, in, uk, ca, au)
```

Other useful commands (run from the project folder):

```bash
python -m smartshop.train --eval                   # retrain the model (~2 min on CPU)
python -m smartshop.search "iPhone 15" --country in # show each result's seller, trust and price
python -m experiments.evaluate_real                # score the model on the labelled real listings
python -m experiments.bandit_baseline --compare-dqn # compare with a simpler learning model
pytest                                             # run the tests (188)
```

---

## Project structure

```
app.py                  The Streamlit app
assets/style.css        The app's styling
smartshop/              The core code
  search.py               Fetches results and runs the full pipeline for each search
  trust.py                How trustworthy a seller is
  pricing.py              The usual price each listing is compared against
  features.py             Turns a listing into the model's inputs
  mock_data.py            Demo listings used without a SerpAPI key
  reference_prices.py     Optional: a product's price at other stores
  environment.py          What the model sees and how it's rewarded in training
  data_generator.py       Creates the synthetic training listings
  train.py                Trains the model
  agent.py                Loads the model and updates it from your 👍 / 👎
  evaluation.py           Scores any policy on the synthetic listings
  config.py               Settings: file paths, thresholds, training settings
models/                 The trained model
experiments/            Baselines and the real-data evaluation scripts
real_data/              101 labelled real listings (see real_data/LABELLING.md)
tests/                  Automated tests, run on every push
```

---

## More detail

<details>
<summary><b>How the model is rewarded</b></summary>

| What happened | Reward |
|---|---|
| Recommended a scam | −100 |
| Skipped a scam | +10 |
| Recommended a real listing at ≥ 10% off | positive: bigger for bigger discounts and categories you like |
| Recommended a real listing at < 10% off | negative |
| Skipped a real listing | 0 |

Your taste can make the model pickier, but it can never make it recommend
something less than 10% off.

</details>

<details>
<summary><b>How the training data is built</b></summary>

The synthetic data is designed so that no single clue gives the answer away:

- **Real shops** range from big retailers (high trust) to small independent
  shops (medium trust), and sell at clearance, normal and inflated prices.
- **Scams** are either obvious (very low trust) or **polished**: medium trust,
  just like small real shops, and only given away by a suspiciously low price.
- **Claimed discounts** overlap between real shops and scams, so the model
  can't rely on them.

</details>

<details>
<summary><b>How seller trust is worked out</b></summary>

- Google Shopping links go to a Google page, not the shop, so the seller is
  identified from the result's store name (e.g. "Flipkart", "Amazon.in").
- **Known retailers** (Amazon, Flipkart, Croma, Reliance Digital, Walmart, and
  others) get high trust.
- **Fake brand names** (`cheap-amazon.com`, `amaz0n.com`) get very low trust.
- **Other sites** are scored on warning signs such as scam domains and words
  like "cheap" or "mega". Unknown shops get a neutral 0.5.

</details>

<details>
<summary><b>How the usual price is worked out</b></summary>

In order of preference:

1. *(Optional)* The price of the same product at other stores, from Google's
   product page. Off by default, because each lookup costs a SerpAPI search
   and it didn't improve the model's results.
2. A trusted seller's own "was" price.
3. The typical price of **the same product** among the other results.
   Listings only count as the same if model, storage, size and
   new/refurbished all match (e.g. iPhone 15 ≠ iPhone 15 Plus, 18 ml ≠ 30 ml).
4. If nothing matches, no price signal is used, unless the price is suspiciously
   far below everything else in the search.

</details>

<details>
<summary><b>How 👍 / 👎 feedback works</b></summary>

After every 3 ratings, your copy of the model is briefly retrained. Liked
listings become more likely to be recommended and disliked ones less likely,
while everything else stays as it was. Each visitor gets their own copy, so
one person's ratings never affect anyone else.

</details>

<details>
<summary><b>Is reinforcement learning really needed?</b></summary>

Not strictly. Each decision stands alone (skipping one listing doesn't change
the next), so this is a simpler kind of problem called a *contextual bandit*.
The DQN beats the simpler linear model here because its neural network can
learn more complex patterns, not because of anything specific to
reinforcement learning. I kept the DQN, but it's worth knowing.

</details>

---

## Deployment

The live demo runs on [Streamlit Community Cloud](https://share.streamlit.io)
from the `main` branch. To enable live results, add `SERPAPI_KEY` (and
optionally `SERPAPI_COUNTRY = "in"`) under **Settings → Secrets**.

- **Saving searches:** results are cached for 6 hours and each visitor gets
  15 live searches, so the free 250 searches/month last longer.
- **Updating the model:** retrain, commit `models/dqn_shopping_agent.zip`, push, then
  click **Reboot app** in the Streamlit dashboard. The running app keeps the
  old model until it restarts.

---

## Limitations

- **Modest on real data:** about 4 in 10 recommendations are real deals, and
  it finds about half of them.
- **Misses big discounts from unknown shops:** these look like scams to it
  (8 of its 10 missed deals).
- **Seller trust is hand-built:** unknown shops, including some genuine ones,
  get a neutral score.
- **Small real test set**, which was also used to improve the model.
- **Shared search limit:** all demo visitors share 250 searches a month.
- **Preferences reset** when you close the page.

---

## Bugs found and fixed

- 👍 / 👎 learning never actually worked (it trained nothing, then crashed).
- The deployed model didn't match the documented training settings.
- The model recommended shops at full price because the reward didn't require a discount.
- The model learned "small discount = safe" and skipped genuine big deals.
- `cheap-amazon.com` got Amazon's full trust; `amazon.in` and Flipkart weren't recognised.
- Every live seller got the same trust score, because the trust check was looking at Google's link.
- Scam prices dragged down the "usual price", making real shops look overpriced.
- "32% off ₹34,990" was read as an old price of ₹32.
- Every 👍 / 👎 used up an extra SerpAPI search.
- Failed live searches silently showed demo data.
- Shop names and links were shown in the page without escaping.
- One visitor's feedback changed the model for everyone.
- Several UI crashes, and a training script that crashed on startup.
