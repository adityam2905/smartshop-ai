# 🛍️ SmartShop: E-Commerce Deal Hunter & Scam Prevention RL Agent

A full-stack AI application that uses a **Deep Q-Network (DQN)** to evaluate
live product search results in real time — maximising deals, blocking scams,
and personalising recommendations based on your live feedback.

---

## Architecture Overview

```
data_generator.py   →   product_listings.csv
                               ↓
shopping_env.py     →   ShoppingEnv (Gymnasium)
                               ↓
train_agent.py      →   dqn_shopping_agent.zip
                               ↓
scraper.py          →   feature dicts from SerpAPI / mock
                               ↓
app.py              →   Streamlit UI + online learning loop
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

---

## File Reference

| File | Phase | Description |
|---|---|---|
| `data_generator.py` | 1 | Generates 5,000 synthetic product listings (legit + scam) |
| `shopping_env.py`   | 2 | Custom Gymnasium environment with reward shaping |
| `train_agent.py`    | 3 | DQN training, checkpointing, evaluation, online fine-tuning |
| `scraper.py`        | 4 | SerpAPI live fetch + feature engineering + trust scoring |
| `app.py`            | 5 | Streamlit UI, agent inference, Like/Dislike feedback loop |
| `requirements.txt`  | — | All Python dependencies |

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
