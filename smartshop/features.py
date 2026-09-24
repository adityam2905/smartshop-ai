"""
Turning a raw listing into the model's four inputs, plus the per-session
category preferences that feed one of them.
"""

from typing import Optional

import numpy as np

from .pricing import parse_old_price, parse_price
from .trust import compute_domain_trust, resolve_seller_domain

NEUTRAL_PREFERENCE = 0.5

# ── Category preferences ─────────────────────────────────────────────────────
# Always passed in explicitly (app.py keeps one dict per browser session in
# st.session_state). Never keep them in a module-level dict: the app process is
# shared by every visitor, so one person's likes would leak into everyone's
# results — a bug this project once had.

def update_user_preference(category: str, delta: float, prefs: dict) -> dict:
    """Shift a category's preference after a Like (+delta) / Dislike (−delta), kept in [0, 1]."""
    prefs[category] = float(np.clip(prefs.get(category, NEUTRAL_PREFERENCE) + delta, 0.0, 1.0))
    return prefs


def get_user_preference(category: str, prefs: Optional[dict] = None) -> float:
    return (prefs or {}).get(category, NEUTRAL_PREFERENCE)


# ── Features ─────────────────────────────────────────────────────────────────

def extract_features(
    item: dict,
    market_avg_price: Optional[float] = None,
    user_prefs: Optional[dict] = None,
) -> dict:
    """
    A raw listing (SerpAPI result or demo listing) → the model's four inputs
    plus display fields.

    `market_avg_price` is the usual price to compare against (see
    pricing.estimate_market_references); `user_prefs` is the caller's
    per-session preference dict.
    """
    price = parse_price(item.get("extracted_price") or item.get("price", 0))
    if market_avg_price is None:
        market_avg_price = price * 1.25 if price > 0 else 1.0
    market_avg_price = max(market_avg_price, 1.0)          # avoid dividing by zero
    normalized_price = float(np.clip(price / market_avg_price, 0.0, 2.0))

    # The discount the seller claims: from their "was" price if they give one,
    # otherwise implied by the price vs the market
    old_price = parse_old_price(item)
    if old_price > price > 0:
        discount_pct = float(np.clip((old_price - price) / old_price, 0.0, 1.0))
    else:
        discount_pct = float(np.clip(1.0 - normalized_price, 0.0, 1.0))

    # Trust is scored on the seller, not on the (usually Google) link
    seller_domain = resolve_seller_domain(item)
    category = item.get("category", "General")

    return {
        # model inputs
        "normalized_price":      round(normalized_price, 4),
        "discount_percentage":   round(discount_pct, 4),
        "site_trust_score":      round(compute_domain_trust(seller_domain), 4),
        "user_preference_score": round(get_user_preference(category, user_prefs), 4),
        # display fields
        "product_name":  item.get("title") or item.get("product_name", "Unknown Product"),
        "price":         price,
        "market_avg":    market_avg_price,
        "site_url":      item.get("link") or item.get("product_link") or item.get("site_url", ""),
        "seller_domain": seller_domain,
        "source":        item.get("source") or seller_domain or "Unknown seller",
        "category":      category,
    }


def features_to_obs(features: dict) -> np.ndarray:
    """Feature dict → the model's observation: [price ratio, claimed discount, trust, preference]."""
    return np.array(
        [
            np.clip(features.get("normalized_price", 1.0), 0.0, 2.0),
            np.clip(features.get("discount_percentage", 0.0), 0.0, 1.0),
            np.clip(features.get("site_trust_score", 0.5), 0.0, 1.0),
            np.clip(features.get("user_preference_score", NEUTRAL_PREFERENCE), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
