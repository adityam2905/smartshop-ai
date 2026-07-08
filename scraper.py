"""
Phase 4: Live Data Pipeline
Fetches real product listings via SerpAPI (Google Shopping),
engineers them into the 4-feature state vector, and provides
a realistic mock fallback when the API is unavailable.

Usage (standalone test):
    python scraper.py "Sony Headphones"
    python scraper.py "Nike Shoes" --mock       # force mock data
"""

import os
import re
import argparse
import hashlib
from urllib.parse import urlparse
from typing import Optional
import numpy as np

# ── Optional SerpAPI import ───────────────────────────────────────────────────
try:
    from serpapi import GoogleSearch        # pip install google-search-results
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")   # set this env var with your key

# ─────────────────────────────────────────────────────────────────────────────
# Domain Trust Score Database
# Maps known domains to a pre-computed trust score.
# Unknown domains fall through to the heuristic scorer below.
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_TRUST_DB: dict[str, float] = {
    # ── Tier 1: major retailers (0.90 – 1.00) ────────────────────────────────
    "amazon.com":         1.00,
    "walmart.com":        0.97,
    "bestbuy.com":        0.96,
    "target.com":         0.95,
    "costco.com":         0.95,
    "apple.com":          0.98,
    "samsung.com":        0.96,
    "bhphotovideo.com":   0.94,
    "adorama.com":        0.93,
    "newegg.com":         0.92,
    "ebay.com":           0.80,     # legit but third-party sellers vary

    # ── Tier 2: well-known specialty stores (0.75 – 0.89) ───────────────────
    "wayfair.com":        0.88,
    "homedepot.com":      0.90,
    "lowes.com":          0.89,
    "macys.com":          0.87,
    "nordstrom.com":      0.88,
    "zappos.com":         0.87,
    "chewy.com":          0.88,
    "etsy.com":           0.78,
    "overstock.com":      0.76,
    "rakuten.com":        0.75,

    # ── Tier 3: marketplace / discount (0.55 – 0.74) ────────────────────────
    "aliexpress.com":     0.60,
    "wish.com":           0.55,
    "temu.com":           0.58,

    # ── Known scam TLDs — scored near-zero ──────────────────────────────────
    # (handled via TLD heuristic below; entries here are just belt-and-suspenders)
}

# TLDs that are overwhelmingly used by scam sites
SCAM_TLDS = {".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".top",
             ".icu", ".ru", ".cc", ".biz", ".info", ".click", ".review"}

# TLDs associated with legitimate commerce
TRUSTED_TLDS = {".com", ".co.uk", ".co.jp", ".com.au", ".ca", ".de", ".fr"}


def compute_domain_trust(url: str) -> float:
    """
    Rule-based trust score for a given URL.

    Priority:
      1. Exact match in DOMAIN_TRUST_DB
      2. TLD heuristic (scam TLD → low score, trusted TLD → medium score)
      3. Keyword heuristic (suspicious words → lower score)
      4. Deterministic fallback using domain hash → [0.35, 0.70]
    """
    # --- extract bare domain ---
    domain = _extract_domain(url)
    if not domain:
        return 0.5

    # 1. Exact DB lookup
    for known_domain, score in DOMAIN_TRUST_DB.items():
        if domain.endswith(known_domain):
            return float(np.clip(score + _stable_jitter(domain, scale=0.01), 0.0, 1.0))

    # 2. TLD heuristic
    tld = _get_tld(domain)
    if tld in SCAM_TLDS:
        return round(float(np.clip(0.12 + _stable_jitter(domain, scale=0.08), 0.0, 0.25)), 4)
    if tld not in TRUSTED_TLDS:
        base_trust = 0.45
    else:
        base_trust = 0.60

    # 3. Suspicious keyword heuristic
    suspicious_keywords = [
        "deal", "cheap", "discount", "free", "sale", "win", "prize",
        "offer", "bargain", "flash", "ultra", "mega", "super", "best-price",
        "save", "hot", "limited", "exclusive", "buy-now",
    ]
    hit_count = sum(1 for kw in suspicious_keywords if kw in domain.lower())
    base_trust -= hit_count * 0.07        # each hit reduces trust

    # 4. Deterministic salt so same domain always gets same score
    jitter = _stable_jitter(domain, scale=0.05)

    trust = float(np.clip(base_trust + jitter, 0.0, 1.0))
    return round(trust, 4)


def _stable_jitter(text: str, scale: float) -> float:
    """Deterministic pseudo-random jitter in [-scale, scale]."""
    digest = hashlib.md5(text.encode()).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * 2.0 * scale


def _extract_domain(url: str) -> str:
    """Strip scheme/path/port, return bare domain."""
    parsed = urlparse(url.strip().lower())
    domain = parsed.netloc or parsed.path
    domain = re.sub(r"^www\.", "", domain)
    return domain.split(":")[0]


def _get_tld(domain: str) -> str:
    """Return the last two dot-segments as the TLD (handles .co.uk etc.)."""
    parts = domain.split(".")
    if len(parts) >= 3 and len(parts[-2]) <= 3:   # e.g. co.uk, com.au
        return "." + ".".join(parts[-2:])
    elif len(parts) >= 2:
        return "." + parts[-1]
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# User preference store (persisted in a simple dict; app.py can swap this)
# ─────────────────────────────────────────────────────────────────────────────

# Category → accumulated preference score (starts neutral at 0.5)
_USER_PREFS: dict[str, float] = {}

def update_user_preference(category: str, delta: float) -> None:
    """
    Called by app.py after a Like (+0.1) or Dislike (-0.1).
    Keeps scores in [0, 1].
    """
    current = _USER_PREFS.get(category, 0.5)
    _USER_PREFS[category] = float(np.clip(current + delta, 0.0, 1.0))

def get_user_preference(category: str) -> float:
    return _USER_PREFS.get(category, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering  ← the core function called by app.py
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(item: dict, market_avg_price: Optional[float] = None) -> dict:
    """
    Converts a raw product dict (from SerpAPI JSON or mock data)
    into the 4-feature state vector consumed by the DQN model.

    Accepted raw keys (SerpAPI Google Shopping format):
        title, price, extracted_price, link, source, category, rating, reviews

    Returns a dict:
        {
            "normalized_price":     float [0, 2],
            "discount_percentage":  float [0, 1],
            "site_trust_score":     float [0, 1],
            "user_preference_score":float [0, 1],
            # passthrough metadata (not fed to the model)
            "product_name": str,
            "price":        float,
            "market_avg":   float,
            "site_url":     str,
            "source":       str,
            "category":     str,
        }
    """
    # --- price ---
    raw_price = item.get("extracted_price") or item.get("price", 0)
    if isinstance(raw_price, str):
        raw_price = float(re.sub(r"[^\d.]", "", raw_price) or 0)
    price = float(raw_price)

    # --- market average (estimate if not provided) ---
    if market_avg_price is None:
        # Use the item's own price as a proxy then scale up slightly
        market_avg_price = price * 1.25 if price > 0 else 1.0

    market_avg_price = max(market_avg_price, 1.0)   # guard div-by-zero

    # --- normalized price ---
    normalized_price = float(np.clip(price / market_avg_price, 0.0, 2.0))

    # --- discount percentage ---
    # SerpAPI sometimes provides "old_price" or "was_price"
    old_price = item.get("old_price") or item.get("was_price")
    if old_price:
        if isinstance(old_price, str):
            old_price = float(re.sub(r"[^\d.]", "", old_price) or 0)
        old_price = float(old_price)
        if old_price > price > 0:
            discount_pct = float(np.clip((old_price - price) / old_price, 0.0, 1.0))
        else:
            discount_pct = float(np.clip(1.0 - normalized_price, 0.0, 1.0))
    else:
        # Infer from normalized price: if price < market avg → discount
        discount_pct = float(np.clip(1.0 - normalized_price, 0.0, 1.0))

    # --- site trust score ---
    url = item.get("link", "") or item.get("site_url", "")
    site_trust = compute_domain_trust(url)

    # --- user preference score ---
    category = item.get("category", "General")
    user_pref = get_user_preference(category)

    # --- metadata ---
    product_name = item.get("title") or item.get("product_name", "Unknown Product")
    source       = item.get("source", _extract_domain(url))

    return {
        # ── model inputs ─────────────────────────────────────────────────────
        "normalized_price":      round(normalized_price, 4),
        "discount_percentage":   round(discount_pct,    4),
        "site_trust_score":      round(site_trust,      4),
        "user_preference_score": round(user_pref,       4),
        # ── UI metadata ──────────────────────────────────────────────────────
        "product_name": product_name,
        "price":        price,
        "market_avg":   market_avg_price,
        "site_url":     url,
        "source":       source,
        "category":     category,
    }


def features_to_obs(features: dict) -> np.ndarray:
    """Converts feature dict → numpy array ready for model.predict()."""
    return np.array(
        [
            features["normalized_price"],
            features["discount_percentage"],
            features["site_trust_score"],
            features["user_preference_score"],
        ],
        dtype=np.float32,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock data fallback
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_PRODUCTS = {
    "Sony Headphones": [
        {"title": "Sony WH-1000XM5 Wireless Headphones",           "extracted_price": 278.00, "old_price": 349.99, "link": "https://www.amazon.com/dp/B09XS7JWHH",   "source": "Amazon",   "category": "Electronics"},
        {"title": "Sony WH-1000XM4 Noise Cancelling Headphones",   "extracted_price": 199.00, "old_price": 279.99, "link": "https://www.bestbuy.com/site/6408356.p",  "source": "Best Buy", "category": "Electronics"},
        {"title": "Sony WH-CH720N Wireless Headphones",            "extracted_price":  79.99, "old_price":  99.99, "link": "https://www.walmart.com/ip/564857219",    "source": "Walmart",  "category": "Electronics"},
        {"title": "Sony WH-1000XM5 — HUGE DISCOUNT 90% OFF!!",     "extracted_price":  34.99,                      "link": "https://ultra-deals99.net/sony-wh",      "source": "ultra-deals99.net", "category": "Electronics"},
        {"title": "Sony MDR-7506 Professional Monitor Headphones",  "extracted_price":  79.00,                      "link": "https://www.bhphotovideo.com/c/product", "source": "B&H Photo", "category": "Electronics"},
        {"title": "Sony Headphones CHEAP BUY NOW",                  "extracted_price":  19.99,                      "link": "https://cheapbuy-store.xyz/sony",        "source": "cheapbuy-store.xyz", "category": "Electronics"},
    ],
    "Nike Shoes": [
        {"title": "Nike Air Max 270 Men's Shoes",                   "extracted_price": 129.99,                      "link": "https://www.nike.com/t/air-max-270",     "source": "Nike",     "category": "Clothing"},
        {"title": "Nike Air Force 1 '07",                          "extracted_price":  90.00,                      "link": "https://www.zappos.com/p/nike-af1",      "source": "Zappos",   "category": "Clothing"},
        {"title": "Nike Revolution 6 Next Nature",                  "extracted_price":  55.00, "old_price":  70.00, "link": "https://www.amazon.com/dp/B09NXK6F4P",   "source": "Amazon",   "category": "Clothing"},
        {"title": "FAKE NIKE ULTRA SALE 95% OFF",                   "extracted_price":   4.99,                      "link": "https://bestprice-deals.tk/nike",        "source": "bestprice-deals.tk", "category": "Clothing"},
        {"title": "Nike Pegasus 40 Running Shoes",                  "extracted_price": 120.00, "old_price": 130.00, "link": "https://www.nordstrom.com/s/nike-peg",   "source": "Nordstrom", "category": "Clothing"},
    ],
    "default": [
        {"title": "Product A - Great Deal",                         "extracted_price":  49.99, "old_price":  79.99, "link": "https://www.amazon.com/dp/XXXXXXXXXX",   "source": "Amazon",   "category": "General"},
        {"title": "Product B - Standard Price",                     "extracted_price":  89.00,                      "link": "https://www.walmart.com/ip/123456789",   "source": "Walmart",  "category": "General"},
        {"title": "Product C - UNBELIEVABLE PRICE 80% OFF",         "extracted_price":   9.99,                      "link": "https://discount-mega.ru/deal",          "source": "discount-mega.ru", "category": "General"},
        {"title": "Product D - Verified Seller",                    "extracted_price": 134.00, "old_price": 150.00, "link": "https://www.bestbuy.com/site/XXXXXXX.p", "source": "Best Buy", "category": "General"},
    ],
}

def fetch_mock_results(query: str) -> list[dict]:
    """Returns realistic mock search results (mix of legit + scam)."""
    # Try to match the query to a mock category
    for keyword, products in _MOCK_PRODUCTS.items():
        if keyword.lower() in query.lower():
            return products
    return _MOCK_PRODUCTS["default"]


# ─────────────────────────────────────────────────────────────────────────────
# SerpAPI fetcher
# ─────────────────────────────────────────────────────────────────────────────

def fetch_serpapi_results(query: str, num_results: int = 10) -> list[dict]:
    """
    Fetches Google Shopping results from SerpAPI.
    Falls back to mock data on any error.
    """
    if not SERPAPI_AVAILABLE:
        print("[scraper] google-search-results not installed → using mock data.")
        return fetch_mock_results(query)

    if not SERPAPI_KEY:
        print("[scraper] SERPAPI_KEY env var not set → using mock data.")
        return fetch_mock_results(query)

    try:
        params = {
            "engine":    "google_shopping",
            "q":         query,
            "api_key":   SERPAPI_KEY,
            "num":       num_results,
            "gl":        "us",
            "hl":        "en",
        }
        results  = GoogleSearch(params).get_dict()
        shopping = results.get("shopping_results", [])

        if not shopping:
            print("[scraper] SerpAPI returned no results → using mock data.")
            return fetch_mock_results(query)

        print(f"[scraper] SerpAPI returned {len(shopping)} results for '{query}'.")
        return shopping[:num_results]

    except Exception as exc:
        print(f"[scraper] SerpAPI error: {exc} → using mock data.")
        return fetch_mock_results(query)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by app.py
# ─────────────────────────────────────────────────────────────────────────────

def search_products(query: str, use_mock: bool = False, num_results: int = 10) -> list[dict]:
    """
    Main entry point for app.py.

    1. Fetches raw results (SerpAPI or mock).
    2. Runs feature engineering on every item.
    3. Returns a list of feature dicts ready for model.predict().

    Args:
        query:       User search string (e.g. "Sony Headphones").
        use_mock:    Force mock data regardless of API availability.
        num_results: Max number of listings to return.

    Returns:
        List of feature dicts (see extract_features() for schema).
    """
    if use_mock:
        raw_results = fetch_mock_results(query)
    else:
        raw_results = fetch_serpapi_results(query, num_results)

    # Estimate a market average price from the batch
    prices = []
    for item in raw_results:
        p = item.get("extracted_price") or item.get("price", 0)
        if isinstance(p, str):
            p = float(re.sub(r"[^\d.]", "", p) or 0)
        if float(p) > 0:
            prices.append(float(p))

    # Use the median as a robust market average
    market_avg = float(np.median(prices)) if prices else None

    feature_list = []
    for item in raw_results:
        try:
            features = extract_features(item, market_avg_price=market_avg)
            feature_list.append(features)
        except Exception as exc:
            print(f"[scraper] Skipping item due to feature error: {exc}")

    return feature_list


# ─────────────────────────────────────────────────────────────────────────────
# CLI test harness
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the scraper pipeline.")
    parser.add_argument("query", nargs="?", default="Sony Headphones",
                        help="Product search query")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock data (no API call)")
    args = parser.parse_args()

    print(f"\nSearching for: '{args.query}' {'[MOCK]' if args.mock else '[LIVE]'}\n")
    products = search_products(args.query, use_mock=args.mock)

    print(f"{'#':<4} {'Product':<45} {'Price':>7} {'Disc%':>6} {'Trust':>6} {'UserPref':>9} {'URL'}")
    print("-" * 110)
    for i, feat in enumerate(products, 1):
        name  = feat["product_name"][:43]
        price = feat["price"]
        disc  = feat["discount_percentage"]
        trust = feat["site_trust_score"]
        pref  = feat["user_preference_score"]
        url   = feat["site_url"][:35]
        flag  = "🚨" if trust < 0.3 else "✅"
        print(f"{i:<4} {name:<45} ${price:>6.2f} {disc:>6.2%} {trust:>6.2f} {pref:>9.2f} {flag} {url}")

    print(f"\n{len(products)} products processed.\n")
    print("Tip: pass these feature dicts to ShoppingEnv.features_to_obs(f) "
          "then model.predict(obs) to get the agent's recommendation.")
