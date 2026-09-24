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
from urllib.parse import parse_qs, urlparse
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

# Google Shopping country (SerpAPI `gl` code) → (display name, currency symbol).
# Live prices come back in the local currency, so the symbol follows the country.
COUNTRIES: dict[str, tuple[str, str]] = {
    "us": ("United States",  "$"),
    "in": ("India",          "₹"),
    "uk": ("United Kingdom", "£"),
    "ca": ("Canada",         "CA$"),
    "au": ("Australia",      "A$"),
}
DEFAULT_COUNTRY = os.environ.get("SERPAPI_COUNTRY", "us").lower()
if DEFAULT_COUNTRY not in COUNTRIES:
    DEFAULT_COUNTRY = "us"
MOCK_CURRENCY = "$"                                 # mock listings are priced in USD

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
    "nike.com":           0.97,
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

    # ── Regional storefronts of Tier 1 retailers ────────────────────────────
    "amazon.in":          0.99,
    "amazon.co.uk":       0.99,
    "amazon.ca":          0.99,
    "amazon.de":          0.99,
    "amazon.fr":          0.99,
    "amazon.co.jp":       0.99,
    "amazon.com.au":      0.99,
    "ebay.co.uk":         0.80,

    # ── India (0.75 – 0.97) ──────────────────────────────────────────────────
    "flipkart.com":       0.97,
    "myntra.com":         0.93,
    "croma.com":          0.93,
    "reliancedigital.in": 0.92,
    "tatacliq.com":       0.90,
    "jiomart.com":        0.90,
    "nykaa.com":          0.90,
    "ajio.com":           0.89,
    "vijaysales.com":     0.90,
    "poorvika.com":       0.88,
    "sangeethamobiles.com": 0.88,
    "cashify.in":         0.82,     # refurbished marketplace
    "snapmint.com":       0.80,     # EMI / buy-now-pay-later storefront
    "desertcart.in":      0.75,     # cross-border importers — legit, but
    "ubuy.co.in":         0.75,     # third-party imported stock
    "snapdeal.com":       0.75,
    "meesho.com":         0.75,

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
TRUSTED_TLDS = {".com", ".co.uk", ".co.jp", ".com.au", ".ca", ".de", ".fr",
                ".in", ".co.in"}

# Two-label public suffixes, so "shop.amazon.co.uk" → registrable "amazon.co.uk"
_MULTI_LABEL_SUFFIXES = {"co.uk", "co.jp", "com.au", "co.in", "com.br", "co.nz", "com.mx", "com.sg"}

# Brand names worth impersonating: the first label of every known retailer
# domain ("amazon", "flipkart", "bestbuy", …). A domain that contains one of
# these but isn't the brand's real domain (cheap-amazon.com, amaz0n-deals.net,
# flipkart-sale.shop) is a classic phishing / scam pattern.
_KNOWN_BRANDS = {d.split(".")[0] for d in DOMAIN_TRUST_DB}
# Short brands that are also everyday words ("wish", "nike" in "nikesh") only
# count when they are a whole hyphen-separated token, not a substring.
_MIN_SUBSTRING_BRAND_LEN = 6
_LOOKALIKE_TRUST = 0.10
# Common digit-for-letter swaps used in look-alike domains
_HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})


def compute_domain_trust(url: str) -> float:
    """
    Rule-based trust score for a given URL.

    Priority:
      1. Match in DOMAIN_TRUST_DB (the exact domain or a real subdomain of it)
      2. Brand look-alike (e.g. cheap-amazon.com) → below the scam threshold
      3. TLD heuristic (scam TLD → low score, trusted TLD → medium score)
      4. Keyword heuristic (suspicious words → lower score)
      5. Deterministic fallback using domain hash → [0.35, 0.70]
    """
    # --- extract bare domain ---
    domain = _extract_domain(url)
    if not domain:
        return 0.5

    # 1. DB lookup on the registrable domain. A plain endswith() check here
    #    used to give "cheap-amazon.com" amazon.com's full trust.
    score = DOMAIN_TRUST_DB.get(_registrable_domain(domain))
    if score is not None:
        return float(np.clip(score + _stable_jitter(domain, scale=0.01), 0.0, 1.0))

    # 2. Impersonation of a known retailer
    if _is_brand_lookalike(domain):
        return round(float(np.clip(_LOOKALIKE_TRUST + _stable_jitter(domain, scale=0.05), 0.0, 0.25)), 4)

    # 3. TLD heuristic
    tld = _get_tld(domain)
    if tld in SCAM_TLDS:
        return round(float(np.clip(0.12 + _stable_jitter(domain, scale=0.08), 0.0, 0.25)), 4)
    if tld not in TRUSTED_TLDS:
        base_trust = 0.45
    else:
        base_trust = 0.60

    # 4. Suspicious keyword heuristic
    suspicious_keywords = [
        "deal", "cheap", "discount", "free", "sale", "win", "prize",
        "offer", "bargain", "flash", "ultra", "mega", "super", "best-price",
        "save", "hot", "limited", "exclusive", "buy-now",
    ]
    hit_count = sum(1 for kw in suspicious_keywords if kw in domain.lower())
    base_trust -= hit_count * 0.07        # each hit reduces trust

    # 5. Deterministic salt so same domain always gets same score
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


# ─────────────────────────────────────────────────────────────────────────────
# Seller resolution for live results
# ─────────────────────────────────────────────────────────────────────────────
# Google Shopping results don't always link straight to the seller: `link` can
# be a Google redirect, a google.com/shopping product page, or missing. Scoring
# that URL would give every live result the same ~0.5–0.6 "unknown .com" trust
# and make the scam filter useless. So trust is computed on the *seller's*
# domain, found from the link when possible and from the `source` name
# ("Flipkart", "Amazon.in", "eBay - seller123") otherwise.

# Normalised seller names that don't reduce to a known domain's first label.
# Matching is exact on purpose: prefix matching would let a seller called
# "Amazon Deals Outlet" inherit Amazon's trust.
_SOURCE_ALIASES = {
    "bh": "bhphotovideo.com", "bhphoto": "bhphotovideo.com", "bhphotovideo": "bhphotovideo.com",
    "bhphotovideoaudio": "bhphotovideo.com",
    "thehomedepot": "homedepot.com", "lowes": "lowes.com", "macys": "macys.com",
    "reliancedigital": "reliancedigital.in", "tatacliq": "tatacliq.com",
    "emisnapmint": "snapmint.com",              # Google Shopping shows "EMI Snapmint"
}
# Seller name → domain for every retailer in DOMAIN_TRUST_DB ("bestbuy" →
# "bestbuy.com"). The first entry per brand wins, i.e. the .com site.
_SOURCE_TO_DOMAIN = {}
for _d in DOMAIN_TRUST_DB:
    _SOURCE_TO_DOMAIN.setdefault(_d.split(".")[0], _d)
_SOURCE_TO_DOMAIN.update(_SOURCE_ALIASES)

# Query parameters Google redirect URLs carry the destination in
_REDIRECT_PARAMS = ("url", "q", "adurl")


def _is_google_domain(domain: str) -> bool:
    label = _registrable_domain(domain).split(".")[0]
    return label in {"google", "googleadservices", "googleusercontent"}


def _unwrap_redirect(url: str) -> str:
    """https://www.google.com/url?url=https://seller.com/p → https://seller.com/p"""
    params = parse_qs(urlparse(url).query)
    for key in _REDIRECT_PARAMS:
        for value in params.get(key, []):
            if value.startswith(("http://", "https://")):
                return value
    return url


def _source_to_domain(source: str) -> Optional[str]:
    """Seller name from SerpAPI → domain, or None when it's not a known retailer."""
    # "eBay - seller123" → "ebay"; "Amazon.com - Seller" → "amazon.com"
    name = re.split(r"\s+[-–|]\s+", source.strip().lower())[0].strip()
    if not name:
        return None
    # "Amazon.in", "walmart.com", "cheap-amazon.com" — already a domain
    if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", name):
        return name
    # "Best Buy" → "bestbuy"; "B&H Photo" → "bhphoto"
    return _SOURCE_TO_DOMAIN.get(re.sub(r"[^a-z0-9]", "", name))


def resolve_seller_domain(item: dict) -> str:
    """
    The seller's domain for a raw listing, or "" if it can't be determined.
    Tries the listing's link (unwrapping Google redirects) first, then the
    `source` seller name.
    """
    for key in ("link", "site_url"):
        url = item.get(key) or ""
        if not url:
            continue
        domain = _extract_domain(url)
        if _is_google_domain(domain):
            domain = _extract_domain(_unwrap_redirect(url))
        if domain and not _is_google_domain(domain):
            return domain
    return _source_to_domain(item.get("source") or "") or ""


def _registrable_domain(domain: str) -> str:
    """"shop.amazon.co.uk" → "amazon.co.uk", "www.amazon.com" → "amazon.com"."""
    parts = domain.split(".")
    n_suffix_labels = 2 if ".".join(parts[-2:]) in _MULTI_LABEL_SUFFIXES else 1
    return ".".join(parts[-(n_suffix_labels + 1):])


def _is_brand_lookalike(domain: str) -> bool:
    """
    True when the domain's name label mentions a known retailer brand but
    the domain isn't one of that brand's real domains (those already
    returned in the DB lookup). Normalises digit homoglyphs first, so
    "amaz0n" counts as "amazon".
    """
    label = _registrable_domain(domain).split(".")[0].translate(_HOMOGLYPHS)
    tokens = set(label.split("-"))
    for brand in _KNOWN_BRANDS:
        if brand in tokens:
            return True
        if len(brand) >= _MIN_SUBSTRING_BRAND_LEN and brand in label:
            return True
    return False


def _get_tld(domain: str) -> str:
    """Return the last two dot-segments as the TLD (handles .co.uk etc.)."""
    parts = domain.split(".")
    if len(parts) >= 3 and len(parts[-2]) <= 3:   # e.g. co.uk, com.au
        return "." + ".".join(parts[-2:])
    elif len(parts) >= 2:
        return "." + parts[-1]
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# User preference store
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: preferences must be kept in a store scoped to one user/session,
# never in a single shared module-level dict. A Streamlit app process is
# shared by every visitor, so a plain module-level dict here would let one
# person's likes/dislikes bleed into everyone else's recommendations.
#
# The functions below take an explicit `prefs` dict (e.g. app.py passes
# `st.session_state.user_prefs`, one per browser session). `_STANDALONE_PREFS`
# only exists as a fallback for CLI/standalone use (`python scraper.py ...`)
# where there's no session to scope to — it is NOT safe to rely on this
# fallback in a multi-user deployment.

_STANDALONE_PREFS: dict[str, float] = {}


def update_user_preference(
    category: str, delta: float, prefs: Optional[dict] = None
) -> dict:
    """
    Called by app.py after a Like (+delta) or Dislike (-delta).
    Keeps scores in [0, 1]. Mutates and returns `prefs` in place (or the
    standalone fallback store if `prefs` is not given).
    """
    store = _STANDALONE_PREFS if prefs is None else prefs
    current = store.get(category, 0.5)
    store[category] = float(np.clip(current + delta, 0.0, 1.0))
    return store


def get_user_preference(category: str, prefs: Optional[dict] = None) -> float:
    store = _STANDALONE_PREFS if prefs is None else prefs
    return store.get(category, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering  ← the core function called by app.py
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    item: dict,
    market_avg_price: Optional[float] = None,
    user_prefs: Optional[dict] = None,
) -> dict:
    """
    Converts a raw product dict (from SerpAPI JSON or mock data)
    into the 4-feature state vector consumed by the DQN model.

    Accepted raw keys (SerpAPI Google Shopping format):
        title, price, extracted_price, link, source, category, rating, reviews

    `user_prefs` should be the caller's session-scoped preference dict (see
    the "User preference store" section above) — pass it through so the
    `user_preference_score` feature reflects the right user's history.

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
    # SerpAPI sometimes provides an old price ("extracted_old_price" is the numeric form)
    old_price = item.get("extracted_old_price") or item.get("old_price") or item.get("was_price")
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
    # Scored on the seller's domain, not whatever `link` points at — see
    # resolve_seller_domain(). An unidentifiable seller gets the neutral 0.5.
    url = item.get("link", "") or item.get("product_link", "") or item.get("site_url", "")
    seller_domain = resolve_seller_domain(item)
    site_trust = compute_domain_trust(seller_domain)

    # --- user preference score ---
    category = item.get("category", "General")
    user_pref = get_user_preference(category, user_prefs)

    # --- metadata ---
    product_name = item.get("title") or item.get("product_name", "Unknown Product")
    source       = item.get("source") or seller_domain or "Unknown seller"

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
        "seller_domain": seller_domain,
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
    "iPhone 15": [
        {"title": "Apple iPhone 15 128GB",                          "extracted_price": 699.00, "old_price": 799.00, "link": "https://www.apple.com/shop/buy-iphone/iphone-15", "source": "Apple", "category": "Electronics"},
        {"title": "iPhone 15 128GB Unlocked",                       "extracted_price": 649.99, "old_price": 729.99, "link": "https://www.bestbuy.com/site/6525421.p",  "source": "Best Buy", "category": "Electronics"},
        {"title": "Apple iPhone 15 128GB (Renewed Premium)",        "extracted_price": 469.00, "old_price": 729.00, "link": "https://www.amazon.com/dp/B0CMPXFKQY",   "source": "Amazon",   "category": "Electronics"},
        {"title": "iPhone 15 128GB Black",                          "extracted_price": 629.00,                      "link": "https://www.walmart.com/ip/5044438434",   "source": "Walmart",  "category": "Electronics"},
        {"title": "iPhone 15 Pro Max 256GB — 85% OFF CLEARANCE",    "extracted_price": 119.00,                      "link": "https://apple-outlet-store.com/iphone15", "source": "apple-outlet-store.com", "category": "Electronics"},
        {"title": "iPhone 15 Wholesale Lot — Limited Stock",        "extracted_price":  89.99,                      "link": "https://phonedeals-mega.top/iphone",      "source": "phonedeals-mega.top", "category": "Electronics"},
    ],
    "MacBook Pro": [
        {"title": "Apple MacBook Pro 14\" M3 8GB/512GB",            "extracted_price": 1399.00, "old_price": 1599.00, "link": "https://www.bestbuy.com/site/6534615.p", "source": "Best Buy", "category": "Electronics"},
        {"title": "Apple MacBook Pro 14\" M3 (2023)",               "extracted_price": 1299.00, "old_price": 1599.00, "link": "https://www.amazon.com/dp/B0CM5JV268",  "source": "Amazon",   "category": "Electronics"},
        {"title": "Apple MacBook Pro 16\" M3 Pro 18GB/512GB",       "extracted_price": 2299.00, "old_price": 2499.00, "link": "https://www.bhphotovideo.com/c/product/1793636", "source": "B&H Photo", "category": "Electronics"},
        {"title": "Apple MacBook Pro 13\" M1 (Renewed)",            "extracted_price":  479.00, "old_price": 1299.00, "link": "https://www.amazon.com/dp/B08N5LNQCX",  "source": "Amazon",   "category": "Electronics"},
        {"title": "MacBook Pro M3 — 90% OFF TODAY ONLY",            "extracted_price":  159.99,                       "link": "https://macbook-sale.xyz/m3",           "source": "macbook-sale.xyz", "category": "Electronics"},
        {"title": "Apple MacBook Pro 14 M3 Sealed",                 "extracted_price":  399.00,                       "link": "https://amazon-deals-outlet.net/mbp",   "source": "amazon-deals-outlet.net", "category": "Electronics"},
    ],
    "Gaming Chair": [
        {"title": "Corsair T3 Rush Gaming Chair",                   "extracted_price": 299.99, "old_price": 349.99, "link": "https://www.bestbuy.com/site/6509960.p",  "source": "Best Buy", "category": "Home & Garden"},
        {"title": "GTRACING Gaming Chair with Footrest",            "extracted_price": 139.99, "old_price": 199.99, "link": "https://www.amazon.com/dp/B07R6WN6ZJ",   "source": "Amazon",   "category": "Home & Garden"},
        {"title": "Respawn 110 Racing Style Gaming Chair",          "extracted_price": 159.00,                      "link": "https://www.walmart.com/ip/55446452",     "source": "Walmart",  "category": "Home & Garden"},
        {"title": "DXRacer Formula Series Gaming Chair",            "extracted_price": 199.00, "old_price": 349.00, "link": "https://www.newegg.com/p/N82E16811996101", "source": "Newegg", "category": "Home & Garden"},
        {"title": "Pro Gaming Chair 95% OFF — Last 3 In Stock",     "extracted_price":  14.99,                      "link": "https://chair-flashsale.icu/pro",         "source": "chair-flashsale.icu", "category": "Home & Garden"},
    ],
    "default": [
        {"title": "Product A - Great Deal",                         "extracted_price":  49.99, "old_price":  79.99, "link": "https://www.amazon.com/dp/XXXXXXXXXX",   "source": "Amazon",   "category": "General"},
        {"title": "Product B - Standard Price",                     "extracted_price":  89.00,                      "link": "https://www.walmart.com/ip/123456789",   "source": "Walmart",  "category": "General"},
        {"title": "Product C - UNBELIEVABLE PRICE 80% OFF",         "extracted_price":   9.99,                      "link": "https://discount-mega.ru/deal",          "source": "discount-mega.ru", "category": "General"},
        {"title": "Product D - Verified Seller",                    "extracted_price": 134.00, "old_price": 150.00, "link": "https://www.bestbuy.com/site/XXXXXXX.p", "source": "Best Buy", "category": "General"},
    ],
}

# Words that route a query to a mock set, so "iphone", "sony wh-1000xm5" or
# "office chair" find the right listings instead of the generic default.
_MOCK_KEYWORDS = {
    "Sony Headphones": ["sony", "headphone"],
    "Nike Shoes":      ["nike", "shoe", "sneaker"],
    "iPhone 15":       ["iphone"],
    "MacBook Pro":     ["macbook", "laptop"],
    "Gaming Chair":    ["chair"],
}


def match_mock_set(query: str) -> Optional[str]:
    """Name of the mock set a query maps to, or None if only the generic default fits."""
    q = query.lower()
    for name, keywords in _MOCK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return name
    return None


def fetch_mock_results(query: str) -> list[dict]:
    """Returns realistic mock search results (mix of legit + scam)."""
    return _MOCK_PRODUCTS[match_mock_set(query) or "default"]


# ─────────────────────────────────────────────────────────────────────────────
# SerpAPI fetcher
# ─────────────────────────────────────────────────────────────────────────────

def _redact_key(message: str) -> str:
    """Network errors can echo the request URL, api_key included — never surface it."""
    return message.replace(SERPAPI_KEY, "***") if SERPAPI_KEY else message


def fetch_serpapi_results(
    query: str, num_results: int = 10, country: str = DEFAULT_COUNTRY
) -> tuple[list[dict], Optional[str]]:
    """
    Fetches Google Shopping results from SerpAPI for the given country code
    (a key of COUNTRIES).

    Returns (raw_results, fallback_reason). On any problem it falls back to
    mock data and says why in `fallback_reason`, so the UI can tell the user
    instead of silently showing demo listings; the reason is None when the
    results are live.
    """
    if not SERPAPI_AVAILABLE:
        reason = "the SerpAPI client (google-search-results) isn't installed"
    elif not SERPAPI_KEY:
        reason = "no SERPAPI_KEY is set"
    else:
        try:
            params = {
                "engine":    "google_shopping",
                "q":         query,
                "api_key":   SERPAPI_KEY,
                "num":       num_results,
                "gl":        country,
                "hl":        "en",
            }
            results  = GoogleSearch(params).get_dict()
            shopping = results.get("shopping_results", [])

            if results.get("error"):
                # e.g. "Your account has run out of searches." or an invalid key
                reason = f"SerpAPI error: {results['error']}"
            elif not shopping:
                reason = f"SerpAPI found no shopping results for '{query}'"
            else:
                print(f"[scraper] SerpAPI returned {len(shopping)} results for '{query}' ({country}).")
                return shopping[:num_results], None

        except Exception as exc:
            reason = f"SerpAPI request failed: {exc}"

    reason = _redact_key(reason)
    print(f"[scraper] {reason} → using mock data.")
    return fetch_mock_results(query), reason


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by app.py
# ─────────────────────────────────────────────────────────────────────────────

MARKET_REFERENCE_MIN_TRUST = 0.7


def _parse_price(value) -> float:
    if isinstance(value, str):
        value = re.sub(r"[^\d.]", "", value) or 0
    return float(value or 0)


def estimate_market_prices(raw_results: list[dict]) -> list[Optional[float]]:
    """
    The market price each listing is compared against (→ normalized_price,
    which the agent uses both to value a deal and to spot a price that's too
    good to be true).

      * A trusted seller's own list price ("was ₹79,900"), when it gives one —
        the most specific reference, and immune to a search that mixes a
        flagship with a budget model.
      * Otherwise the median price among trusted sellers in the results.
        Scam lures are deliberately far below market, so including them
        dragged the plain median down and made real retailers look
        overpriced.
      * Otherwise (fewer than 2 trusted sellers) the median of all results.
    """
    prices = [_parse_price(item.get("extracted_price") or item.get("price", 0)) for item in raw_results]
    trusts = [compute_domain_trust(resolve_seller_domain(item)) for item in raw_results]

    trusted = [p for p, t in zip(prices, trusts) if p > 0 and t >= MARKET_REFERENCE_MIN_TRUST]
    everyone = [p for p in prices if p > 0]
    pool = trusted if len(trusted) >= 2 else everyone
    reference = float(np.median(pool)) if pool else None

    market = []
    for item, price, trust in zip(raw_results, prices, trusts):
        list_price = _parse_price(item.get("extracted_old_price") or item.get("old_price") or item.get("was_price") or 0)
        if trust >= MARKET_REFERENCE_MIN_TRUST and list_price > price > 0:
            market.append(list_price)
        else:
            market.append(reference)
    return market


def search_products(
    query: str,
    use_mock: bool = False,
    num_results: int = 10,
    user_prefs: Optional[dict] = None,
    country: str = DEFAULT_COUNTRY,
) -> list[dict]:
    """Feature dicts for a query — see search_products_detailed()."""
    return search_products_detailed(query, use_mock, num_results, user_prefs, country)["results"]


def search_products_detailed(
    query: str,
    use_mock: bool = False,
    num_results: int = 10,
    user_prefs: Optional[dict] = None,
    country: str = DEFAULT_COUNTRY,
    fetcher=None,
) -> dict:
    """
    Main entry point for app.py.

    `fetcher(query, num_results, country) -> (raw_results, fallback_reason)`
    replaces fetch_serpapi_results for live searches — app.py passes a cached
    version so repeat searches don't spend SerpAPI quota.

    1. Fetches raw results (SerpAPI or mock).
    2. Runs feature engineering on every item.
    3. Reports where the results came from.

    Args:
        query:       User search string (e.g. "Sony Headphones").
        use_mock:    Force mock data regardless of API availability.
        num_results: Max number of listings to return.
        user_prefs:  Caller's session-scoped preference dict, forwarded to
                     extract_features() so results reflect the right user.
        country:     Google Shopping country code (a key of COUNTRIES).

    Returns:
        {
            "results":         list of feature dicts (see extract_features()),
                               each with a "currency" symbol added,
            "source":          "live" or "mock",
            "fallback_reason": why live search wasn't used, or None if it was
                               or mock data was requested,
            "mock_matched":    False when mock data had no listings for this
                               query and the generic default set was used,
        }
    """
    fallback_reason = None
    if use_mock:
        raw_results = fetch_mock_results(query)
    else:
        raw_results, fallback_reason = (fetcher or fetch_serpapi_results)(query, num_results, country)
    is_live = not use_mock and fallback_reason is None
    currency = COUNTRIES.get(country, COUNTRIES[DEFAULT_COUNTRY])[1] if is_live else MOCK_CURRENCY

    feature_list = []
    for item, market_price in zip(raw_results, estimate_market_prices(raw_results)):
        try:
            features = extract_features(item, market_avg_price=market_price, user_prefs=user_prefs)
            features["currency"] = currency
            feature_list.append(features)
        except Exception as exc:
            print(f"[scraper] Skipping item due to feature error: {exc}")

    return {
        "results":         feature_list,
        "source":          "live" if is_live else "mock",
        "fallback_reason": fallback_reason,
        "mock_matched":    is_live or match_mock_set(query) is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI test harness
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        # Console output below uses non-ASCII markers (→, 🚨, ✅); on Windows
        # the default console codepage (cp1252) can't encode them and this
        # script would crash with a UnicodeEncodeError otherwise.
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Test the scraper pipeline.")
    parser.add_argument("query", nargs="?", default="Sony Headphones",
                        help="Product search query")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock data (no API call)")
    parser.add_argument("--country", default=DEFAULT_COUNTRY, choices=sorted(COUNTRIES),
                        help=f"Google Shopping country (default: {DEFAULT_COUNTRY})")
    args = parser.parse_args()

    search = search_products_detailed(args.query, use_mock=args.mock, country=args.country)
    products = search["results"]
    print(f"\nSearching for: '{args.query}' [{search['source'].upper()}]")
    if search["fallback_reason"]:
        print(f"Live search unavailable: {search['fallback_reason']}")
    print()

    # Seller name and the domain trust was scored on — the columns to check
    # when a live result's trust looks wrong (unrecognised sellers show "?").
    print(f"{'#':<4} {'Product':<38} {'Price':>12} {'Disc%':>7} {'Trust':>6}   {'Seller':<26} {'Scored as'}")
    print("-" * 120)
    for i, feat in enumerate(products, 1):
        name   = feat["product_name"][:36]
        price  = f"{feat['currency']}{feat['price']:,.0f}"
        disc   = feat["discount_percentage"]
        trust  = feat["site_trust_score"]
        seller = str(feat["source"])[:26]
        scored = feat["seller_domain"] or "?"
        flag   = "🚨" if trust < 0.3 else "✅"
        print(f"{i:<4} {name:<38} {price:>12} {disc:>7.1%} {trust:>6.2f} {flag} {seller:<26} {scored}")

    print(f"\n{len(products)} products processed.\n")
    print("Tip: pass these feature dicts to ShoppingEnv.features_to_obs(f) "
          "then model.predict(obs) to get the agent's recommendation.")
