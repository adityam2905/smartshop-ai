"""
Seller trust: how trustworthy a listing's seller looks, from 0 (scam) to 1.

Live Google Shopping results link to a Google page, not the shop, so the
seller is first identified (resolve_seller_domain) and then scored
(compute_domain_trust) from a table of known retailers plus warning signs:
scam domain endings, look-alike brand names, spammy keywords.
"""

import hashlib
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

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
