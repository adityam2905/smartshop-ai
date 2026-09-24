"""
Reference ("usual") prices: what other stores charge for the exact same
product, from Google Shopping's product page (SerpAPI google_immersive_product).

Each lookup costs one SerpAPI search, so the app only uses this when
SERPAPI_REFERENCE_PRICES is set. The evaluation cache is built by
experiments/fetch_reference_prices.py.
"""

import json
import re
from typing import Optional

import numpy as np

from . import search
from .config import REAL_DATA_DIR, SCAM_TRUST_THRESHOLD
from .pricing import same_product, title_profile
from .trust import compute_domain_trust, resolve_seller_domain

MIN_OTHER_STORES = 1          # other stores needed before a reference counts
REFERENCE_CACHE = REAL_DATA_DIR / "reference_prices.json"


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower().split(" - ")[0])


def fetch_product_stores(page_token: str) -> tuple[list[dict], Optional[str]]:
    """Every store's offer for one product: ([{name, price, link}, …], error or None)."""
    if not (search.SERPAPI_AVAILABLE and search.SERPAPI_KEY):
        return [], "SerpAPI not configured"
    try:
        result = search.GoogleSearch({
            "engine": "google_immersive_product",
            "page_token": page_token,
            "api_key": search.SERPAPI_KEY,
        }).get_dict()
    except Exception as exc:
        return [], search._redact_key(f"request failed: {exc}")
    if result.get("error"):
        return [], search._redact_key(str(result["error"]))
    stores = (result.get("product_results") or {}).get("stores") or result.get("stores") or []
    return [
        {"name": s.get("name", ""), "price": float(s["extracted_price"]), "link": s.get("link", "")}
        for s in stores if s.get("extracted_price")
    ], None


def reference_from_stores(stores: list[dict], own_seller: str) -> tuple[Optional[float], int]:
    """
    Median price of the product across the *other* stores that aren't
    hard-blocked scams, and how many stores that was. The listing's own store
    is excluded so a listing can't set its own reference.
    """
    own = _norm_name(own_seller)
    prices = []
    for s in stores:
        if own and _norm_name(s["name"]) == own:
            continue
        trust = compute_domain_trust(resolve_seller_domain({"source": s["name"], "link": s["link"]}))
        if trust >= SCAM_TRUST_THRESHOLD:
            prices.append(s["price"])
    if len(prices) < MIN_OTHER_STORES:
        return None, len(prices)
    return float(np.median(prices)), len(prices)


def fetch_reference_prices(raw_results: list[dict], max_products: int, fetch=fetch_product_stores) -> list[Optional[float]]:
    """
    Reference prices aligned with `raw_results` (None where not fetched), for
    at most `max_products` listings that carry a page token — used by the app.
    """
    refs: list[Optional[float]] = [None] * len(raw_results)
    for i, item in enumerate(raw_results):
        if max_products <= 0:
            break
        token = item.get("immersive_product_page_token")
        if not token or compute_domain_trust(resolve_seller_domain(item)) < SCAM_TRUST_THRESHOLD:
            continue                       # hard-blocked anyway — don't spend a search on it
        stores, _ = fetch(token)
        refs[i], _ = reference_from_stores(stores, item.get("source", ""))
        max_products -= 1
    return refs


def load_reference_cache(path=REFERENCE_CACHE) -> dict:
    """The evaluation cache: labelled listing id → {\"reference\": price, …}."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ── Finding a saved listing again in fresh search results ───────────────────

def product_ids(item: dict) -> set[str]:
    """All of Google's IDs for a listing — the same product carries several
    (catalog, product, group), and different searches expose different ones."""
    ids = set(re.findall(r"(?:catalogid|productid|gpcid|mid):(\d+)", item.get("product_link") or ""))
    if item.get("product_id"):
        ids.add(str(item["product_id"]))
    return ids


def product_id(item: dict) -> Optional[str]:
    ids = sorted(product_ids(item))
    return ids[0] if ids else None


def find_listing(listing: dict, seller: str, candidates: list[dict]) -> Optional[dict]:
    """The candidate for the same product: shared Google ID, else same seller and title."""
    ids = product_ids(listing)
    for c in candidates:
        if ids & product_ids(c) and c.get("immersive_product_page_token"):
            return c
    profile = title_profile(listing.get("title", ""))
    same = [c for c in candidates if c.get("immersive_product_page_token")
            and same_product(profile, title_profile(c.get("title", "")))]
    for c in same:
        if _norm_name(c.get("source", "")) == _norm_name(seller):
            return c
    return same[0] if same else None
