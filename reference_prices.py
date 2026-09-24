"""
Reference ("usual") prices from Google Shopping's product pages, via SerpAPI.

A single search's results often don't reveal what a product normally costs —
the main source of errors in the real-data evaluation. Google's product page
lists every store selling that exact product, so the median of the *other*
stores' prices is a far better "usual price" than anything inferred from one
search. SerpAPI exposes it as engine=google_immersive_product, keyed by the
`immersive_product_page_token` on each google_shopping result.

Costs one SerpAPI search per product, so:
  * the app only uses it when SERPAPI_REFERENCE_PRICES is set (see app.py),
    for a few listings per search, with long caching;
  * for evaluation, `python reference_prices.py` fetches references for the
    labelled listings once and caches them in real_data/reference_prices.json
    (resumable; stops before your quota runs low).
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

import numpy as np

import scraper
from scraper import HARD_BLOCK_TRUST, _redact_key, compute_domain_trust, resolve_seller_domain

MIN_OTHER_STORES = 1          # stores other than the listing's own needed for a reference
REFERENCE_CACHE = os.path.join("real_data", "reference_prices.json")


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower().split(" - ")[0])


def fetch_product_stores(page_token: str) -> tuple[list[dict], Optional[str]]:
    """Every store's offer for one product: ([{name, price, link}, …], error or None)."""
    if not (scraper.SERPAPI_AVAILABLE and scraper.SERPAPI_KEY):
        return [], "SerpAPI not configured"
    try:
        result = scraper.GoogleSearch({
            "engine": "google_immersive_product",
            "page_token": page_token,
            "api_key": scraper.SERPAPI_KEY,
        }).get_dict()
    except Exception as exc:
        return [], _redact_key(f"request failed: {exc}")
    if result.get("error"):
        return [], _redact_key(str(result["error"]))
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
        if trust >= HARD_BLOCK_TRUST:
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
        if not token or compute_domain_trust(resolve_seller_domain(item)) < HARD_BLOCK_TRUST:
            continue                       # hard-blocked anyway — don't spend a search on it
        stores, _ = fetch(token)
        refs[i], _ = reference_from_stores(stores, item.get("source", ""))
        max_products -= 1
    return refs


def load_reference_cache(path: str = REFERENCE_CACHE) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Building the evaluation cache
# ─────────────────────────────────────────────────────────────────────────────

def _product_ids(item: dict) -> set[str]:
    """All of Google's IDs for a listing — the same product carries several
    (catalog, product, group), and different searches expose different ones."""
    ids = set(re.findall(r"(?:catalogid|productid|gpcid|mid):(\d+)", item.get("product_link") or ""))
    if item.get("product_id"):
        ids.add(str(item["product_id"]))
    return ids


def _product_id(item: dict) -> Optional[str]:
    ids = sorted(_product_ids(item))
    return ids[0] if ids else None


def _find_listing(listing: dict, seller: str, candidates: list[dict]) -> Optional[dict]:
    """The candidate for the same product: shared Google ID, else same seller and title."""
    ids = _product_ids(listing)
    for c in candidates:
        if ids & _product_ids(c) and c.get("immersive_product_page_token"):
            return c
    profile = scraper.title_profile(listing.get("title", ""))
    same = [c for c in candidates if c.get("immersive_product_page_token")
            and scraper.same_product(profile, scraper.title_profile(c.get("title", "")))]
    for c in same:
        if _norm_name(c.get("source", "")) == _norm_name(seller):
            return c
    return same[0] if same else None


def _searches_left() -> Optional[int]:
    """SerpAPI's account endpoint — doesn't count as a search."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"https://serpapi.com/account.json?api_key={scraper.SERPAPI_KEY}", timeout=30) as r:
            return json.load(r).get("total_searches_left")
    except Exception:
        return None


def build_cache(labels_csv: str, raw_dir: str, max_searches: int, reserve: int) -> None:
    import pandas as pd

    labels = pd.read_csv(labels_csv, dtype=str).fillna("")
    labels = labels[labels["seller"].str.strip() != ""]
    cache = load_reference_cache()
    searches = 0

    def budget_ok() -> bool:
        left = _searches_left()
        return searches < max_searches and (left is None or left > reserve)

    for (query, country), group in labels.groupby(["query", "country"], sort=False):
        todo = group[~group["id"].isin(cache)]
        if todo.empty:
            continue
        if not budget_ok():
            print("Stopping: search budget reached."); break

        # 1. Re-run the search to get page tokens (the saved raw results predate them)
        fresh, reason = scraper.fetch_serpapi_results(query, 40, country)
        searches += 1
        if reason:
            print(f"  ✗ {query}: {reason}"); continue

        slug = todo["id"].iloc[0].rsplit("_", 1)[0]
        with open(os.path.join(raw_dir, slug + ".json"), encoding="utf-8") as f:
            saved = json.load(f)["results"]

        # 2. Fetch each labelled product's store list (one search per distinct product)
        stores_by_pid: dict[str, list[dict]] = {}
        for _, row in todo.iterrows():
            listing = saved[int(row["position"])]
            match = _find_listing(listing, row["seller"], fresh)
            if match is None and budget_ok():
                # Results shift over time — look the listing up by its exact title
                by_title, reason = scraper.fetch_serpapi_results(listing.get("title", ""), 10, country)
                searches += 1
                match = None if reason else _find_listing(listing, row["seller"], by_title)
            pid = _product_id(match) if match else _product_id(listing)
            entry = {"product_id": pid, "reference": None, "n_stores": 0}
            if not match:
                entry["note"] = "product not found again"
            else:
                if pid not in stores_by_pid:
                    if not budget_ok():
                        print("Stopping: search budget reached."); break
                    stores_by_pid[pid], err = fetch_product_stores(match["immersive_product_page_token"])
                    searches += 1
                    if err:
                        entry["note"] = err
                stores = stores_by_pid[pid]
                entry["reference"], entry["n_stores"] = reference_from_stores(stores, row["seller"])
                entry["stores"] = stores
            cache[row["id"]] = entry
            with open(REFERENCE_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)
        got = sum(1 for i in todo["id"] if cache.get(i, {}).get("reference"))
        print(f"  ✓ {query}: references for {got}/{len(todo)} listings ({searches} searches so far)")

    have = sum(1 for v in cache.values() if v.get("reference"))
    print(f"\n{REFERENCE_CACHE}: {have}/{len(labels)} labelled listings have a reference price; "
          f"{searches} SerpAPI searches used this run.")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Fetch reference prices for the labelled real listings.")
    parser.add_argument("--labels", default=os.path.join("real_data", "labels.csv"))
    parser.add_argument("--raw-dir", default=os.path.join("real_data", "raw"))
    parser.add_argument("--max-searches", type=int, default=130)
    parser.add_argument("--reserve", type=int, default=40,
                        help="Stop when this many SerpAPI searches are left this month")
    args = parser.parse_args()
    if not (scraper.SERPAPI_AVAILABLE and scraper.SERPAPI_KEY):
        sys.exit("Needs SERPAPI_KEY and `pip install google-search-results`.")
    build_cache(args.labels, args.raw_dir, args.max_searches, args.reserve)
