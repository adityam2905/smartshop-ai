"""
Build real_data/reference_prices.json: the usual price of each labelled real
listing, from what other stores charge (see smartshop/reference_prices.py).

    python -m experiments.fetch_reference_prices      # needs SERPAPI_KEY

Spends SerpAPI searches: one to re-run each query (to get page tokens), one
per distinct product. Resumable, and stops before your monthly quota runs low.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from smartshop import search
from smartshop.config import REAL_DATA_DIR
from smartshop.reference_prices import (
    REFERENCE_CACHE, fetch_product_stores, find_listing, load_reference_cache, product_id, reference_from_stores,
)


def _searches_left() -> Optional[int]:
    """SerpAPI's account endpoint — doesn't count as a search."""
    try:
        with urllib.request.urlopen(f"https://serpapi.com/account.json?api_key={search.SERPAPI_KEY}", timeout=30) as r:
            return json.load(r).get("total_searches_left")
    except Exception:
        return None


def build_cache(labels_csv: Path, raw_dir: Path, max_searches: int, reserve: int) -> None:
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
        fresh, reason = search.fetch_serpapi_results(query, 40, country)
        searches += 1
        if reason:
            print(f"  ✗ {query}: {reason}"); continue

        slug = todo["id"].iloc[0].rsplit("_", 1)[0]
        with open(raw_dir / f"{slug}.json", encoding="utf-8") as f:
            saved = json.load(f)["results"]

        # 2. Fetch each labelled product's store list (one search per distinct product)
        stores_by_pid: dict[str, list[dict]] = {}
        for _, row in todo.iterrows():
            listing = saved[int(row["position"])]
            match = find_listing(listing, row["seller"], fresh)
            if match is None and budget_ok():
                # Results shift over time — look the listing up by its exact title
                by_title, reason = search.fetch_serpapi_results(listing.get("title", ""), 10, country)
                searches += 1
                match = None if reason else find_listing(listing, row["seller"], by_title)
            pid = product_id(match) if match else product_id(listing)
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


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Fetch reference prices for the labelled real listings.")
    parser.add_argument("--labels", default=REAL_DATA_DIR / "labels.csv", type=Path)
    parser.add_argument("--raw-dir", default=REAL_DATA_DIR / "raw", type=Path)
    parser.add_argument("--max-searches", type=int, default=130)
    parser.add_argument("--reserve", type=int, default=40,
                        help="Stop when this many SerpAPI searches are left this month")
    args = parser.parse_args()
    if not (search.SERPAPI_AVAILABLE and search.SERPAPI_KEY):
        sys.exit("Needs SERPAPI_KEY and `pip install google-search-results`.")
    build_cache(args.labels, args.raw_dir, args.max_searches, args.reserve)


if __name__ == "__main__":
    main()
