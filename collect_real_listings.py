"""
Collect real Google Shopping listings for evaluation (see evaluate_real.py).

Runs each query in QUERIES once through SerpAPI, saves the raw results to
real_data/raw/, and adds the first few listings per query to
real_data/labels.csv for hand-labelling. Saved queries are skipped on later
runs, so re-running never spends quota twice.

The labelling sheet deliberately does NOT show the model's decisions or the
trust score, so they can't bias the labels.

Usage (needs SERPAPI_KEY; spends one search per new query):
    python collect_real_listings.py                 # India, default queries
    python collect_real_listings.py --country us
    python collect_real_listings.py --per-query 10
"""

import argparse
import json
import os
import re
import sys
from datetime import date

import pandas as pd

from scraper import COUNTRIES, DEFAULT_COUNTRY, SERPAPI_AVAILABLE, SERPAPI_KEY, fetch_serpapi_results

DATA_DIR   = "real_data"
RAW_DIR    = os.path.join(DATA_DIR, "raw")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")

# A spread of categories and price points. The last three tend to surface
# replica / counterfeit sellers — real scams are otherwise rare on Google
# Shopping (merchants are vetted), and without some the scam-detection side
# of the evaluation would have nothing to measure.
QUERIES = [
    "iPhone 15",
    "Samsung Galaxy S24",
    "boAt Airdopes earbuds",
    "Sony WH-1000XM5",
    "HP Victus gaming laptop",
    "Apple Watch Series 9",
    "Nike Air Force 1",
    "Philips air fryer",
    "Prestige pressure cooker",
    "Atomic Habits book",
    "LEGO Technic",
    "Maybelline foundation",
    "Jordan 1 first copy shoes",
    "Rolex Submariner watch",
    "Ray-Ban Aviator sunglasses",
]

# Fields kept from each SerpAPI result — everything the pipeline uses plus a
# little context for the labeller. Nothing from search_metadata /
# search_parameters is saved, so the API key can't end up in the repo.
KEPT_FIELDS = [
    "position", "title", "source", "price", "extracted_price", "old_price",
    "extracted_old_price", "link", "product_link", "rating", "reviews", "delivery",
    # for reference prices (reference_prices.py) — page tokens don't contain the key
    "product_id", "immersive_product_page_token",
]

LABEL_COLUMNS = [
    "id", "query", "country", "position", "title", "seller", "price", "old_price", "link",
    "legit", "good_deal", "notes",
]


def slugify(query: str, country: str) -> str:
    return f"{country}_" + re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")


def trim_result(result: dict) -> dict:
    return {key: result[key] for key in KEPT_FIELDS if key in result}


def label_rows(query: str, country: str, results: list[dict], per_query: int) -> list[dict]:
    slug = slugify(query, country)
    rows = []
    for i, r in enumerate(results[:per_query]):
        rows.append({
            "id":        f"{slug}_{i}",
            "query":     query,
            "country":   country,
            "position":  i,                       # index into the saved raw results
            "title":     r.get("title", ""),
            "seller":    r.get("source", ""),
            "price":     r.get("price", r.get("extracted_price", "")),
            "old_price": r.get("old_price", r.get("extracted_old_price", "")),
            "link":      r.get("link") or r.get("product_link", ""),
            "legit":     "",
            "good_deal": "",
            "notes":     "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real listings for evaluation.")
    parser.add_argument("--country", default=DEFAULT_COUNTRY, choices=sorted(COUNTRIES))
    parser.add_argument("--per-query", type=int, default=10,
                        help="Listings per query added to the labelling sheet (all are saved)")
    parser.add_argument("--num", type=int, default=40, help="Results requested per search")
    args = parser.parse_args()

    if not (SERPAPI_AVAILABLE and SERPAPI_KEY):
        sys.exit("Needs SERPAPI_KEY set and `pip install google-search-results`.")

    os.makedirs(RAW_DIR, exist_ok=True)
    labels = pd.read_csv(LABELS_CSV, dtype=str).fillna("") if os.path.exists(LABELS_CSV) \
        else pd.DataFrame(columns=LABEL_COLUMNS)

    todo = [q for q in QUERIES if not os.path.exists(os.path.join(RAW_DIR, slugify(q, args.country) + ".json"))]
    print(f"{len(QUERIES) - len(todo)} queries already saved; {len(todo)} to fetch "
          f"(= {len(todo)} SerpAPI searches).")

    new_rows = []
    for query in todo:
        results, reason = fetch_serpapi_results(query, args.num, args.country)
        if reason:
            # fetch_serpapi_results falls back to mock data on failure — never
            # save that as "real" data.
            print(f"  ✗ {query}: {reason} — skipped")
            if "run out of searches" in reason:
                break
            continue

        trimmed = [trim_result(r) for r in results]
        slug = slugify(query, args.country)
        with open(os.path.join(RAW_DIR, slug + ".json"), "w", encoding="utf-8") as f:
            json.dump({"query": query, "country": args.country, "collected": date.today().isoformat(),
                       "results": trimmed}, f, ensure_ascii=False, indent=1)
        new_rows += label_rows(query, args.country, trimmed, args.per_query)
        print(f"  ✓ {query}: {len(trimmed)} results saved")

    if new_rows:
        labels = pd.concat([labels, pd.DataFrame(new_rows, columns=LABEL_COLUMNS)], ignore_index=True)
        labels.to_csv(LABELS_CSV, index=False, encoding="utf-8-sig")   # BOM so Excel shows ₹ correctly
    print(f"\n{LABELS_CSV}: {len(labels)} listings, "
          f"{(labels['legit'] == '').sum()} still to label. See real_data/LABELLING.md.")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
