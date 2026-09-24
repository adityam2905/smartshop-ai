"""
Product search: fetches Google Shopping results through SerpAPI (or the demo
listings) and turns each one into the model's inputs.

This is the pipeline app.py calls: fetch → estimate the usual price → build
features. Try it from the command line:

    python -m smartshop.search "Sony Headphones" --mock
    python -m smartshop.search "iPhone 15" --country in     # needs SERPAPI_KEY
"""

import argparse
import os
import sys
from typing import Optional

from .features import extract_features
from .mock_data import MOCK_CURRENCY, fetch_mock_results, match_mock_set
from .pricing import estimate_market_references

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
                print(f"[search] SerpAPI returned {len(shopping)} results for '{query}' ({country}).")
                return shopping[:num_results], None

        except Exception as exc:
            reason = f"SerpAPI request failed: {exc}"

    reason = _redact_key(reason)
    print(f"[search] {reason} → using mock data.")
    return fetch_mock_results(query), reason


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by app.py
# ─────────────────────────────────────────────────────────────────────────────

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
    reference_fn=None,
) -> dict:
    """
    Main entry point for app.py.

    `fetcher(query, num_results, country) -> (raw_results, fallback_reason)`
    replaces fetch_serpapi_results for live searches — app.py passes a cached
    version so repeat searches don't spend SerpAPI quota.
    `reference_fn(raw_results) -> [reference price or None, …]` supplies
    per-product reference prices for live results (reference_prices.py);
    each costs a SerpAPI search, so the app only passes it when enabled.

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
    reference_prices = reference_fn(raw_results) if (reference_fn and is_live) else None
    for item, (market_price, basis) in zip(raw_results, estimate_market_references(raw_results, reference_prices)):
        try:
            features = extract_features(item, market_avg_price=market_price, user_prefs=user_prefs)
            features["currency"] = currency
            features["market_basis"] = basis
            feature_list.append(features)
        except Exception as exc:
            print(f"[search] Skipping item due to feature error: {exc}")

    return {
        "results":         feature_list,
        "source":          "live" if is_live else "mock",
        "fallback_reason": fallback_reason,
        "mock_matched":    is_live or match_mock_set(query) is not None,
    }


def main() -> None:
    """Print each result's seller, trust and price — handy for checking live data."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")   # ₹, 🚨 and ✅ on Windows consoles

    parser = argparse.ArgumentParser(description="Search and show how each listing is scored.")
    parser.add_argument("query", nargs="?", default="Sony Headphones")
    parser.add_argument("--mock", action="store_true", help="Use the demo listings (no API call)")
    parser.add_argument("--country", default=DEFAULT_COUNTRY, choices=sorted(COUNTRIES))
    args = parser.parse_args()

    search = search_products_detailed(args.query, use_mock=args.mock, country=args.country)
    print()
    print(f"Searching for: '{args.query}' [{search['source'].upper()}]")
    if search["fallback_reason"]:
        print(f"Live search unavailable: {search['fallback_reason']}")
    print()

    # "Scored as" is the seller domain trust was based on ("?" = unrecognised seller)
    print(f"{'#':<4} {'Product':<38} {'Price':>12} {'Disc%':>7} {'Trust':>6}   {'Seller':<26} {'Scored as'}")
    print("-" * 120)
    for i, feat in enumerate(search["results"], 1):
        price = f"{feat['currency']}{feat['price']:,.0f}"
        trust = feat["site_trust_score"]
        flag  = "🚨" if trust < 0.3 else "✅"
        print(f"{i:<4} {feat['product_name'][:36]:<38} {price:>12} {feat['discount_percentage']:>7.1%} "
              f"{trust:>6.2f} {flag} {str(feat['source'])[:26]:<26} {feat['seller_domain'] or '?'}")
    print()
    print(f"{len(search['results'])} products.")


if __name__ == "__main__":
    main()
