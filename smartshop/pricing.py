"""
The "usual" market price each listing is compared against.

normalized_price = price ÷ market price is the model's main signal, both for
valuing a deal and for spotting a price that's too good to be true. This
module parses prices, decides which listings are the same product, and picks
the best available market price for each listing.
"""

import re
from typing import Optional

import numpy as np

from .config import SCAM_TRUST_THRESHOLD
from .trust import compute_domain_trust, resolve_seller_domain

MARKET_REFERENCE_MIN_TRUST = 0.7      # only sellers this trusted set the "usual" price


def parse_price(value) -> float:
    """"₹1,01,559" / "$49.99" / 49.99 → 101559.0 / 49.99 / 49.99 (0 if missing)."""
    if isinstance(value, str):
        value = re.sub(r"[^\d.]", "", value) or 0
    return float(value or 0)


_CURRENCY_AMOUNT = re.compile(r"(?:₹|\$|£|€|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)


def parse_old_price(item: dict) -> float:
    """
    The listing's "was" price, or 0 if none. SerpAPI's `extracted_old_price`
    can't be trusted on its own: for "32% off₹34,990" it returns 32 (the
    percentage). Prefer the last currency amount in the `old_price` text
    ("Usually ₹44,410", "Was ₹768", "32% off₹34,990" → 34990), then fall back
    to the extracted number, then to the bare text.
    """
    text = item.get("old_price") or item.get("was_price") or ""
    if isinstance(text, (int, float)):
        return float(text)
    amounts = _CURRENCY_AMOUNT.findall(str(text))
    if amounts:
        return float(amounts[-1].replace(",", ""))
    if item.get("extracted_old_price"):
        return float(item["extracted_old_price"])
    return parse_price(text)


# ── Which listings are the same product? ─────────────────────────────────────
# Search results mix variants (128GB vs 256GB, 18 ml vs 30 ml, Pro vs Pro Max)
# and sometimes unrelated products. Comparing a listing against the median of
# the whole search made "below market" meaningless: in the real-data
# evaluation only 14 of 41 listings judged ≥5% below market were real deals.
# So each listing is compared only against listings for the same product.

# Words that change which product it is (and its price class)
_TIER_WORDS = {"pro", "max", "plus", "ultra", "mini", "lite", "fe", "se", "air", "elite", "neo"}
_USED_WORDS = {"refurbished", "renewed", "recertified", "used", "preowned", "pre-owned",
               "unboxed", "open-box", "openbox", "as-is", "second-hand", "secondhand"}
# Variant attributes that must not conflict between two listings of "the same" product
_ATTRIBUTE_PATTERNS = {
    "storage": re.compile(r"\b(\d+)\s?(gb|tb)\b"),
    "volume":  re.compile(r"\b(\d+(?:\.\d+)?)\s?(ml|l|litre|liter|ltr)\b"),
    "size_mm": re.compile(r"\b(\d+)\s?mm\b"),
}
# Words that don't identify a product: filler, marketing, colours
_NOISE_WORDS = {
    "a", "an", "and", "the", "for", "with", "of", "in", "by", "to", "on", "at", "new", "latest",
    "buy", "online", "shop", "now", "best", "men", "mens", "women", "womens", "unisex", "size",
    "black", "white", "blue", "red", "green", "grey", "gray", "silver", "gold", "pink", "purple",
    "yellow", "beige", "midnight", "graphite", "titanium", "mint", "violet", "negro",
}
_SAME_PRODUCT_MIN_OVERLAP = 0.5
_LURE_RATIO = 0.4      # below this × the search-wide median, a price is suspicious regardless


def _normalise_title(title: str) -> str:
    title = (title or "").lower().replace("pre owned", "pre-owned").replace("open box", "open-box")
    return re.sub(r"[^\w\s\-./]", " ", title.replace("_", " "))


_UNIT_ALIASES = {"litre": "l", "liter": "l", "ltr": "l"}


def _attribute_value(match_text: str) -> str:
    """"30 ml" → "30ml", "3 litre" → "3l", "128GB" → "128gb"."""
    number = re.match(r"[\d.]+", match_text).group()
    unit = re.sub(r"[\d.\s]", "", match_text)
    return number + _UNIT_ALIASES.get(unit, unit)


def title_profile(title: str) -> dict:
    """What identifies the product in a listing title."""
    text = _normalise_title(title)
    attributes = {name: {_attribute_value(m.group(0)) for m in pat.finditer(text)}
                  for name, pat in _ATTRIBUTE_PATTERNS.items()}
    for pat in _ATTRIBUTE_PATTERNS.values():
        text = pat.sub(" ", text)
    tokens = {t.strip("-./") for t in text.split()} - {""}
    # Model codes: letters+digits ("rb3025", "1000xm5", "fb3383ax") or LEGO-style set numbers,
    # matched on the parts of hyphenated tokens so "mic-wh-1000xm5" still matches "wh-1000xm5"
    parts = {p for t in tokens for p in re.split(r"[-/.]", t) if p}
    codes = {p for p in parts
             if (re.search(r"[a-z]", p) and re.search(r"\d", p) and len(p) >= 4) or re.fullmatch(r"\d{5}", p)}
    return {
        "used":       bool(tokens & _USED_WORDS),
        "tiers":      frozenset(tokens & _TIER_WORDS),
        "attributes": attributes,
        "codes":      codes,
        # Plain numbers are usually model/generation numbers too: "Airdopes 141",
        # "Series 11", "iPhone 15", shade "125"
        "numbers":    {p for p in parts if p.isdigit()} - codes,
        "words":      tokens - _NOISE_WORDS - _USED_WORDS,
    }


def same_product(a: dict, b: dict) -> bool:
    """Whether two title_profile()s describe the same product and variant."""
    if a["used"] != b["used"] or a["tiers"] != b["tiers"]:
        return False
    for name in _ATTRIBUTE_PATTERNS:
        if a["attributes"][name] and b["attributes"][name] and not a["attributes"][name] & b["attributes"][name]:
            return False
    if a["codes"] and b["codes"]:
        return bool(a["codes"] & b["codes"])
    # Model numbers must agree: "Airdopes 141" ≠ "Airdopes 131/138", "Series 11" ≠ "Series 9";
    # one title may carry extra numbers ("iPhone 15" vs "iPhone 15 2023")
    if a["numbers"] and b["numbers"] and not (a["numbers"] <= b["numbers"] or b["numbers"] <= a["numbers"]):
        return False
    if not a["words"] and not b["words"]:
        return True                        # no titles to go on: same search, assume comparable
    overlap = len(a["words"] & b["words"]) / max(min(len(a["words"]), len(b["words"])), 1)
    return overlap >= _SAME_PRODUCT_MIN_OVERLAP


def estimate_market_references(
    raw_results: list[dict],
    reference_prices: Optional[list[Optional[float]]] = None,
) -> list[tuple[Optional[float], str]]:
    """
    (market price, basis) for each listing — the price its normalized_price is
    measured against, which the agent uses both to value a deal and to spot a
    price that's too good to be true. In order of preference:

      0. A reference price: the median price of this exact product across
         other stores on Google's product page (reference_prices.py), when
         the caller fetched one — the only source that actually knows the
         product's usual price.
      1. A trusted seller's own list price ("was ₹79,900").
      2. The median price of the *same product* from trusted sellers.
      3. The median price of the same product from any seller that isn't
         hard-blocked (trust ≥ 0.3).
      4. Nothing comparable → no reference: the listing is priced "at market"
         (ratio 1.0, i.e. no evidence either way) — unless it's below
         _LURE_RATIO × the search-wide trusted median, which is suspicious
         even against loosely related products.

    Scam lures never set the reference for others: step 2 uses trusted
    sellers only, and step 3 excludes hard-blocked ones.
    """
    prices   = [parse_price(item.get("extracted_price") or item.get("price", 0)) for item in raw_results]
    trusts   = [compute_domain_trust(resolve_seller_domain(item)) for item in raw_results]
    profiles = [title_profile(item.get("title") or item.get("product_name") or "") for item in raw_results]

    trusted_all = [p for p, t in zip(prices, trusts) if p > 0 and t >= MARKET_REFERENCE_MIN_TRUST]
    everyone    = [p for p in prices if p > 0]
    search_wide = float(np.median(trusted_all if len(trusted_all) >= 2 else everyone)) if everyone else None

    refs = []
    for i, (item, price) in enumerate(zip(raw_results, prices)):
        if reference_prices and reference_prices[i]:
            refs.append((reference_prices[i], "other stores, same product"))
            continue
        list_price = parse_old_price(item)
        if trusts[i] >= MARKET_REFERENCE_MIN_TRUST and list_price > price > 0:
            refs.append((list_price, "own list price"))
            continue

        same = [j for j in range(len(raw_results))
                if j != i and prices[j] > 0 and same_product(profiles[i], profiles[j])]
        trusted_same = [prices[j] for j in same if trusts[j] >= MARKET_REFERENCE_MIN_TRUST]
        open_same    = [prices[j] for j in same if trusts[j] >= SCAM_TRUST_THRESHOLD]
        if trusted_same:
            refs.append((float(np.median(trusted_same)), f"{len(trusted_same)} similar trusted listing(s)"))
        elif open_same:
            refs.append((float(np.median(open_same)), f"{len(open_same)} similar listing(s)"))
        elif price > 0 and search_wide and price < _LURE_RATIO * search_wide:
            refs.append((search_wide, "far below everything in the search"))
        else:
            refs.append((price if price > 0 else None, "no comparable listing"))
    return refs


def estimate_market_prices(raw_results: list[dict], reference_prices=None) -> list[Optional[float]]:
    """Market price per listing — see estimate_market_references()."""
    return [ref for ref, _ in estimate_market_references(raw_results, reference_prices)]
