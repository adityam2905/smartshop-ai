"""
Phase 1: Synthetic Data Generator
Generates 5,000 product listings with scam/legit labels.

Each listing has two independent price signals, as in live results:
  * normalized_price     — the real price ÷ the market price (< 1 is cheap,
                           > 1 is overpriced). This is what a deal is worth.
  * discount_percentage  — the discount the seller *claims*, which can be
                           inflated (legit "MRP" markdowns, scam lures).

and is built so that no single feature decides the label:
  * Legit sellers range from big retailers (trust ~0.9) to small shops
    (trust 0.40–0.70), and list clearance deals, normal prices and
    overpriced items.
  * Scams are either obviously untrusted (trust < 0.28), or polished shops
    with middling trust (0.30–0.60) whose price is far below market — the
    "too good to be true" pattern the trust < 0.3 hard rule can't catch.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Toys", "Beauty", "Automotive"]

TIER1_DOMAINS = [
    "amazon.com", "walmart.com", "bestbuy.com", "target.com", "amazon.in", "flipkart.com",
]
LEGIT_DOMAINS = TIER1_DOMAINS + [
    "ebay.com", "costco.com", "newegg.com", "bhphotovideo.com", "adorama.com",
    "wayfair.com", "homedepot.com", "macys.com", "nordstrom.com", "zappos.com",
    "chewy.com", "croma.com",
]

# Small but legitimate independent shops. scraper.compute_domain_trust() gives
# unknown-but-clean domains a middling score (~0.45–0.65), so the training
# data has to contain legit listings in that trust range too — otherwise the
# model never sees the trust values it's actually fed for most live results.
SMALL_LEGIT_DOMAINS = [
    "harbor-audio.com", "northside-outfitters.com", "kettle-and-co.com",
    "pageturner-books.com", "brightbrick-toys.com", "motorline-parts.com",
]
SMALL_LEGIT_RATIO = 0.15

# Obvious scam sites: scam TLDs, spammy names → trust < 0.28
SCAM_DOMAINS = [
    "ultra-deals99.net", "cheapbuy-store.xyz", "discount-mega.ru", "bestprice-deals.tk",
    "shop-fast-now.biz", "topdeal-online.cc", "savebig-store.pw", "bargain-hunt.gq",
    "flashsale-today.ml", "pricedown-shop.cf", "deal-xpress.top", "buylow-online.icu",
]
# Polished scam shops on clean-looking .com domains → trust 0.30–0.60, the
# same range as small legit shops. Only the implausibly low price gives them
# away, so the agent (not the trust < 0.3 hard rule) has to catch them.
POLISHED_SCAM_DOMAINS = [
    "urbanstyle-outlet.com", "techhaven-store.com", "luxe-home-direct.com",
    "gadgetplanet-shop.com", "primegear-store.com", "novahome-goods.com",
]
POLISHED_SCAM_RATIO = 0.45

# Real price ÷ market price, per seller type: (probability, low, high)
BIG_LEGIT_PRICE_RATIO   = [(0.20, 0.20, 0.60), (0.60, 0.60, 1.00), (0.20, 1.00, 1.50)]  # clearance / normal / overpriced
SMALL_LEGIT_PRICE_RATIO = [(0.75, 0.65, 1.00), (0.25, 1.00, 1.50)]                     # small shops rarely deep-discount
SCAM_PRICE_RATIO        = [(0.92, 0.05, 0.50), (0.08, 0.50, 0.70)]                     # lures, a few "believable"

# Claimed discount must NOT separate scam from legit on its own. Earlier
# versions used non-overlapping ranges (legit 5–40%, scam 60–95%), so the
# agent learned "small discount = safe" instead of "trusted site = safe".
SCAM_BELIEVABLE_DISCOUNT_RATIO = 0.35   # scams claiming a modest 10–50% off

PRODUCT_TEMPLATES = {
    "Electronics": ["Sony Headphones WH-1000XM5", "Samsung 4K TV 55\"", "Apple AirPods Pro",
                    "Logitech MX Master Mouse", "Dell XPS 15 Laptop", "iPad Pro 12.9\"",
                    "Canon EOS R6 Camera", "Bose SoundBar 700"],
    "Clothing":    ["Nike Air Max Sneakers", "Levi's 501 Jeans", "Patagonia Fleece Jacket",
                    "Ray-Ban Aviator Sunglasses", "The North Face Parka", "Adidas Ultraboost"],
    "Home & Garden": ["Dyson V15 Vacuum", "Instant Pot Duo 7-in-1", "Roomba i7+ Robot Vacuum",
                      "KitchenAid Stand Mixer", "Nespresso Vertuo Coffee Maker"],
    "Sports":      ["Peloton Bike+", "Garmin Forerunner 945", "TRX Suspension Trainer",
                    "Callaway Golf Driver", "Wilson Tennis Racket Pro"],
    "Books":       ["Python Crash Course 3rd Ed", "Atomic Habits", "The Pragmatic Programmer",
                    "Clean Code", "Designing Data-Intensive Applications"],
    "Toys":        ["LEGO Technic Bugatti", "Nintendo Switch OLED", "Barbie Dreamhouse",
                    "Hot Wheels Ultimate Garage", "Magna-Tiles 100 Piece Set"],
    "Beauty":      ["Dyson Airwrap Styler", "La Mer Moisturizing Cream", "Fenty Beauty Foundation",
                    "NARS Blush Orgasm", "Olaplex Hair Perfector"],
    "Automotive":  ["Garmin DriveSmart 65 GPS", "Thinkware U1000 Dash Cam",
                    "NOCO Genius5 Battery Charger", "Michelin X-Tour Tires"],
}

MARKET_PRICES = {
    "Electronics": (50, 3000), "Clothing": (20, 500), "Home & Garden": (30, 800),
    "Sports": (15, 2500), "Books": (10, 60), "Toys": (15, 400),
    "Beauty": (20, 600), "Automotive": (25, 1000),
}


def _sample_ratio(mixture: list[tuple[float, float, float]]) -> float:
    """Draw from a mixture of uniform ranges [(probability, low, high), …]."""
    r = random.random()
    for prob, low, high in mixture:
        if r < prob:
            return random.uniform(low, high)
        r -= prob
    return random.uniform(*mixture[-1][1:])


def generate_legit_listing(category: str) -> dict:
    product = random.choice(PRODUCT_TEMPLATES[category])
    low, high = MARKET_PRICES[category]
    market_avg = round(random.uniform(low, high), 2)

    if random.random() < SMALL_LEGIT_RATIO:
        domain = random.choice(SMALL_LEGIT_DOMAINS)
        price_ratio = _sample_ratio(SMALL_LEGIT_PRICE_RATIO)
    else:
        domain = random.choice(LEGIT_DOMAINS)
        price_ratio = _sample_ratio(BIG_LEGIT_PRICE_RATIO)
    price = round(max(market_avg * price_ratio, 0.99), 2)

    # Legit sellers' claimed discount tracks the real one (off MRP, with a
    # little noise); overpriced listings may still advertise a small markdown.
    if price_ratio < 1.0:
        discount_pct = float(np.clip(1.0 - price_ratio + np.random.normal(0, 0.04), 0.0, 0.95))
    else:
        discount_pct = random.uniform(0.0, 0.15)
    discount_pct = round(discount_pct, 4)
    site_url = f"https://www.{domain}/dp/{random.randint(1000000, 9999999)}"

    return {
        "product_name": product,
        "category": category,
        "price": price,
        "market_avg_price": market_avg,
        "discount_percentage": discount_pct,
        "site_url": site_url,
        "domain": domain,
        "is_scam": False,
    }


def generate_scam_listing(category: str) -> dict:
    product = random.choice(PRODUCT_TEMPLATES[category])
    low, high = MARKET_PRICES[category]
    market_avg = round(random.uniform(low, high), 2)

    # The real price is a lure, far below market; the *claimed* discount is
    # usually huge but sometimes modest, to look believable.
    price = round(max(market_avg * _sample_ratio(SCAM_PRICE_RATIO), 0.99), 2)
    if random.random() < SCAM_BELIEVABLE_DISCOUNT_RATIO:
        discount_pct = round(random.uniform(0.10, 0.50), 4)
    else:
        discount_pct = round(random.uniform(0.50, 0.95), 4)

    if random.random() < POLISHED_SCAM_RATIO:
        domain = random.choice(POLISHED_SCAM_DOMAINS)
    else:
        domain = random.choice(SCAM_DOMAINS)
    site_url = f"https://{domain}/product/{random.randint(100, 99999)}"

    return {
        "product_name": product,
        "category": category,
        "price": price,
        "market_avg_price": market_avg,
        "discount_percentage": discount_pct,
        "site_url": site_url,
        "domain": domain,
        "is_scam": True,
    }


def compute_site_trust_score(domain: str, is_scam: bool) -> float:
    """
    Rule-based trust score, mirroring what scraper.compute_domain_trust()
    produces for each kind of seller:
      Big retailers:       0.80 – 1.00
      Other known shops:   0.60 – 0.90
      Small legit shops:   0.40 – 0.70
      Polished scam shops: 0.30 – 0.60   (overlaps small legit shops)
      Obvious scam sites:  0.00 – 0.28
    """
    if domain in POLISHED_SCAM_DOMAINS:
        return round(random.uniform(0.30, 0.60), 4)
    if is_scam:
        return round(float(np.clip(np.random.beta(1.5, 8), 0.0, 0.28)), 4)
    if domain in SMALL_LEGIT_DOMAINS:
        return round(float(np.clip(np.random.normal(0.55, 0.06), 0.40, 0.70)), 4)
    if domain in TIER1_DOMAINS:
        return round(float(np.clip(np.random.normal(0.92, 0.04), 0.80, 1.00)), 4)
    return round(float(np.clip(np.random.normal(0.75, 0.06), 0.60, 0.90)), 4)


def compute_user_preference_score(category: str) -> float:
    """Simulate a per-category user preference (random per run, fixed per category)."""
    # Fixed preferences seeded for reproducibility
    prefs = {
        "Electronics": 0.85, "Clothing": 0.60, "Home & Garden": 0.45,
        "Sports": 0.70, "Books": 0.90, "Toys": 0.35,
        "Beauty": 0.50, "Automotive": 0.40,
    }
    base = prefs.get(category, 0.5)
    # Wide spread so the agent sees the whole [0, 1] range — the preference
    # now affects the reward, and live values drift with every Like/Dislike.
    noise = np.random.normal(0, 0.15)
    return round(float(np.clip(base + noise, 0.0, 1.0)), 4)


def generate_dataset(n: int = 5000, scam_ratio: float = 0.25) -> pd.DataFrame:
    records = []
    n_scam = int(n * scam_ratio)
    n_legit = n - n_scam

    for _ in range(n_legit):
        cat = random.choice(CATEGORIES)
        rec = generate_legit_listing(cat)
        rec["site_trust_score"] = compute_site_trust_score(rec["domain"], False)
        rec["user_preference_score"] = compute_user_preference_score(cat)
        records.append(rec)

    for _ in range(n_scam):
        cat = random.choice(CATEGORIES)
        rec = generate_scam_listing(cat)
        rec["site_trust_score"] = compute_site_trust_score(rec["domain"], True)
        rec["user_preference_score"] = compute_user_preference_score(cat)
        records.append(rec)

    df = pd.DataFrame(records)

    # Derived feature: normalized_price (price / market_avg_price), clipped to [0, 2]
    df["normalized_price"] = (df["price"] / df["market_avg_price"]).clip(0.0, 2.0).round(4)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Reorder columns
    cols = [
        "product_name", "category", "price", "market_avg_price",
        "normalized_price", "discount_percentage",
        "site_trust_score", "user_preference_score",
        "site_url", "domain", "is_scam",
    ]
    return df[cols]


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        # Console prints below use non-ASCII characters (→); on Windows the
        # default console codepage (cp1252) can't encode them and this
        # script would crash with a UnicodeEncodeError otherwise.
        sys.stdout.reconfigure(encoding="utf-8")

    print("Generating synthetic dataset...")
    df = generate_dataset(n=5000, scam_ratio=0.25)
    df.to_csv("product_listings.csv", index=False)

    print(f"Dataset saved → product_listings.csv")
    print(f"Total rows    : {len(df)}")
    print(f"Scam listings : {df['is_scam'].sum()} ({df['is_scam'].mean()*100:.1f}%)")
    print(f"Legit listings: {(~df['is_scam']).sum()}")
    print("\nSample rows:")
    print(df[["product_name", "normalized_price", "discount_percentage",
              "site_trust_score", "user_preference_score", "is_scam"]].head(10).to_string())
    print("\nFeature statistics:")
    print(df[["normalized_price", "discount_percentage",
              "site_trust_score", "user_preference_score"]].describe().round(3))
