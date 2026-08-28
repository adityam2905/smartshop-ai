"""
Phase 1: Synthetic Data Generator
Generates 5,000 product listings with scam/legit labels.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Toys", "Beauty", "Automotive"]

LEGIT_DOMAINS = [
    "amazon.com", "walmart.com", "bestbuy.com", "target.com", "ebay.com",
    "costco.com", "newegg.com", "bhphotovideo.com", "adorama.com", "wayfair.com",
    "homedepot.com", "macys.com", "nordstrom.com", "zappos.com", "chewy.com",
]

SCAM_DOMAINS = [
    "ultra-deals99.net", "cheapbuy-store.xyz", "discount-mega.ru", "bestprice-deals.tk",
    "shop-fast-now.biz", "topdeal-online.cc", "savebig-store.pw", "bargain-hunt.gq",
    "flashsale-today.ml", "pricedown-shop.cf", "deal-xpress.top", "buylow-online.icu",
]

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


def generate_legit_listing(category: str) -> dict:
    product = random.choice(PRODUCT_TEMPLATES[category])
    low, high = MARKET_PRICES[category]
    market_avg = round(random.uniform(low, high), 2)

    # Legit sites offer modest discounts (5%–40%)
    discount_pct = round(random.uniform(0.05, 0.40), 4)
    price = round(market_avg * (1 - discount_pct), 2)

    domain = random.choice(LEGIT_DOMAINS)
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

    # Scam sites advertise huge discounts (60%–95%) to lure users
    discount_pct = round(random.uniform(0.60, 0.95), 4)
    price = round(market_avg * (1 - discount_pct), 2)
    price = max(price, 0.99)  # floor

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
    Rule-based trust score.
    Legit sites: 0.55 – 1.0 (with some noise)
    Scam sites:  0.0  – 0.28 (with some noise, always below 0.3 threshold)
    """
    if is_scam:
        return round(np.clip(np.random.beta(1.5, 8), 0.0, 0.28), 4)
    else:
        # Top-tier domains score higher
        tier1 = {"amazon.com", "walmart.com", "bestbuy.com", "target.com"}
        base = 0.90 if domain in tier1 else 0.70
        noise = np.random.normal(0, 0.05)
        return round(float(np.clip(base + noise, 0.55, 1.0)), 4)


def compute_user_preference_score(category: str) -> float:
    """Simulate a per-category user preference (random per run, fixed per category)."""
    # Fixed preferences seeded for reproducibility
    prefs = {
        "Electronics": 0.85, "Clothing": 0.60, "Home & Garden": 0.45,
        "Sports": 0.70, "Books": 0.90, "Toys": 0.35,
        "Beauty": 0.50, "Automotive": 0.40,
    }
    base = prefs.get(category, 0.5)
    noise = np.random.normal(0, 0.08)
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
