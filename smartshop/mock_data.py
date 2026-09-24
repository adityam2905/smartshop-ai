"""
Built-in demo listings, used when there's no SerpAPI key or a live search
fails. Each set mixes real retailers with scam and look-alike sites.
"""

from typing import Optional

MOCK_CURRENCY = "$"          # demo listings are priced in USD

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
