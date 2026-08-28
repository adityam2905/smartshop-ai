import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tiny_csv(tmp_path) -> str:
    """
    A tiny, hand-crafted product-listing CSV with exactly one scam row and
    one legit row, covering every branch of ShoppingEnv's reward function.
    Kept deliberately small so env tests are fast and easy to reason about.
    """
    rows = [
        dict(
            product_name="Scam Item", category="Electronics",
            price=10.0, market_avg_price=100.0,
            normalized_price=0.1, discount_percentage=0.90,
            site_trust_score=0.05, user_preference_score=0.5,
            site_url="https://scam.xyz/1", domain="scam.xyz", is_scam=True,
        ),
        dict(
            product_name="Legit Item", category="Books",
            price=40.0, market_avg_price=50.0,
            normalized_price=0.8, discount_percentage=0.20,
            site_trust_score=0.9, user_preference_score=0.5,
            site_url="https://amazon.com/2", domain="amazon.com", is_scam=False,
        ),
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "tiny_listings.csv"
    df.to_csv(path, index=False)
    return str(path)
