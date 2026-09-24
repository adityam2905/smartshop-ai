"""
Escaping of third-party listing data before it's rendered as HTML.

Needs streamlit + torch (importing app.py loads the model stack), so it's
skipped under the lean requirements-test.txt used in CI.
"""

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("stable_baselines3")

from app import safe_text, safe_url  # noqa: E402


def test_safe_text_escapes_markup():
    assert safe_text('<img src=x onerror="alert(1)">') == "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;"


@pytest.mark.parametrize("url", [
    "javascript:alert(1)",
    "JavaScript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "",
    "not a url",
])
def test_safe_url_rejects_non_http_links(url):
    assert safe_url(url) == ""


def test_safe_url_keeps_http_links_and_escapes_quotes():
    assert safe_url("https://www.amazon.com/dp/1") == "https://www.amazon.com/dp/1"
    assert safe_url('https://x.com/"onmouseover="alert(1)') == "https://x.com/&quot;onmouseover=&quot;alert(1)"
