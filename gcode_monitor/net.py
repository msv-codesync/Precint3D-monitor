"""Refuse anything that is not a normal http(s) printer URL."""

from __future__ import annotations

from urllib.parse import urlparse


def http_url(url: str) -> str:
    text = (url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL must be http or https with a host.")
    if parsed.username or parsed.password:
        raise ValueError("Do not put credentials in the URL.")
    return text.rstrip("/")
