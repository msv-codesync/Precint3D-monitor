"""Refuse anything that is not a normal http(s) printer URL."""

from __future__ import annotations

from ipaddress import ip_address, ip_network
from urllib.parse import urlparse

# Cloud instance metadata — not a printer. LAN printers (octopi.local, 192.168.x) stay allowed.
_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",
        "169.254.169.253",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)
_METADATA_NETS = (
    ip_network("169.254.169.254/32"),
    ip_network("169.254.169.253/32"),
    ip_network("fd00:ec2::254/128"),
)


def _reject_metadata_host(host: str) -> None:
    name = (host or "").lower().rstrip(".")
    if name in _METADATA_HOSTS:
        raise ValueError("That host is not a printer.")
    try:
        addr = ip_address(name)
    except ValueError:
        return
    for net in _METADATA_NETS:
        if addr in net:
            raise ValueError("That host is not a printer.")


def http_url(url: str) -> str:
    text = (url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL must be http or https with a host.")
    if parsed.username or parsed.password:
        raise ValueError("Do not put credentials in the URL.")
    _reject_metadata_host(parsed.hostname or "")
    return text.rstrip("/")
