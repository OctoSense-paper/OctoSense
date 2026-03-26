"""HTTP storage source helper."""

from __future__ import annotations

import urllib.request
from urllib.parse import urlparse
from collections.abc import Callable
from pathlib import Path


def resolve_http_download_size(url: str, response: object) -> int | None:
    headers = getattr(response, "headers", {})
    get_header = getattr(headers, "get", None)
    if callable(get_header):
        raw_length = get_header("Content-Length")
        if raw_length:
            try:
                return int(raw_length)
            except ValueError:
                pass

    parsed = urllib.request.url2pathname(urlparse(url).path) if url.startswith("file:") else None
    if parsed:
        try:
            return Path(parsed).stat().st_size
        except OSError:
            return None
    return None


def download_from_http(
    url: str,
    target: str | Path,
    *,
    user_agent: str = "octosense-datasets/1.0",
    chunk_size: int = 1024 * 1024,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> Path:
    destination = Path(target)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        total_bytes = resolve_http_download_size(url, response)
        downloaded_bytes = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded_bytes += len(chunk)
            if progress_callback is not None:
                progress_callback(downloaded_bytes, total_bytes)
    return destination


__all__ = ["download_from_http", "resolve_http_download_size"]
