import sys
from urllib.parse import urljoin
from datetime import timedelta
from typing import Callable

from bs4 import BeautifulSoup
from requests import Response
from requests_cache import CachedSession  # type: ignore


def _absolutize_links(resp: Response, *args, **kwargs) -> Response:
    """Convert relative <a href> links to absolute, based on the final URL."""
    # Only process HTML-ish responses with content
    ctype = resp.headers.get("Content-Type", "")
    if not resp.content or "html" not in ctype.lower():
        return resp

    soup = BeautifulSoup(resp.content, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "://" not in href:
            link["href"] = urljoin(resp.url, href)

    # Replace response body; keep bytes and respect encoding
    html_out: bytes = soup.prettify("utf-8")  # returns bytes when encoding given
    resp._content = html_out  # requests caches/serves from this bytes payload
    # Optional: keep text properties consistent
    resp.encoding = resp.encoding or "utf-8"
    # Do not touch Content-Length; requests will not recalc it for cached bodies,
    # but servers rarely rely on it client-side. Safe to leave as-is.
    return resp


class AFLTablesSession(CachedSession):
    """A CachedSession that rewrites AFL Tables links to absolute paths."""

    def __init__(self, *args, **kwargs):
        # Inject our response hook while allowing callers to pass their own
        hook: Callable[[Response], Response] = _absolutize_links
        user_hook = kwargs.pop("response_hook", None)

        if user_hook is None:
            kwargs["response_hook"] = hook
        else:
            # Chain both if the caller provided one
            def chained(resp: Response, *a, **k):
                return hook(user_hook(resp, *a, **k), *a, **k)

            kwargs["response_hook"] = chained

        super().__init__(*args, **kwargs)

    def get(self, url, force_live: bool = False, **kwargs) -> Response:
        # If `force_live` is provided, temporarily bypass the cache
        if force_live:
            with self.cache_disabled():
                return super().get(url, **kwargs)
        return super().get(url, **kwargs)


session = AFLTablesSession(
    "test_db" if "pytest" in sys.modules else "pyAFL_html_cache",
    backend="filesystem",
    use_cache_dir=True,  # Save files in the default user cache dir
    cache_control=False,  # Ignore Cache-Control; use our expire_after
    expire_after=timedelta(days=365),  # Expire responses after 365 days
    allowable_codes=[200],  # Cache only successful responses
    allowable_methods=["GET"],  # Cache GET requests
    stale_if_error=False,  # Don't serve stale cache on errors
)
