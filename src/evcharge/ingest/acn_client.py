"""Client for the ACN-Data REST API (https://ev.caltech.edu/api/v1/).

Auth is HTTP Basic with the API token as username and an empty password, matching
the reference acnportal client. Responses are python-eve paginated (`_items` / `_links`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from evcharge.config import SITES, Settings

logger = logging.getLogger(__name__)

PAGE_SIZE = 100
TIMEOUT = (10, 60)  # (connect, read) seconds


class ACNDataError(RuntimeError):
    """Raised when the ACN-Data API cannot be reached or rejects a request."""


class ACNDataClient:
    def __init__(self, settings: Settings, page_size: int = PAGE_SIZE) -> None:
        self._base_url = settings.acn_api_url
        self._page_size = page_size

        self._session = requests.Session()
        self._session.auth = (settings.require_token().reveal(), "")
        self._session.headers["Accept"] = "application/json"

        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    def __enter__(self) -> ACNDataClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    @staticmethod
    def _validate_site(site: str) -> str:
        if site not in SITES:
            raise ValueError(f"Unknown site {site!r}. Expected one of {SITES}.")
        return site

    def _get(self, path: str) -> dict[str, Any]:
        url = path if path.startswith("https://") else self._base_url + path
        try:
            response = self._session.get(url, timeout=TIMEOUT)
        except requests.RequestException as exc:
            raise ACNDataError(f"Request to ACN-Data failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise ACNDataError(
                "ACN-Data rejected the API token (HTTP "
                f"{response.status_code}). Check ACN_API_TOKEN in your .env."
            )
        if not response.ok:
            raise ACNDataError(
                f"ACN-Data returned HTTP {response.status_code} for {path}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ACNDataError(f"ACN-Data returned non-JSON for {path}") from exc

    def check_access(self) -> int:
        """Fetch a single session to prove the token works. Returns sessions seen."""
        payload = self._get(f"sessions/{SITES[0]}?max_results=1")
        return len(payload.get("_items", []))

    def iter_sessions(
        self,
        site: str,
        where: str | None = None,
        sort: str | None = "connectionTime",
    ) -> Iterator[dict[str, Any]]:
        """Yield every session for a site, following pagination links."""
        self._validate_site(site)

        params = [f"max_results={self._page_size}"]
        if where:
            params.append(f"where={where}")
        if sort:
            params.append(f"sort={sort}")
        path = f"sessions/{site}?" + "&".join(params)

        page = 0
        while True:
            payload = self._get(path)
            items = payload.get("_items", [])
            page += 1
            logger.debug("site=%s page=%d items=%d", site, page, len(items))
            yield from items

            next_link = payload.get("_links", {}).get("next", {}).get("href")
            if not next_link:
                return
            path = next_link.lstrip("/")
