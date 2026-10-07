"""Verify ACN-Data credentials are configured correctly.

Usage:  python -m evcharge.ingest.check_access
"""

from __future__ import annotations

import sys

from evcharge.config import ConfigError, get_settings
from evcharge.ingest.acn_client import ACNDataClient, ACNDataError


def main() -> int:
    try:
        settings = get_settings()
        secret = settings.require_token()
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    token = secret.reveal()
    print(f"[ok]   ACN_API_TOKEN loaded (length {len(token)}, ends ...{token[-4:]})")
    print(f"[ok]   ACN_API_URL   {settings.acn_api_url}")
    print(f"[ok]   DATA_DIR      {settings.data_dir}")

    try:
        with ACNDataClient(settings) as client:
            seen = client.check_access()
    except ACNDataError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if seen == 0:
        print("[WARN] Authenticated, but the API returned no sessions.")
        return 1

    print("[ok]   Authenticated against ACN-Data and read a live session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
