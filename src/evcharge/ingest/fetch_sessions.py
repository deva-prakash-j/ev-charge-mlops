"""Download ACN-Data charging sessions to newline-delimited JSON.

Usage:
    python -m evcharge.ingest.fetch_sessions                 # all sites
    python -m evcharge.ingest.fetch_sessions --site caltech  # one site

Raw payloads are written verbatim so the download stays an auditable record of
what the API returned; all cleaning happens downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from evcharge.config import SITES, ConfigError, Settings, get_settings
from evcharge.ingest.acn_client import ACNDataClient, ACNDataError

logger = logging.getLogger(__name__)


def fetch_site(client: ACNDataClient, site: str, out_dir: Path) -> dict[str, object]:
    """Stream one site's sessions to disk; return a manifest entry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"sessions_{site}.jsonl"
    tmp = target.with_suffix(".jsonl.partial")

    digest = hashlib.sha256()
    count = 0

    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for session in client.iter_sessions(site):
            line = json.dumps(session, separators=(",", ":"), sort_keys=True)
            fh.write(line + "\n")
            digest.update(line.encode("utf-8"))
            count += 1
            if count % 1000 == 0:
                logger.info("%s: %d sessions", site, count)

    if count == 0:
        tmp.unlink(missing_ok=True)
        raise ACNDataError(f"No sessions returned for site {site!r}.")

    tmp.replace(target)
    logger.info("%s: wrote %d sessions -> %s", site, count, target.name)

    return {
        "site": site,
        "file": target.name,
        "sessions": count,
        "sha256": digest.hexdigest(),
        "bytes": target.stat().st_size,
    }


def write_manifest(settings: Settings, entries: list[dict[str, object]]) -> Path:
    """Record counts and hashes so the extract is citable and verifiable."""
    manifest = {
        "source": "ACN-Data",
        "api_url": settings.acn_api_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "citation": (
            "Zachary J. Lee, Tongxin Li and Steven H. Low. 'ACN-Data: Analysis and "
            "Applications of an Open EV Charging Dataset.' Proceedings of the Tenth "
            "ACM International Conference on Future Energy Systems (e-Energy '19), 2019."
        ),
        "total_sessions": sum(int(e["sessions"]) for e in entries),
        "sites": entries,
    }
    path = settings.raw_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ACN-Data sessions.")
    parser.add_argument(
        "--site",
        action="append",
        choices=SITES,
        help="Site to download; repeatable. Defaults to all sites.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    sites = args.site or list(SITES)
    entries: list[dict[str, object]] = []

    try:
        with ACNDataClient(settings) as client:
            for site in sites:
                entries.append(fetch_site(client, site, settings.raw_dir))
    except ACNDataError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    manifest_path = write_manifest(settings, entries)
    total = sum(int(e["sessions"]) for e in entries)
    logger.info("Done. %d sessions across %d site(s).", total, len(entries))
    logger.info("Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
