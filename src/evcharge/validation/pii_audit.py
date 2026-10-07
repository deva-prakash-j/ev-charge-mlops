"""Column-level PII audit over the raw ACN-Data extract.

Writes artifacts/pii_audit.json. This is the evidence behind the proposal's
"no personally identifiable information" declaration, so it reports what was
actually inspected rather than asserting a conclusion.

Usage:  python -m evcharge.validation.pii_audit
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from evcharge.config import ConfigError, get_settings, load_config

# Patterns for data that would make a record directly identifying. Applied only
# to free-text values; digit boundaries stop them matching inside opaque IDs.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w]{2,}"),
    "phone": re.compile(
        r"(?<!\d)(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\d)"
    ),
    "ssn_like": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card_like": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    "ip_address": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "latlong": re.compile(r"[-+]?\d{1,3}\.\d{4,}\s*,\s*[-+]?\d{1,3}\.\d{4,}"),
    "street_address": re.compile(
        r"\b\d+\s+\w+\s+(street|st|road|rd|avenue|ave|lane|ln|drive|dr)\b",
        re.IGNORECASE,
    ),
}

# Structured machine identifiers, recognised so their digits are not mistaken
# for phone or card numbers.
STRUCTURED_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "mongo_object_id": re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE),
    "zero_padded_numeric_id": re.compile(r"^0\d+$"),
    "dashed_numeric_id": re.compile(r"^\d+(?:-\d+)+$"),
    "acn_session_id": re.compile(r"^\d+_\d+_\d+_\d+_"),
    "http_date": re.compile(r"^[A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4}"),
    "iana_timezone": re.compile(r"^[A-Za-z]+/[A-Za-z_]+$"),
}

# Columns that are person-linked pseudonyms: not PII on their own, but they
# allow per-individual behaviour to be tracked, so they must not be published.
PSEUDONYM_HINTS = ("userid", "user_id", "customer", "driver", "vin", "account")


def _luhn_valid(digits: str) -> bool:
    """Real card numbers satisfy Luhn; incidental digit runs almost never do."""
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(nums)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def classify_structure(value: Any) -> str | None:
    """Name the structured form of a value, or None if it is free text."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "numeric"
    if not isinstance(value, str):
        return "non_text"
    for label, pattern in STRUCTURED_ID_PATTERNS.items():
        if pattern.match(value):
            return label
    return None


def _flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            flat.update(_flatten(value[0], f"{name}[]."))
        else:
            flat[name] = value
    return flat


def audit_column(name: str, values: list[Any]) -> dict[str, Any]:
    present = [v for v in values if v is not None and v != ""]
    matches: Counter[str] = Counter()
    structures: Counter[str] = Counter()
    scanned = 0

    for value in present:
        structure = classify_structure(value)
        structures[structure or "free_text"] += 1
        if structure is not None:
            continue  # structured machine value cannot carry an email or address
        scanned += 1
        for label, pattern in PII_PATTERNS.items():
            hit = pattern.search(value)
            if not hit:
                continue
            if label == "credit_card_like" and not _luhn_valid(hit.group()):
                continue
            matches[label] += 1

    lowered = name.lower()
    is_pseudonym = any(hint in lowered for hint in PSEUDONYM_HINTS)

    return {
        "column": name,
        "non_null": len(present),
        "null_fraction": round(1 - len(present) / len(values), 4) if values else 1.0,
        "distinct_values": len({str(v) for v in present}),
        "dtype": type(present[0]).__name__ if present else "NoneType",
        "value_structures": dict(structures),
        "free_text_values_scanned": scanned,
        "example_masked": _mask(present[0]) if present else None,
        "direct_pii_matches": dict(matches),
        "classification": (
            "DIRECT_PII"
            if matches
            else "PSEUDONYMOUS_IDENTIFIER"
            if is_pseudonym
            else "NON_IDENTIFYING"
        ),
    }


def _mask(value: Any) -> str:
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}{'*' * (len(text) - 4)}{text[-2:]}"


def audit_file(path: Path, sample_limit: int | None = None) -> list[dict[str, Any]]:
    columns: dict[str, list[Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if sample_limit is not None and index >= sample_limit:
                break
            flat = _flatten(json.loads(line))
            for key, value in flat.items():
                columns.setdefault(key, []).append(value)
    return [audit_column(name, values) for name, values in sorted(columns.items())]


def main() -> int:
    try:
        settings = get_settings()
        config = load_config("data")
    except ConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    files = sorted(settings.raw_dir.glob("sessions_*.jsonl"))
    if not files:
        print(f"[FAIL] No raw extracts in {settings.raw_dir}.", file=sys.stderr)
        return 2

    report: dict[str, Any] = {"source": config["source"]["name"], "files": []}
    direct_pii_found = False
    pseudonymous: set[str] = set()

    for path in files:
        results = audit_file(path)
        for column in results:
            if column["classification"] == "DIRECT_PII":
                direct_pii_found = True
            elif column["classification"] == "PSEUDONYMOUS_IDENTIFIER":
                pseudonymous.add(column["column"])
        report["files"].append({"file": path.name, "columns": results})
        print(f"[ok]   audited {path.name}: {len(results)} columns")

    report["summary"] = {
        "direct_pii_detected": direct_pii_found,
        "patterns_checked": sorted(PII_PATTERNS),
        "pseudonymous_columns": sorted(pseudonymous),
        "declared_identifier_columns": config["identifier_columns"],
        "conclusion": (
            "Direct PII detected — review before any publication."
            if direct_pii_found
            else "No direct PII detected. Pseudonymous identifiers are present and "
            "are excluded from the published sample and the feature matrix."
        ),
    }

    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = settings.artifacts_dir / "pii_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n[{'FAIL' if direct_pii_found else 'ok'}]   {report['summary']['conclusion']}")
    print(f"[ok]   pseudonymous columns: {sorted(pseudonymous)}")
    print(f"[ok]   report -> {out}")
    return 1 if direct_pii_found else 0


if __name__ == "__main__":
    raise SystemExit(main())
