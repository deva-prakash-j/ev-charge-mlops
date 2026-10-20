"""Run or serve the Prefect orchestration flows."""

from __future__ import annotations

import argparse
import sys

from evcharge.config import SOURCES
from evcharge.orchestration.flows import (
    monthly_monitoring_flow,
    serve_monthly_monitor,
    training_flow,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EV charge MLOps Prefect flows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Register and serve the monthly raw-data flow.")
    train_parser = subparsers.add_parser("train", help="Run the training flow once.")
    train_parser.add_argument("--source", choices=SOURCES, default="raw")
    train_parser.add_argument("--fetch", action="store_true")
    monitor_parser = subparsers.add_parser(
        "monitor", help="Run the monthly batch monitor once."
    )
    monitor_parser.add_argument("--source", choices=SOURCES, default="raw")
    monitor_parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip the ACN-Data download (only the raw source is ever fetched).",
    )

    args = parser.parse_args(argv)
    if args.command == "serve":
        serve_monthly_monitor()
    elif args.command == "train":
        training_flow(source=args.source, fetch=args.fetch)
    elif args.command == "monitor":
        monthly_monitoring_flow(source=args.source, fetch=not args.no_fetch)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
