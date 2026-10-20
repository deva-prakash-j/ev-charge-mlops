"""Train and register the sample model when the Compose registry is empty."""

from __future__ import annotations

import subprocess
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from evcharge.config import get_settings, load_config
from evcharge.models.registry import registered_model_name, resolve_tracking_uri


def production_alias_exists(client: MlflowClient, model_name: str) -> bool:
    try:
        model = client.get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return False
        raise
    return "production" in model.aliases


def main() -> int:
    settings = get_settings()
    model_config = load_config("model")
    tracking_uri = resolve_tracking_uri(settings, model_config)
    model_name = registered_model_name(
        "sample",
        "energy",
        model_config["registry"]["registered_model_prefix"],
    )
    client = MlflowClient(tracking_uri=tracking_uri)

    if production_alias_exists(client, model_name):
        print(f"[ok]   Production model already registered: {model_name}")
        return 0

    print("[info] No Production alias found; training and registering the sample models.")
    result = subprocess.run(
        [sys.executable, "-m", "evcharge.pipeline", "--source", "sample"],
        check=False,
    )
    if result.returncode != 0:
        print(
            f"[FAIL] Sample pipeline exited with status {result.returncode}.",
            file=sys.stderr,
        )
        return result.returncode

    if not production_alias_exists(client, model_name):
        print(
            f"[FAIL] Training completed without a Production alias for {model_name}.",
            file=sys.stderr,
        )
        return 1

    print(f"[ok]   Production alias is ready: {model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
