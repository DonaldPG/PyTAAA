"""Validate minimal model JSON schema for pytaaa model manifests.

Usage:
    uv run python scripts/validate_model_json.py /path/to/pytaaa_sp500_pine.json

This script asserts the presence of required top-level keys and prints
resolution suggestions for `data_store` if absent.
"""
import json
import os
import sys
from typing import Any, Dict

REQUIRED_KEYS = ["id", "name"]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def validate_model_json(path: str) -> None:
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        raise SystemExit(2)
    data = load_json(path)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        print(f"ERROR: Missing required keys: {missing}")
        raise SystemExit(3)
    if not isinstance(data.get("id"), str):
        print("ERROR: `id` must be a string")
        raise SystemExit(4)
    if data.get("id") != "sp500_pine":
        print(f"WARNING: `id` is '{data.get('id')}', expected 'sp500_pine' for this model")
    ds = data.get("data_store")
    if ds:
        resolved = os.path.expanduser(ds)
        if not os.path.exists(resolved):
            print(f"WARNING: data_store path does not exist: {resolved}")
        else:
            print(f"OK: data_store resolved to: {resolved}")
    else:
        suggested = os.path.join(os.path.dirname(path), "data_store")
        print("INFO: No `data_store` key present.")
        print(f"INFO: Suggest using: {suggested}")

    print("Model JSON looks valid (basic checks passed).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/validate_model_json.py /path/to/model.json")
        raise SystemExit(1)
    validate_model_json(sys.argv[1])
