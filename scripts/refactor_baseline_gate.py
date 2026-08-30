#!/usr/bin/env python3
"""Run production workflows and compare stable refactor artifacts."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DATA_ROOT = Path("/Users/donaldpg/pyTAAA_data_static")
MODEL_CONFIGS = (
    STATIC_DATA_ROOT / "naz100_pine/pytaaa_naz100_pine.json",
    STATIC_DATA_ROOT / "naz100_hma/pytaaa_naz100_hma.json",
    STATIC_DATA_ROOT / "naz100_pi/pytaaa_naz100_pi.json",
    STATIC_DATA_ROOT / "sp500_hma/pytaaa_sp500_hma.json",
    STATIC_DATA_ROOT / "sp500_pine/pytaaa_sp500_pine.json",
)
ABACUS_CONFIG = (
    STATIC_DATA_ROOT
    / "naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json"
)
TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"
)


def _run_command(command: list[str], log_path: Path) -> None:
    """Run one production command and save its combined output."""
    with log_path.open("w") as log_file:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{log_path}"
        )


def run_production_commands(after_dir: Path) -> None:
    """Execute the production model and Abacus workflows."""
    for config in MODEL_CONFIGS:
        _run_command(
            ["uv", "run", "python", "pytaaa_main.py", "--json", str(config)],
            after_dir / f"{config.stem}.log",
        )

    _run_command(
        [
            "uv",
            "run",
            "python",
            "recommend_model.py",
            "--json",
            str(ABACUS_CONFIG),
        ],
        after_dir / "abacus_recommendation.log",
    )
    _run_command(
        [
            "uv",
            "run",
            "python",
            "daily_abacus_update.py",
            "--json",
            str(ABACUS_CONFIG),
            "--verbose",
        ],
        after_dir / "abacus_daily.log",
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_params(output_dir: Path) -> None:
    """Capture params files using the baseline's flattened layout."""
    params_dir = output_dir / "params_files"
    shutil.rmtree(params_dir, ignore_errors=True)
    params_dir.mkdir(parents=True)

    source_files = sorted(STATIC_DATA_ROOT.rglob("*.params"))
    checksum_lines = []
    for source_path in source_files:
        checksum_lines.append(
            f"MD5 ({source_path}) = {_md5(source_path)}\n"
        )
        shutil.copy2(source_path, params_dir / source_path.name)

    (output_dir / "params_checksums.txt").write_text(
        "".join(checksum_lines)
    )


def _normalized_lines(path: Path) -> list[str]:
    lines = path.read_text(errors="replace").splitlines()
    return [TIMESTAMP_PATTERN.sub("TIMESTAMP", line) for line in lines]


def compare_artifacts(before_dir: Path, after_dir: Path) -> list[str]:
    """Compare baseline params content while ignoring timestamps."""
    mismatches: list[str] = []
    before_params = before_dir / "params_files"
    after_params = after_dir / "params_files"

    for before_path in sorted(before_params.glob("*.params")):
        after_path = after_params / before_path.name
        if not after_path.exists():
            mismatches.append(f"missing_after: params_files/{before_path.name}")
        elif _normalized_lines(before_path) != _normalized_lines(after_path):
            mismatches.append(f"content_mismatch: {before_path.name}")

    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if not args.before.exists():
        print(f"ERROR: baseline directory does not exist: {args.before}")
        return 2

    args.after.mkdir(parents=True, exist_ok=True)
    if args.run:
        try:
            run_production_commands(args.after)
            capture_params(args.after)
        except Exception as error:
            print(f"BASELINE GATE: FAIL - production command failed: {error}")
            return 1

    mismatches = compare_artifacts(args.before, args.after)
    if mismatches:
        print("BASELINE GATE: FAIL")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        return 1

    print("BASELINE GATE: PASS")
    print(f"  before={args.before}")
    print(f"  after={args.after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
