#!/usr/bin/env python3
"""Run production workflows and compare stable refactor artifacts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DATA_ROOT = Path("/Users/donaldpg/pyTAAA_data_static")
GATE_LOCK = REPO_ROOT / ".refactor_baseline" / "production_gate.lock"
MAX_CONCURRENT_AFTER_TESTS = 2
GATE_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTAAA_SKIP_PLOTS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
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
    r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?"
)
LATEST_RECORD_LINES = {
    "PyTAAA_diagnostic.params": 8,
    "PyTAAA_ranks.params": 3,
}


def _resource_limited_command(command: list[str]) -> list[str]:
    """Apply background QoS on macOS when ``taskpolicy`` is available."""
    taskpolicy = shutil.which("taskpolicy")
    if taskpolicy is None:
        return command
    return [taskpolicy, "-b", *command]


def _run_command(command: list[str], log_path: Path) -> None:
    """Run one production command and save its combined output."""
    environment = os.environ.copy()
    environment.update(GATE_ENVIRONMENT)
    with log_path.open("w") as log_file:
        completed = subprocess.run(
            _resource_limited_command(command),
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=lambda: os.nice(15),
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{log_path}"
        )


def run_production_commands(after_dir: Path) -> None:
    """Execute at most two independent production workflows concurrently."""
    model_jobs = []
    for config in MODEL_CONFIGS:
        model_jobs.append(
            (
                [
                    "uv",
                    "run",
                    "python",
                    "pytaaa_main.py",
                    "--json",
                    str(config),
                ],
                after_dir / f"{config.stem}.log",
            )
        )

    with ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_AFTER_TESTS
    ) as executor:
        futures = [
            executor.submit(_run_command, command, log_path)
            for command, log_path in model_jobs
        ]
        for future in futures:
            future.result()

    # These commands share the Abacus data store and must remain sequential.
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


def _comparable_lines(path: Path) -> list[str]:
    """Return stable content for exact or append-only params files."""
    lines = _normalized_lines(path)
    record_length = LATEST_RECORD_LINES.get(path.name)
    if record_length is None:
        return lines
    return lines[-record_length:]


def compare_artifacts(before_dir: Path, after_dir: Path) -> list[str]:
    """Compare stable params content and latest append-only records."""
    mismatches: list[str] = []
    before_params = before_dir / "params_files"
    after_params = after_dir / "params_files"

    for before_path in sorted(before_params.glob("*.params")):
        after_path = after_params / before_path.name
        if not after_path.exists():
            mismatches.append(f"missing_after: params_files/{before_path.name}")
        elif _comparable_lines(before_path) != _comparable_lines(after_path):
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
        GATE_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with GATE_LOCK.open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                run_production_commands(args.after)
                capture_params(args.after)
            except Exception as error:
                print(
                    "BASELINE GATE: FAIL - production command failed: "
                    f"{error}"
                )
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
