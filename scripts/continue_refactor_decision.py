#!/usr/bin/env python3
"""Autonomously manage the refactor gate loop.

This controller is intentionally autonomous: it waits for active gate work,
executes the production baseline gate when needed, continues when the gate
passes, and reverts the last refactor change when the gate fails.

Return codes:
- 0: PROCEED, the gate passed
- 1: REVERT, the gate failed and the last change was reverted
- 2: WAIT, the gate is still running or the current state is incomplete
- 3: the gate could not be executed or a fatal error occurred
"""

from __future__ import annotations

import subprocess
import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_LOG = REPO_ROOT / ".refactor_baseline" / "after_gate" / "gate_run.log"
GATE_DIR = GATE_LOG.parent
GATE_COMMIT = GATE_DIR / "candidate_commit.txt"
MAX_WAIT_ATTEMPTS = 60
WAIT_SECONDS = 15


def _find_gate_processes() -> list[str]:
    """Return running baseline-gate-related processes if any are active."""
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,comm,args"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    patterns = (
        "scripts/refactor_baseline_gate.py",
        "pytaaa_main.py --json",
        "recommend_model.py",
        "daily_abacus_update.py",
    )
    matches: list[str] = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        if any(pattern in text for pattern in patterns):
            matches.append(text)
    return matches


def _read_log() -> str:
    if not GATE_LOG.exists():
        return ""
    return GATE_LOG.read_text(errors="replace")


def _git_output(*args: str) -> str:
    """Return stripped output from a read-only Git command."""
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _current_commit() -> str:
    return _git_output("rev-parse", "HEAD")


def _gate_commit() -> str:
    if not GATE_COMMIT.exists():
        return ""
    return GATE_COMMIT.read_text().strip()


def _run_gate(candidate_commit: str) -> int:
    """Execute the production baseline gate and return the exit code."""
    shutil.rmtree(GATE_DIR, ignore_errors=True)
    GATE_DIR.mkdir(parents=True)
    GATE_COMMIT.write_text(f"{candidate_commit}\n")
    print(
        "RUN: executing the production gate automatically for "
        f"{candidate_commit[:12]}."
    )
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/refactor_baseline_gate.py",
            "--before",
            ".refactor_baseline/before",
            "--after",
            ".refactor_baseline/after_gate",
            "--run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    GATE_LOG.write_text(completed.stdout + completed.stderr)
    if GATE_LOG.stat().st_size:
        print(GATE_LOG.read_text(errors="replace").strip())
    return completed.returncode


def _revert_candidate(candidate_commit: str) -> int:
    """Revert the exact candidate commit associated with the failed gate."""
    print(
        "REVERT: production gate failed. Reverting candidate "
        f"{candidate_commit[:12]} automatically."
    )
    completed = subprocess.run(
        ["git", "revert", "--no-edit", candidate_commit],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip())

    if completed.returncode != 0:
        return 3

    GATE_COMMIT.write_text(f"reverted:{candidate_commit}\n")
    return 1


def main() -> int:
    attempts = 0

    while True:
        running = _find_gate_processes()

        if running:
            if attempts >= MAX_WAIT_ATTEMPTS:
                print("WAIT: the baseline gate is still running beyond the auto-wait limit.")
                for item in running[:10]:
                    print(f"  - {item}")
                return 2
            print("WAIT: production baseline gate is still running. Waiting for it to finish.")
            for item in running[:10]:
                print(f"  - {item}")
            time.sleep(WAIT_SECONDS)
            attempts += 1
            continue

        current_commit = _current_commit()
        recorded_commit = _gate_commit()
        if recorded_commit != current_commit:
            _run_gate(current_commit)
            attempts = 0
            continue

        log_text = _read_log()
        normalized = log_text.lower()
        if "baseline gate: pass" in normalized:
            print("PROCEED: the production baseline gate passed.")
            print("Autonomous refactor loop continues to the next step.")
            return 0

        if "baseline gate: fail" in normalized:
            return _revert_candidate(recorded_commit)

        if not log_text.strip():
            _run_gate(current_commit)
            continue

        print("WAIT: gate log exists, but no final PASS/FAIL verdict was written yet.")
        if attempts >= MAX_WAIT_ATTEMPTS:
            return 2
        time.sleep(WAIT_SECONDS)
        attempts += 1


if __name__ == "__main__":
    raise SystemExit(main())
