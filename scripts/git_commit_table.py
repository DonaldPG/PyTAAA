#!/usr/bin/env python3
"""
git_commit_table.py – Produce a Markdown table of every commit in the
repository (including orphaned / unreachable commits left behind by
git-filter-repo or similar history rewrites).

Columns
-------
1. Date (ISO-8601, author date)
2. Commit hash (short 12-char)
3. Parent hash(es) – short hash(es), comma-separated.
4. Branch – the branch on whose first-parent lineage this commit sits.
            When multiple branches qualify, the one where the commit is
            closest to the tip (fewest first-parent steps away) wins,
            which identifies the branch where it was originally made.
            "ORPHAN (unreachable)" when no branch reaches it at all.
5. Parent branch – same lookup applied to the first parent commit.

Usage
-----
    uv run python scripts/git_commit_table.py [--repo-root PATH]

    # Write to logs/ (default)
    uv run python scripts/git_commit_table.py

    # Write to a custom directory
    uv run python scripts/git_commit_table.py --output-dir path/to/dir

Notes
-----
* Branches in folders matching ".git_bak*" are ignored.
* All git operations use subprocess; no git Python library required.
* The script reads the .git folder directly for object enumeration so
  that orphaned commits (unreachable from any ref) are included.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_git(args: list[str], repo_root: Path) -> str:
    """Run a git command and return stdout as a string."""
    result = subprocess.run(
        ["git"] + args,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return result.stdout


def short_hash(full_hash: str) -> str:
    """Return the first 12 characters of a commit hash."""
    return full_hash[:12]


# ---------------------------------------------------------------------------
# Step 1 – discover all local branches
# ---------------------------------------------------------------------------

def discover_branches(repo_root: Path) -> dict[str, str]:
    """
    Return a mapping of branch_name → tip_commit_hash for every local
    branch, including those stored only in packed-refs.

    Branches whose path contains a component matching ".git_bak*" are
    excluded.
    """
    branches: dict[str, str] = {}

    ##########################################################################
    # packed-refs
    packed_refs_path = repo_root / ".git" / "packed-refs"
    if packed_refs_path.exists():
        for line in packed_refs_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # Peeled tag lines start with "^"; skip them.
            if line.startswith("^"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            commit_hash, ref = parts
            if not ref.startswith("refs/heads/"):
                continue
            branch_name = ref[len("refs/heads/"):]
            # Ignore branches under .git_bak* folders.
            if _is_bak_branch(branch_name):
                continue
            branches[branch_name] = commit_hash

    ##########################################################################
    # Loose refs under refs/heads/ (may override packed-refs).
    heads_dir = repo_root / ".git" / "refs" / "heads"
    if heads_dir.exists():
        for ref_file in sorted(heads_dir.rglob("*")):
            if ref_file.is_dir():
                continue
            branch_name = str(ref_file.relative_to(heads_dir))
            if _is_bak_branch(branch_name):
                continue
            tip = ref_file.read_text().strip()
            if tip:
                branches[branch_name] = tip

    return branches


def _is_bak_branch(branch_name: str) -> bool:
    """Return True if any path component matches the .git_bak* pattern."""
    return any(
        part.startswith(".git_bak")
        for part in Path(branch_name).parts
    )


# ---------------------------------------------------------------------------
# Step 2 – build first-parent commit → branch mapping
# ---------------------------------------------------------------------------

def build_first_parent_map(
    branches: dict[str, str],
    repo_root: Path,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """
    Walk each branch's --first-parent history and return:

    first_parent_branch : dict[full_hash, branch_name]
        Maps each commit to the branch where it sits closest to the
        tip on the first-parent lineage.  When multiple branches claim
        the same commit (e.g. a shared root), the branch where the
        commit is fewest steps from the tip wins.

    reachable_branches : dict[full_hash, set[branch_name]]
        All branches whose *full* ancestry (not just first-parent)
        reaches each commit, used for orphan detection.
    """
    # first_parent_pos[hash] = (branch_name, steps_from_tip)
    # Lower steps_from_tip wins (commit is more native to that branch).
    first_parent_pos: dict[str, tuple[str, int]] = {}

    # Full ancestry set for orphan detection.
    reachable: dict[str, set[str]] = defaultdict(set)

    for branch_name, tip_hash in sorted(branches.items()):
        # -- first-parent lineage -----------------------------------------
        fp_output = run_git(
            ["log", tip_hash, "--first-parent", "--format=%H"],
            repo_root,
        )
        for steps, commit_hash in enumerate(
            h.strip() for h in fp_output.splitlines() if h.strip()
        ):
            existing = first_parent_pos.get(commit_hash)
            if existing is None or steps < existing[1]:
                first_parent_pos[commit_hash] = (branch_name, steps)

        # -- full ancestry (for reachability / orphan detection) ----------
        full_output = run_git(
            ["log", tip_hash, "--format=%H"],
            repo_root,
        )
        for commit_hash in (
            h.strip() for h in full_output.splitlines() if h.strip()
        ):
            reachable[commit_hash].add(branch_name)

    first_parent_branch: dict[str, str] = {
        h: v[0] for h, v in first_parent_pos.items()
    }
    return first_parent_branch, dict(reachable)


# Keep a thin shim so callers using the old name still work during
# any future refactors; it just delegates to the new function.
def build_commit_branch_map(
    branches: dict[str, str],
    repo_root: Path,
) -> dict[str, set[str]]:
    """Shim – returns reachability map (all branches per commit)."""
    _, reachable = build_first_parent_map(branches, repo_root)
    return reachable  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Step 3 – enumerate ALL commit objects (including orphans)
# ---------------------------------------------------------------------------

def enumerate_all_commits(repo_root: Path) -> list[str]:
    """
    Return a list of full commit hashes for every commit object in the
    repository, including unreachable (orphaned) ones.

    Uses ``git cat-file --batch-all-objects --batch-check`` which reads
    every object in both loose and pack files.
    """
    # Default output format is: <hash> <type> <size>
    output = run_git(
        ["cat-file", "--batch-all-objects", "--batch-check"],
        repo_root,
    )
    hashes = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "commit":
            hashes.append(parts[0])
    return hashes


# ---------------------------------------------------------------------------
# Step 4 – parse per-commit metadata
# ---------------------------------------------------------------------------

_COMMIT_RE = re.compile(
    r"^commit (?P<hash>[0-9a-f]{40})$", re.MULTILINE
)


def parse_commit(commit_hash: str, repo_root: Path) -> dict:
    """
    Return a dict with keys:
        hash, date (datetime), parents ([str])
    parsed from ``git cat-file -p <hash>``.
    """
    raw = run_git(["cat-file", "-p", commit_hash], repo_root)
    parents: list[str] = []
    author_ts: int | None = None
    author_tz: str = "+0000"

    for line in raw.splitlines():
        if line.startswith("parent "):
            parents.append(line[7:].strip())
        elif line.startswith("author "):
            # Format: author Name <email> <unix-ts> <tz>
            # The timestamp is the second-to-last token.
            tokens = line.split()
            if len(tokens) >= 2:
                try:
                    author_ts = int(tokens[-2])
                    author_tz = tokens[-1]
                except ValueError:
                    pass
        elif line == "":
            # Blank line separates header from message body.
            break

    if author_ts is not None:
        dt = datetime.datetime.utcfromtimestamp(author_ts)
        # Apply timezone offset so the displayed time matches the
        # author's local clock (purely cosmetic).
        try:
            sign = 1 if author_tz.startswith("+") else -1
            tz_h = int(author_tz[1:3])
            tz_m = int(author_tz[3:5])
            offset = sign * (tz_h * 60 + tz_m)
            dt = dt + datetime.timedelta(minutes=offset)
        except (ValueError, IndexError):
            pass
        date_str = dt.strftime("%Y-%m-%d %H:%M")
    else:
        date_str = "unknown"
        author_ts = 0

    return {
        "hash": commit_hash,
        "date": date_str,
        "sort_key": author_ts,
        "parents": parents,
    }


def parse_all_commits(
    all_hashes: list[str],
    repo_root: Path,
) -> dict[str, dict]:
    """
    Parse every commit hash and return a dict mapping full hash to
    its metadata (keys: hash, date, sort_key, parents).
    """
    total = len(all_hashes)
    parsed: dict[str, dict] = {}
    for idx, h in enumerate(all_hashes, start=1):
        if idx % 50 == 0 or idx == total:
            print(
                f"  Parsing commit {idx}/{total} …",
                file=sys.stderr,
                flush=True,
            )
        parsed[h] = parse_commit(h, repo_root)
    return parsed


# ---------------------------------------------------------------------------
# Step 5 – assemble rows
# ---------------------------------------------------------------------------

ORPHAN_LABEL = "ORPHAN (unreachable)"


def get_original_branch(
    commit_hash: str,
    first_parent_branch: dict[str, str],
    reachable: dict[str, set[str]],
) -> str:
    """
    Return the branch on whose first-parent lineage this commit sits
    closest to the tip — i.e. the branch where it was originally made.
    Falls back to ORPHAN_LABEL when no branch can reach the commit.
    """
    # Prefer the first-parent winner (already argmin'd by steps-from-tip).
    branch = first_parent_branch.get(commit_hash)
    if branch:
        return branch
    # Commit is reachable via full ancestry but not on any first-parent
    # lineage (e.g. it arrived as a second parent in a merge commit).
    # Return any branch that can reach it so the row isn't blank.
    reachable_set = reachable.get(commit_hash)
    if reachable_set:
        return sorted(reachable_set)[0]
    return ORPHAN_LABEL


def build_rows(
    all_hashes: list[str],
    first_parent_branch: dict[str, str],
    reachable: dict[str, set[str]],
    all_parsed: dict[str, dict],
) -> list[dict]:
    """
    Assemble a list of row dicts from pre-parsed commit metadata.

    Branch is determined by first-parent lineage proximity (the branch
    where the commit was originally made), not by tip-date.
    """
    rows = []
    for commit_hash in all_hashes:
        meta = all_parsed[commit_hash]

        # Original branch via first-parent lineage.
        branch = get_original_branch(
            commit_hash, first_parent_branch, reachable
        )

        # Parent information.
        parent_hashes = meta["parents"]
        parent_hash_cell = (
            ", ".join(short_hash(p) for p in parent_hashes)
            if parent_hashes
            else "— (root)"
        )

        # Original branch for each parent (first + second for merges).
        if parent_hashes:
            parent_branch = get_original_branch(
                parent_hashes[0], first_parent_branch, reachable
            )
            if len(parent_hashes) > 1:
                # Merge commit – include second-parent branch.
                pb2 = get_original_branch(
                    parent_hashes[1], first_parent_branch, reachable
                )
                parent_branch = f"{parent_branch} | {pb2}"
        else:
            parent_branch = "— (root)"

        rows.append({
            "hash": short_hash(commit_hash),
            "full_hash": commit_hash,
            "date": meta["date"],
            "sort_key": meta["sort_key"],
            "branch": branch,
            "parent_branch": parent_branch,
            "parent_hashes": parent_hash_cell,
        })

    # Sort oldest-first.
    rows.sort(key=lambda r: r["sort_key"])
    return rows


# ---------------------------------------------------------------------------
# Step 6 – render Markdown table
# ---------------------------------------------------------------------------

def render_markdown(rows: list[dict]) -> str:
    """Return the full Markdown table as a string."""
    lines: list[str] = []

    lines.append(
        "# Git Commit History Table\n\n"
        "_Generated by `scripts/git_commit_table.py`. "
        "Includes orphaned (unreachable) commits. "
        "Branch shown is the branch on whose first-parent lineage "
        "the commit sits closest to the tip (i.e. where it was "
        "originally made)._\n"
    )

    header = (
        "| Date | Commit Hash | Parent Hash "
        "| Branch | Parent Branch |"
    )
    separator = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(separator)

    for row in rows:
        cells = [
            row["date"],
            f"`{row['hash']}`",
            row["parent_hashes"],
            row["branch"],
            row["parent_branch"],
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append(f"\n_Total commits: {len(rows)}_\n")
    return "\n".join(lines)


def render_csv(rows: list[dict]) -> str:
    """Return a CSV string with the same columns as the Markdown table."""
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "date", "commit_hash", "parent_hash",
        "branch", "parent_branch",
    ])
    for row in rows:
        writer.writerow([
            row["date"],
            row["hash"],
            row["parent_hashes"],
            row["branch"],
            row["parent_branch"],
        ])
    return output.getvalue()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write Markdown and CSV tables of all git commits, "
                    "including orphaned ones.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the git repository root (default: current dir).",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Directory for output files (default: logs/).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    git_dir = repo_root / ".git"
    if not git_dir.is_dir():
        print(
            f"ERROR: {repo_root} does not appear to be a git repository "
            f"(no .git directory found).",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "git_commit_table.md"
    csv_path = output_dir / "git_commit_table.csv"

    print("Discovering branches …", file=sys.stderr)
    branches = discover_branches(repo_root)
    print(f"  Found {len(branches)} local branches.", file=sys.stderr)

    print("Building first-parent branch map …", file=sys.stderr)
    first_parent_branch, reachable = build_first_parent_map(
        branches, repo_root
    )
    print(
        f"  {len(reachable)} reachable commits indexed.",
        file=sys.stderr,
    )

    print("Enumerating all commit objects (incl. orphans) …",
          file=sys.stderr)
    all_hashes = enumerate_all_commits(repo_root)
    orphan_count = sum(
        1 for h in all_hashes if h not in reachable
    )
    print(
        f"  {len(all_hashes)} total commits "
        f"({orphan_count} orphaned / unreachable).",
        file=sys.stderr,
    )

    print("Parsing commit metadata …", file=sys.stderr)
    all_parsed = parse_all_commits(all_hashes, repo_root)

    print("Assembling rows …", file=sys.stderr)
    rows = build_rows(
        all_hashes, first_parent_branch, reachable, all_parsed
    )

    print(f"Writing {md_path} …", file=sys.stderr)
    md_path.write_text(render_markdown(rows), encoding="utf-8")

    print(f"Writing {csv_path} …", file=sys.stderr)
    csv_path.write_text(render_csv(rows), encoding="utf-8")

    print(
        f"\nDone. {len(rows)} commits written to:\n"
        f"  {md_path}\n"
        f"  {csv_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
