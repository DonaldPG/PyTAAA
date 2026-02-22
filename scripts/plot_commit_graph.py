#!/usr/bin/env python3
"""plot_commit_graph.py - Plot a commit graph (all or filtered by date).

Reads logs/git_commit_table.csv, optionally filters to a target date,
builds an undirected graph of commit -> parent relationships using
networkx, and saves the plot to logs/commit_graph_<date|all>.png.

Usage
-----
    # All commits (default)
    uv run python scripts/plot_commit_graph.py

    # Filter to a single date
    uv run python scripts/plot_commit_graph.py --date 2014-02-09

    # Custom input / output paths
    uv run python scripts/plot_commit_graph.py \\
        --csv logs/git_commit_table.csv \\
        --output-dir logs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT_PLACEHOLDER = "— (root)"
MERGE_SEPARATOR = ", "


def node_label(commit_hash: str) -> str:
    """Return the first 5 characters of a commit hash as a node label."""
    return commit_hash[:5]


def parse_parents(parent_hash_cell: str) -> list[str]:
    """
    Parse the parent_hash column value and return a list of full
    (or partial) commit hashes.

    The column may contain:
      - "— (root)"        → no parents
      - "<hash>"          → one parent
      - "<hash>, <hash>"  → two parents (merge commit)
    """
    cell = parent_hash_cell.strip()
    if not cell or cell == ROOT_PLACEHOLDER:
        return []
    return [p.strip() for p in cell.split(MERGE_SEPARATOR) if p.strip()]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_rows(
    csv_path: Path,
    target_date: str,
) -> tuple[list[dict], dict[str, int], dict[str, str]]:
    """
    Read the CSV and return:
      - rows: all rows when target_date is empty, otherwise only those
        whose date starts with target_date
      - global_order: mapping of commit_hash -> row-index across the
        full CSV (0 = oldest), used to assign x-positions.
      - commit_branch: mapping of every commit_hash in the full CSV
        to its branch name, used to assign y-positions.
    """
    rows = []
    global_order: dict[str, int] = {}
    commit_branch: dict[str, str] = {}
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            h = row["commit_hash"].strip()
            global_order[h] = idx
            commit_branch[h] = row.get("branch", "unknown").strip()
            if not target_date or row["date"].startswith(target_date):
                rows.append(row)
    return rows, global_order, commit_branch


def build_graph(rows: list[dict]) -> tuple[nx.Graph, dict[str, str]]:
    """
    Build an undirected graph from the filtered rows.

    Returns
    -------
    graph : nx.Graph
    labels : dict mapping node-id → display label (first 5 chars)
    """
    graph = nx.Graph()
    labels: dict[str, str] = {}

    def add_node(commit_hash: str) -> None:
        """Add a node if not already present."""
        if commit_hash not in graph:
            graph.add_node(commit_hash)
            labels[commit_hash] = node_label(commit_hash)

    for row in rows:
        commit = row["commit_hash"].strip()
        parents = parse_parents(row["parent_hash"])

        # Ensure the commit node exists.
        add_node(commit)

        for parent in parents:
            # Ensure parent node exists (may not be in the filtered rows).
            add_node(parent)
            # Add edge: commit ↔ parent.
            graph.add_edge(commit, parent)

    return graph, labels


def compute_branch_y_tracks(
    graph: nx.Graph,
    commit_branch: dict[str, str],
) -> tuple[dict[str, float], list[str]]:
    """
    Assign a fixed integer y-track to every unique branch name found
    among the graph's nodes.

    Returns
    -------
    branch_y : dict mapping branch_name -> float y-value
    ordered_branches : branch names in y-order (index 0 = y 0.0)
    """
    # Collect unique branch names from nodes actually in the graph.
    seen: set[str] = set()
    for node in graph.nodes():
        seen.add(commit_branch.get(node, "unknown"))

    # Sort: master/main first (y=0), then alphabetically, unknown last.
    def sort_key(b: str) -> tuple[int, str]:
        if b in ("master", "main"):
            return (0, b)
        if b.startswith("ORPHAN") or b == "unknown":
            return (2, b)
        return (1, b)

    ordered_branches = sorted(seen, key=sort_key)
    branch_y = {b: float(i) for i, b in enumerate(ordered_branches)}
    return branch_y, ordered_branches


def draw_and_save(
    graph: nx.Graph,
    labels: dict[str, str],
    global_order: dict[str, int],
    commit_branch: dict[str, str],
    target_date: str,
    output_path: Path,
) -> None:
    """
    Draw the graph with a left-to-right timeline layout. Each branch
    name from the CSV is assigned a fixed horizontal lane (y-track).
    Branch names are labelled on the left margin. Save as PNG.
    """
    # Build a directed graph (parent -> child) for topology analysis.
    digraph = nx.DiGraph()
    digraph.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        # Edge direction: older node -> newer node.
        if global_order.get(u, -1) < global_order.get(v, -1):
            digraph.add_edge(u, v)
        else:
            digraph.add_edge(v, u)

    # Assign a fixed y-track to every branch name.
    branch_y, ordered_branches = compute_branch_y_tracks(
        graph, commit_branch
    )

    # Build final positions: x = global time index, y = branch lane.
    pos: dict[str, tuple[float, float]] = {
        node: (
            float(global_order.get(node, -1)),
            branch_y.get(commit_branch.get(node, "unknown"), 0.0),
        )
        for node in graph.nodes()
    }

    # Label offsets: push labels slightly above the node.
    OFFSET = 0.3
    label_pos: dict[str, tuple[float, float]] = {
        node: (pos[node][0], pos[node][1] + OFFSET)
        for node in graph.nodes()
    }

    # Colour nodes: orange for merge commits (2 parents in the digraph),
    # green for branch points (2+ children), blue for ordinary commits.
    node_colors = []
    for node in graph.nodes():
        in_deg = digraph.in_degree(node)
        out_deg = digraph.out_degree(node)
        if in_deg >= 2:
            node_colors.append("#E07B39")  # merge commit - orange
        elif out_deg >= 2:
            node_colors.append("#4CAF50")  # branch point - green
        else:
            node_colors.append("#4C72B0")  # ordinary - blue

    all_y = [p[1] for p in pos.values()]
    x_vals = [p[0] for p in pos.values()]
    x_min, x_max = min(x_vals), max(x_vals)

    fig, ax = plt.subplots(figsize=(75, 8))

    # Faint horizontal gridlines, one per branch lane.
    for y_val in branch_y.values():
        ax.axhline(
            y=y_val, color="#e0e0e0", linewidth=0.6, zorder=0
        )

    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        edge_color="#888888", width=1.5, alpha=0.75,
        arrows=False,
    )
    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=node_colors, node_size=350, alpha=0.95,
    )
    # Draw commit hash labels rotated 90 degrees (vertical).
    for node, (lx, ly) in label_pos.items():
        ax.text(
            lx, ly, labels.get(node, ""),
            ha="center", va="bottom",
            fontsize=9, color="#111111",
            rotation=90,
        )

    # Branch name labels on the left margin.
    label_x = x_min - 1.5
    for branch_name, y_val in branch_y.items():
        ax.text(
            label_x, y_val, branch_name,
            ha="right", va="center", fontsize=7,
            color="#222222", style="italic",
        )

    # Legend.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Ordinary commit"),
        Patch(facecolor="#4CAF50", label="Branch point (2 children)"),
        Patch(facecolor="#E07B39", label="Merge commit (2 parents)"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left",
        fontsize=10, framealpha=0.8,
    )

    date_label = f"dated {target_date}" if target_date else "all dates"
    ax.set_title(
        f"Commit graph - {date_label}   "
        f"({graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges)   "
        f"oldest -> newest (left -> right)",
        fontsize=11,
        pad=10,
    )
    ax.set_xlim(x_min - 12, x_max + 1)
    ax.set_ylim(min(all_y) - 0.7, max(all_y) + 0.7)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot an undirected commit graph for a given date.",
    )
    parser.add_argument(
        "--date",
        default="",
        help=(
            "Date to filter rows on (YYYY-MM-DD). "
            "Omit or pass empty string to include all dates."
        ),
    )
    parser.add_argument(
        "--csv",
        default="logs/git_commit_table.csv",
        help="Path to input CSV (default: logs/git_commit_table.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Directory for the output PNG (default: logs/).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    date_slug = args.date if args.date else "all"
    output_path = output_dir / f"commit_graph_{date_slug}.png"

    print(f"Loading rows from {csv_path} …", file=sys.stderr)
    rows, global_order, commit_branch = load_rows(csv_path, args.date)
    if not rows:
        print(
            f"ERROR: No rows found"
            + (f" for date '{args.date}'" if args.date else "") + ". "
            f"Check the CSV path.",
            file=sys.stderr,
        )
        sys.exit(1)
    date_msg = f"date '{args.date}'" if args.date else "all dates"
    print(f"  {len(rows)} rows matched {date_msg}.", file=sys.stderr)

    print("Building graph …", file=sys.stderr)
    graph, labels = build_graph(rows)
    print(
        f"  {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges.",
        file=sys.stderr,
    )

    print("Drawing graph …", file=sys.stderr)
    draw_and_save(
        graph, labels, global_order, commit_branch, args.date, output_path
    )
    print(f"Done. Output: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
