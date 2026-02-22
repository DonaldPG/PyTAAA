import os

import networkx as nx
import numpy as np
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from functions.GetParams import get_holdings, get_symbols_file
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF


def make_networkx_spanning_tree_plot(
        output_path: str, json_fn: str
) -> None:
    """Generate and save a minimum spanning tree network plot of stocks.

    Computes pairwise correlations between stocks over the most recent
    22 trading days, converts to a Mantegna distance matrix, computes
    the minimum spanning tree, and saves the resulting network plot as
    a PNG file.

    Args:
        output_path: Full file path where the PNG will be saved.
        json_fn: Path to the JSON configuration file used to locate
            the HDF5 quotes store.
    """
    #########################################
    # Load stock quotes from HDF5 file.
    #########################################
    symbols_file = get_symbols_file(json_fn)
    adjClose, symbols, _, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)

    # Load current stock holdings; fall back to empty set if unavailable.
    try:
        holdings_data = get_holdings(json_fn)
        held_symbols: set[str] = set(holdings_data.get("stocks", []))
    except Exception as exc:
        print(
            " ... make_networkx_spanning_tree_plot: could not load "
            "holdings (%s); no halos will be drawn." % exc
        )
        held_symbols = set()

    # Use the most recent 22 trading days of daily returns.
    n_days = 22
    if adjClose.shape[1] < n_days + 1:
        n_days = adjClose.shape[1] - 1

    # adjClose shape: (n_stocks, n_dates)
    prices = adjClose[:, -(n_days + 1):]
    returns = prices[:, 1:] / prices[:, :-1] - 1.0

    #########################################
    # Filter stocks with invalid data.
    #########################################
    valid_mask = (
        np.all(np.isfinite(returns), axis=1)
        & (np.std(returns, axis=1) > 0)
    )
    returns = returns[valid_mask]
    symbols_valid = [s for s, v in zip(symbols, valid_mask) if v]
    n = len(symbols_valid)

    if n < 3:
        print(
            " ... make_networkx_spanning_tree_plot: not enough valid "
            "stocks to build graph (n=%d)" % n
        )
        return

    #########################################
    # Compute correlation and distance matrices.
    #########################################
    corr = np.corrcoef(returns)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    # Mantegna metric: d = sqrt(2 * (1 - rho))
    dist = np.sqrt(2.0 * (1.0 - corr))

    #########################################
    # Compute the minimum spanning tree.
    #########################################
    # only keep upper triangle (scipy's minimum_spanning_tree expects this format)
    dist = np.triu(dist, k=1)
    print("   . DEBUG: dist.size in min spanning tree: %s" % str(dist.size))
    dist_non_zero_size = dist[dist > 0].size
    print("   . DEBUG: dist non-zero size: %s" % str(dist_non_zero_size))
    mst = minimum_spanning_tree(dist)
    print("   . DEBUG: mst non-zero size: %s" % str(mst.nnz))
    mst_coo = coo_matrix(mst)

    #########################################
    # Build networkx graph from MST edges.
    #########################################
    # Trees are always planar; Kamada-Kawai layout uses edge weights as
    # target distances, producing crossing-free layouts for sparse trees.
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for r, c, d in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        G.add_edge(int(r), int(c), weight=float(d))

    node_labels = {i: sym for i, sym in enumerate(symbols_valid)}

    # Kamada-Kawai: minimises stress between graph distances and
    # Euclidean positions — standard layout for financial MST plots.
    pos = nx.kamada_kawai_layout(G, weight="weight")

    #########################################
    # Draw the network graph.
    #########################################
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor("#F8F8F8")

    # Identify nodes for currently held stocks.
    held_nodes = [
        i for i, sym in enumerate(symbols_valid) if sym in held_symbols
    ]

    # Draw order: edges → yellow halos → regular nodes → labels.
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="black", width=0.7, alpha=0.85,
    )
    # Yellow halo: 33% larger than base node, drawn behind regular nodes.
    if held_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=held_nodes,
            node_size=480, node_color="#FFE033",
            edgecolors="#FFD700", linewidths=1.0,
        )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=360, node_color="#F1F1F1",
        edgecolors="white", linewidths=0.5,
    )
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax,
        font_size=8, font_color="#111111",
    )

    ax.set_title(
        "Stock Minimum Spanning Tree  (22-day return correlations)",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
