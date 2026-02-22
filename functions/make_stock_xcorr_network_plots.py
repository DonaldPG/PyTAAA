import os

import numpy as np
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from functions.GetParams import get_symbols_file
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
    mst = minimum_spanning_tree(dist)
    mst_coo = coo_matrix(mst)

    #########################################
    # Compute 2D positions via classical MDS.
    #########################################
    d_sq = dist ** 2
    # Double-centering without allocating a full n×n centering matrix.
    row_mean = d_sq.mean(axis=1, keepdims=True)
    col_mean = d_sq.mean(axis=0, keepdims=True)
    grand_mean = d_sq.mean()
    gram = -0.5 * (d_sq - row_mean - col_mean + grand_mean)

    # Compute only the top 2 eigenvalues/vectors (much faster for large n).
    from scipy.sparse.linalg import eigsh
    eigenvalues, eigenvectors = eigsh(gram, k=2, which='LA')

    # eigsh returns in ascending order; reverse to descending.
    pos_eigs = np.maximum(eigenvalues[::-1], 0.0)
    embedding = eigenvectors[:, ::-1] * np.sqrt(pos_eigs)
    x_pos = embedding[:, 0]
    y_pos = embedding[:, 1]

    #########################################
    # Draw the network graph.
    #########################################
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('#F0F0F0')

    # Draw MST edges.
    for r, c in zip(mst_coo.row, mst_coo.col):
        ax.plot(
            [x_pos[r], x_pos[c]],
            [y_pos[r], y_pos[c]],
            color='steelblue',
            alpha=0.6,
            linewidth=1.0,
            zorder=1,
        )

    # Draw nodes.
    ax.scatter(
        x_pos, y_pos,
        s=50, c='steelblue', zorder=3, edgecolors='white', linewidths=0.5
    )

    # Label each node with its stock symbol.
    for i, sym in enumerate(symbols_valid):
        ax.annotate(
            sym,
            (x_pos[i], y_pos[i]),
            fontsize=6,
            ha='center',
            va='bottom',
            xytext=(0, 4),
            textcoords='offset points',
        )

    ax.set_title(
        'Stock Minimum Spanning Tree  (22-day return correlations)',
        fontsize=11,
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
