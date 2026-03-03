"""Graph and network plot generation for PyTAAA.

This module provides standalone graph-plotting functions that are used
by both MakeValuePlot and WriteWebPage_pi. Extracting them here
eliminates the previous MakeValuePlot <-> WriteWebPage_pi coupling and
makes each function independently testable.
"""

import os

from functions.GetParams import get_webpage_store


def makeMinimumSpanningTree(json_fn: str) -> str:
    """Generate a minimum spanning tree plot from stock correlations.

    Builds a network graph where nodes are stocks in the universe and
    edges are weighted by the 22-day return correlation between each
    pair. Saves the plot as ``minimum_spanning_tree.png`` in the
    configured web output directory.

    Args:
        json_fn: Path to JSON configuration file.

    Returns:
        HTML fragment string containing an ``<img>`` tag pointing to
        the saved plot, ready for embedding in the dashboard page.
    """
    from functions.make_stock_xcorr_network_plots import (
        make_networkx_spanning_tree_plot,
    )

    ##########################################################################
    # Resolve output path from config, then delegate to the network-plot
    # helper which handles all matplotlib and networkx operations.
    ##########################################################################
    webpage_dir = get_webpage_store(json_fn)
    figure7apath = os.path.join(webpage_dir, "minimum_spanning_tree.png")

    make_networkx_spanning_tree_plot(figure7apath, json_fn)

    # Use a relative path for the HTML <img> src so it works regardless
    # of where the web server serves files from.
    figure7apath = "minimum_spanning_tree.png"
    figure7a_htmlText = (
        "\n<br><h3>Daily stock minimum-spanning tree analyis. "
        "Based on 22 day performance correlations.</h3>\n"
    )
    figure7a_htmlText += (
        "\nCorrelations for graph network based on daily variation "
        "quotes for the stock universe.\n"
    )
    figure7a_htmlText += (
        "\nUse to visually observe if patterns are related to "
        "(desirable attributes from) portfolio diversity\n"
    )
    figure7a_htmlText += (
        '<br><img src="'
        + figure7apath
        + '" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'
    )

    return figure7a_htmlText
