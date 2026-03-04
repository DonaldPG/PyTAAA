"""HTML report generation for PyTAAA portfolio status emails and web pages.

This module owns the HTML rendering concern, separating it from the
orchestration logic in ``run_pytaaa.py``.  Templates are stored in
``functions/templates/`` and rendered via Jinja2.

Key Functions:
    build_holdings_html_report: Render the holdings table and report
        footer from structured data.

Example::

    from functions.report_builders import build_holdings_html_report

    html = build_holdings_html_report(
        holdings_rows=[
            {
                "symbol": "AAPL ",
                "shares": "   100",
                "buy_price": "150.00",
                "purchase_cost": "15000.00",
                "cumu_purchase": "15000.00",
                "current_price": "175.00",
                "profit_pct": " 16.67%",
                "value": "17500.00",
                "cumu_value": "17500.00",
                "rank": "  5",
                "cluster": "2",
                "sector": "Technology",
                "industry": "Consumer Electronics",
            }
        ],
        trade_message="<br>",
        lifetime_profit=7500.0,
        cumulative_cash_in=10000.0,
        lifetime_profit_annualized=7500.0,
        removed_tickers=[],
        added_tickers=[],
        edition="1.0",
        ip="127.0.0.1",
        params={"stockList": "Naz100", "uptrendSignalMethod": "hma"},
    )
"""

import logging
import os
from typing import Dict, List

import jinja2

logger = logging.getLogger(__name__)

##############################################################################
# Template loader (singleton at module level)
##############################################################################

_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_TEMPLATES_DIR),
    autoescape=jinja2.select_autoescape(["html"]),
    # Keep undefined variables as empty string rather than raising.
    undefined=jinja2.Undefined,
)


##############################################################################
# Public API
##############################################################################


def build_holdings_html_report(
    holdings_rows: List[Dict[str, str]],
    trade_message: str,
    lifetime_profit: float,
    cumulative_cash_in: float,
    lifetime_profit_annualized: float,
    removed_tickers: List[str],
    added_tickers: List[str],
    edition: str,
    ip: str,
    params: dict,
) -> str:
    """Render the holdings HTML table and report footer.

    All per-row numeric values must be pre-formatted by the caller (using
    Python's ``format()`` built-in) and passed as strings in each row
    dict.  The function performs no numeric computation — it only hands
    the structured data to the Jinja2 template.

    Args:
        holdings_rows: List of dicts, one per holding.  Each dict must
            contain the following string-valued keys:

            - ``symbol``: 5-char formatted ticker (e.g. ``"AAPL "``)
            - ``shares``: 6-wide formatted share count
            - ``buy_price``: purchase price (6.2f)
            - ``purchase_cost``: ``shares * buy_price`` (6.2f)
            - ``cumu_purchase``: running cumulative purchase cost (6.2f)
            - ``current_price``: latest quote (6.2f)
            - ``profit_pct``: gain/loss percentage (6.2%)
            - ``value``: ``shares * current_price`` (6.2f)
            - ``cumu_value``: running cumulative current value (6.2f)
            - ``rank``: current signal rank (3d)
            - ``cluster``: cluster label string
            - ``sector``: sector name (empty string for CASH)
            - ``industry``: industry name (empty string for CASH)

        trade_message: Raw HTML string produced by
            ``calculateTrades()`` (may be ``"<br>"`` when no trades).
        lifetime_profit: Raw lifetime profit in dollars.
        cumulative_cash_in: Total cash deposited (denominator for
            percentage computations).
        lifetime_profit_annualized: Annualized lifetime profit value
            (formatted as a percentage against *cumulative_cash_in*).
        removed_tickers: Tickers removed from the stock-index list
            since the last update.
        added_tickers: Tickers added to the stock-index list since the
            last update.
        edition: Software edition string from ``GetEdition()``.
        ip: Host IP address string from ``GetIP()``.
        params: JSON configuration dict; must contain ``'stockList'``
            and ``'uptrendSignalMethod'``.

    Returns:
        Rendered HTML string suitable for use as ``message_text`` in
        ``run_pytaaa.py``.
    """
    try:
        template = _jinja_env.get_template("holdings_report.html.j2")
    except jinja2.TemplateNotFound:
        logger.error(
            "Jinja2 template not found at %s/holdings_report.html.j2; "
            "falling back to empty string.",
            _TEMPLATES_DIR,
        )
        return ""

    # Format lifetime profit values for the template.
    denom = float(cumulative_cash_in) if cumulative_cash_in else 1.0
    lifetime_profit_pct = format(lifetime_profit / denom, "6.1%")
    lifetime_profit_annualized_pct = format(
        lifetime_profit_annualized / denom, "6.1%"
    )

    context = {
        "holdings_rows": holdings_rows,
        "trade_message": trade_message,
        "lifetime_profit": str(lifetime_profit),
        "lifetime_profit_pct": lifetime_profit_pct,
        "lifetime_profit_annualized_pct": lifetime_profit_annualized_pct,
        "removed_tickers": removed_tickers,
        "added_tickers": added_tickers,
        "edition": edition,
        "ip": str(ip),
        "stock_list": params.get("stockList", ""),
        "uptrend_method": params.get("uptrendSignalMethod", ""),
    }

    html = template.render(context)
    logger.debug(
        "build_holdings_html_report: rendered %d bytes for %d holding rows.",
        len(html),
        len(holdings_rows),
    )
    return html
