"""Validation helpers for PyTAAA configuration values.

This module checks that configuration values are internally consistent
and that referenced files and directories exist on disk. It has no
dependency on the JSON config cache, ``TAfunctions``, or any I/O helper
beyond ``os``.

Functions:
    validate_model_choices: Check that model JSON paths resolve to
        existing files.
    validate_required_keys: Assert all required keys are present in a
        config dict (raises ``KeyError`` with a clear message otherwise).
    validate_stock_weight_method: Raise ``ValueError`` if the supplied
        stock-weighting method name is not one of the three recognised
        values.
"""

import os
from typing import Dict


def validate_model_choices(model_choices: Dict[str, str]) -> Dict[str, bool]:
    """Validate file paths in the ``model_choices`` mapping.

    Checks that each path in ``model_choices`` points to an existing file.
    The cash model conventionally has an empty string or ``None`` as its
    path and is treated as always valid.

    Args:
        model_choices: Mapping of model name to file path, e.g.
            ``{"naz100_pine": "/path/to/config.json", "cash": ""}``.

    Returns:
        Mapping of model name to ``True`` (path exists or cash model)
        or ``False`` (path given but file not found).

    Example:
        >>> results = validate_model_choices(
        ...     {"naz100_pine": "/valid/path.json", "cash": ""}
        ... )
        >>> results["naz100_pine"]
        True
        >>> results["cash"]
        True
    """
    validation_results: Dict[str, bool] = {}
    for model, path in model_choices.items():
        if path:
            validation_results[model] = os.path.exists(path)
        else:
            # Cash model has no associated config file.
            validation_results[model] = True
    return validation_results


def validate_required_keys(
    config: dict,
    required_keys: list,
    context: str = "configuration",
) -> None:
    """Assert that all required keys are present in a config dict.

    Args:
        config: The configuration dictionary to check.
        required_keys: List of key names that must be present.
        context: Human-readable label for the config source, used in
            the error message (e.g. ``"Valuation section"``).

    Raises:
        KeyError: If any required key is missing, listing all missing
            keys in the error message.

    Example:
        >>> validate_required_keys(
        ...     {"MA1": 10, "MA2": 20},
        ...     ["MA1", "MA2", "MA3"],
        ...     context="Valuation",
        ... )
        KeyError: "Missing keys in Valuation: ['MA3']"
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing keys in {context}: {missing}")


#############
# Stock-weighting method validation
#############

_VALID_STOCK_WEIGHT_METHODS = frozenset({
    "delta_rank_sharpe_weight",
    "equal_weight",
    "abs_sharpe_weight",
})


def validate_stock_weight_method(method: str) -> None:
    """Raise ``ValueError`` if *method* is not a recognised weighting method.

    Three stock-weighting methods are supported:

    ``"delta_rank_sharpe_weight"``
        Momentum-of-momentum delta-rank with inverse-Sharpe ratio
        weights and soft signal suppression (default; Method A).
    ``"equal_weight"``
        Delta-rank selection with equal allocation across selected
        stocks (Method B).
    ``"abs_sharpe_weight"``
        Hard binary signal gate with absolute Sharpe ratio ranking
        (Method C; the current worktree2 implementation).

    Args:
        method: Value of the ``stockWeightMethod`` config key.

    Raises:
        ValueError: If *method* is not one of the three valid strings.

    Example:
        >>> validate_stock_weight_method("equal_weight")  # OK
        >>> validate_stock_weight_method("bogus")
        ValueError: Invalid stockWeightMethod 'bogus'. ...
    """
    if method not in _VALID_STOCK_WEIGHT_METHODS:
        valid = sorted(_VALID_STOCK_WEIGHT_METHODS)
        raise ValueError(
            f"Invalid stockWeightMethod {method!r}. "
            f"Must be one of: {valid}"
        )
