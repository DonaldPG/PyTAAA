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
