"""Backward-compatible re-exports from the split config modules.

All names previously defined in this module are still importable here.
New code should import from the specific submodule directly:

    from functions.config_accessors import get_json_params
    from functions.config_validators import validate_model_choices
    from functions.config_loader import parse_pytaaa_status
"""

from functions.config_loader import (
    from_config_file,
    parse_pytaaa_status,
)
from functions.config_validators import (
    validate_model_choices,
    validate_required_keys,
)
from functions.config_accessors import (
    get_json_params,
    get_json_ftp_params,
    get_symbols_file,
    get_performance_store,
    get_webpage_store,
    get_web_output_dir,
    get_central_std_values,
    get_holdings,
    get_json_status,
    get_status,
    compute_long_hold_signal,
    put_status,
    GetIP,
    GetEdition,
)

__all__ = [
    "from_config_file",
    "parse_pytaaa_status",
    "validate_model_choices",
    "validate_required_keys",
    "get_json_params",
    "get_json_ftp_params",
    "get_symbols_file",
    "get_performance_store",
    "get_webpage_store",
    "get_web_output_dir",
    "get_central_std_values",
    "get_holdings",
    "get_json_status",
    "get_status",
    "compute_long_hold_signal",
    "put_status",
    "GetIP",
    "GetEdition",
]
