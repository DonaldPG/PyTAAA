#!/usr/bin/env python3
"""Legacy entry point for PyTAAA system.

**DEPRECATED:** This file is deprecated and will be removed in a future release.

Use `pytaaa_main.py` with JSON configuration instead:
    uv run python pytaaa_main.py --json your_config.json

This file is kept for backward compatibility only. When executed, it displays
a deprecation warning and redirects to the modern JSON-based entry point using
`pytaaa_generic.json` as the default configuration.

For more information, see:
- pytaaa_main.py - Modern JSON-based entry point
- docs/PYTAAA_SYSTEM_SUMMARY_AND_JSON_GUIDE.md - JSON configuration guide
- docs/ARCHITECTURE.md - System architecture documentation
"""

import warnings
import sys
import os


def main():
    """Main function with deprecation warning and redirect."""
    warnings.warn(
        "\n\n"
        "=" * 70 + "\n"
        "DEPRECATION WARNING: PyTAAA.py is deprecated\n"
        "=" * 70 + "\n"
        "This legacy entry point will be removed in a future release.\n"
        "\n"
        "Please use the modern JSON-based entry point instead:\n"
        "    uv run python pytaaa_main.py --json your_config.json\n"
        "\n"
        "Redirecting to pytaaa_main.py with pytaaa_generic.json...\n"
        "=" * 70 + "\n",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Redirect to modern entry point
    try:
        from pytaaa_main import main as modern_main
        
        # Set up arguments for JSON-based entry point
        default_json = 'pytaaa_generic.json'
        if not os.path.exists(default_json):
            print(f"\nERROR: Default configuration file '{default_json}' not found.")
            print("Please create a JSON configuration file or specify one:")
            print("    uv run python pytaaa_main.py --json your_config.json")
            sys.exit(1)
        
        # Update sys.argv to pass JSON file to modern entry point
        sys.argv = ['pytaaa_main.py', '--json', default_json]
        
        # Call modern entry point
        modern_main()
        
    except ImportError as e:
        print(f"\nERROR: Could not import pytaaa_main: {e}")
        print("Please ensure pytaaa_main.py exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
