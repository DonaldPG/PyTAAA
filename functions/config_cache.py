"""Thread-safe JSON configuration file cache.

Provides a singleton ConfigCache that parses each JSON config file
only once per process, reducing repeated disk I/O during pipeline runs
and Monte Carlo worker execution.

Usage:
    from functions.config_cache import config_cache

    cfg = config_cache.get("path/to/config.json")   # loaded once
    cfg = config_cache.get("path/to/config.json")   # returned from cache

    # After writing the JSON file, invalidate so next get re-reads disk:
    config_cache.invalidate("path/to/config.json")
"""

import json
import threading
from pathlib import Path
from typing import Any


class ConfigCache:
    """Singleton cache for parsed JSON configuration files.

    Only one instance is ever created (singleton pattern). All public
    methods are protected by a reentrant lock, making concurrent calls
    from Monte Carlo worker threads safe.
    """

    _instance: "ConfigCache | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ConfigCache":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._cache: dict[str, dict[str, Any]] = {}
        return cls._instance

    def get(self, json_path: str) -> dict[str, Any]:
        """Return parsed JSON, loading from disk only on first access.

        The resolved absolute path is used as the cache key so that
        relative paths and symlinks map to the same cached entry.

        Args:
            json_path: Path to the JSON configuration file.

        Returns:
            Parsed JSON configuration dictionary.

        Raises:
            FileNotFoundError: If json_path does not exist.
            json.JSONDecodeError: If the file contains malformed JSON.
        """
        key = str(Path(json_path).resolve())
        with self._lock:
            if key not in self._cache:
                with open(key) as f:
                    self._cache[key] = json.load(f)
            return self._cache[key]

    def invalidate(self, json_path: str | None = None) -> None:
        """Invalidate one entry or clear the entire cache.

        Call this after writing changes to a JSON config file so that
        the next get() call re-reads the updated contents from disk.

        Args:
            json_path: Path to invalidate. If None, clears all entries.
        """
        with self._lock:
            if json_path is not None:
                self._cache.pop(str(Path(json_path).resolve()), None)
            else:
                self._cache.clear()


##############################################################################
# Module-level singleton — import this directly in other modules.
##############################################################################
config_cache = ConfigCache()
