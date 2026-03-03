"""Phase F: ConfigCache singleton tests.

Verifies that the ConfigCache:
- Returns the identical object on repeated calls (no re-parse)
- Reloads from disk after invalidation
- Handles concurrent reads without errors or data corruption
"""
import json
import threading
import pytest


def test_cache_returns_same_object(tmp_path):
    """Two consecutive get() calls must return the identical dict object."""
    from functions.config_cache import ConfigCache

    cache = ConfigCache()
    cache.invalidate()  # Start with a clean slate.

    cfg = tmp_path / "test.json"
    cfg.write_text('{"key": "value"}')

    first = cache.get(str(cfg))
    second = cache.get(str(cfg))

    assert first is second, (
        "Expected both calls to return the same cached object."
    )


def test_invalidate_reloads(tmp_path):
    """After invalidation, get() must reflect the new file contents."""
    from functions.config_cache import ConfigCache

    cache = ConfigCache()

    cfg = tmp_path / "reload.json"
    cfg.write_text('{"v": 1}')
    cache.invalidate(str(cfg))          # Ensure no stale entry.

    first = cache.get(str(cfg))
    assert first["v"] == 1

    cfg.write_text('{"v": 2}')          # Overwrite file on disk.
    cache.invalidate(str(cfg))          # Evict stale cache entry.

    second = cache.get(str(cfg))
    assert second["v"] == 2, (
        "Cache should reload the updated file after invalidation."
    )


def test_invalidate_all_clears_cache(tmp_path):
    """Calling invalidate() with no argument clears the entire cache."""
    from functions.config_cache import ConfigCache

    cache = ConfigCache()

    cfg_a = tmp_path / "a.json"
    cfg_b = tmp_path / "b.json"
    cfg_a.write_text('{"a": 1}')
    cfg_b.write_text('{"b": 2}')

    a1 = cache.get(str(cfg_a))
    b1 = cache.get(str(cfg_b))

    cache.invalidate()                  # Clear everything.

    a2 = cache.get(str(cfg_a))
    b2 = cache.get(str(cfg_b))

    assert a1 is not a2, "After full invalidation, a new object must be loaded."
    assert b1 is not b2, "After full invalidation, a new object must be loaded."
    assert a2["a"] == 1
    assert b2["b"] == 2


def test_thread_safe(tmp_path):
    """20 concurrent threads reading the same config must not error."""
    from functions.config_cache import ConfigCache

    cache = ConfigCache()
    cfg = tmp_path / "thread.json"
    cfg.write_text('{"x": 42}')
    cache.invalidate(str(cfg))

    results: list[int] = []
    errors: list[Exception] = []

    def reader() -> None:
        try:
            results.append(cache.get(str(cfg))["x"])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(results) == 20
    assert all(r == 42 for r in results)


def test_singleton_identity():
    """Two ConfigCache() calls must return the exact same instance."""
    from functions.config_cache import ConfigCache

    cache_a = ConfigCache()
    cache_b = ConfigCache()

    assert cache_a is cache_b, "ConfigCache must be a singleton."


def test_module_level_instance_is_singleton():
    """The module-level config_cache must be the same instance as ConfigCache()."""
    from functions.config_cache import ConfigCache, config_cache

    assert config_cache is ConfigCache(), (
        "Module-level config_cache must be the singleton instance."
    )
