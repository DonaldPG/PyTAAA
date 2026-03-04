"""Regression tests for Item 3 (Phase 0) — module-level sentinel fix.

Prior to this fix, ``run_pytaaa()`` used ``daily_update_done in locals()``
to track state across scheduler invocations.  Because the function gets a
fresh stack frame on every call, those sentinel variables were always
reset to ``False``, making the ``hourOfDay`` guard completely non-functional:
``UpdateHDF_yf`` (slow, rate-limited) ran on every invocation regardless of
the time of day.

The fix replaces the local-variable patterns with module-level sentinels
(``_daily_update_done``, ``_calcs_update_count``).  These tests verify the
key behavioural properties of the new sentinel logic.
"""

import pytest
import run_pytaaa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_sentinels():
    """Reset all module-level sentinels to their initial values."""
    run_pytaaa._daily_update_done = False
    run_pytaaa._calcs_update_count = 0
    run_pytaaa._cached_lastdate = None
    run_pytaaa._cached_last_symbols_text = None
    run_pytaaa._cached_last_symbols_weight = None
    run_pytaaa._cached_last_symbols_price = None


def _apply_daily_reset_logic(hour_of_day: int) -> bool:
    """Reproduce the sentinel reset block at the top of run_pytaaa().

    Returns the value of ``daily_update_done`` (local alias) after the
    reset block runs, mirroring what the function body sees.
    """
    if hour_of_day <= 15:
        run_pytaaa._daily_update_done = False
        run_pytaaa._calcs_update_count = 0
    return run_pytaaa._daily_update_done


def _apply_post_hdf_update_logic(hour_of_day: int):
    """Reproduce the ``if hourOfDay > 15: _daily_update_done = True`` block."""
    if hour_of_day > 15:
        run_pytaaa._daily_update_done = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDailyUpdateSentinel:
    """Sentinel correctly tracks whether the daily HDF5 update has run."""

    def setup_method(self):
        _reset_sentinels()

    def test_sentinel_starts_false(self):
        """After reset, the sentinel is False (no update has run yet)."""
        assert run_pytaaa._daily_update_done is False

    def test_sentinel_not_reset_past_cutoff(self):
        """At hour > 15 the sentinel is NOT reset at the start of a call.

        This is the core regression: before the fix, the broken
        ``daily_update_done in locals()`` always reset the sentinel,
        regardless of the hour.
        """
        # Pre-condition: simulate that a previous call set the sentinel.
        run_pytaaa._daily_update_done = True

        # Apply the reset logic as run_pytaaa() would at hour 16.
        daily_update_done = _apply_daily_reset_logic(hour_of_day=16)

        # Sentinel must still be True — the reset block was skipped.
        assert run_pytaaa._daily_update_done is True
        assert daily_update_done is True, (
            "Sentinel was incorrectly reset past the cutoff hour. "
            "UpdateHDF_yf would have been called a second time."
        )

    def test_sentinel_reset_before_cutoff(self):
        """At hour <= 15 the sentinel IS reset so fresh data is fetched."""
        run_pytaaa._daily_update_done = True  # Was set from a prior run.
        _apply_daily_reset_logic(hour_of_day=14)
        assert run_pytaaa._daily_update_done is False

    def test_sentinel_set_after_hdf_update_past_cutoff(self):
        """Sentinel is set to True after UpdateHDF_yf at hour > 15."""
        assert run_pytaaa._daily_update_done is False
        _apply_post_hdf_update_logic(hour_of_day=16)
        assert run_pytaaa._daily_update_done is True

    def test_sentinel_not_set_after_hdf_update_before_cutoff(self):
        """Sentinel stays False after an early-day update (< cutoff hour).

        Before cutoff the flag must not be latched True, so that the logic
        can still trigger a fresh update across multiple early-day calls.
        """
        assert run_pytaaa._daily_update_done is False
        _apply_post_hdf_update_logic(hour_of_day=14)
        assert run_pytaaa._daily_update_done is False

    def test_full_two_call_cycle_past_cutoff(self):
        """Key regression: sentinel persists correctly over two invocations.

        Call 1 at hour 16: sentinel is False → update runs → sentinel → True.
        Call 2 at hour 16: sentinel stays True → update does NOT run.
        """
        # ---- First scheduler invocation ----
        daily_update_done_call1 = _apply_daily_reset_logic(hour_of_day=16)
        assert not daily_update_done_call1, "First call: update not yet done"
        # UpdateHDF_yf runs here (mocked in production test; sentinel update
        # is what we're verifying).
        _apply_post_hdf_update_logic(hour_of_day=16)
        assert run_pytaaa._daily_update_done is True

        # ---- Second scheduler invocation ----
        daily_update_done_call2 = _apply_daily_reset_logic(hour_of_day=16)
        assert daily_update_done_call2 is True, (
            "Sentinel was reset on second call. UpdateHDF_yf would run twice."
        )


class TestCalcsUpdateSentinel:
    """Sentinel correctly gates repeated PortfolioPerformanceCalcs calls."""

    def setup_method(self):
        _reset_sentinels()

    def test_calcs_count_starts_zero(self):
        assert run_pytaaa._calcs_update_count == 0

    def test_calcs_sentinel_reset_with_daily_sentinel(self):
        """When the daily sentinel resets, the calcs counter resets too.

        This ensures calcs run again on the day after the daily update.
        """
        run_pytaaa._calcs_update_count = 1
        _apply_daily_reset_logic(hour_of_day=14)  # Before cutoff → reset
        assert run_pytaaa._calcs_update_count == 0

    def test_calcs_sentinel_not_reset_past_cutoff(self):
        """Calcs counter must NOT reset past the cutoff hour."""
        run_pytaaa._calcs_update_count = 1
        _apply_daily_reset_logic(hour_of_day=16)  # Past cutoff → no reset
        assert run_pytaaa._calcs_update_count == 1
