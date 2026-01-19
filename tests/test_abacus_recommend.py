#!/usr/bin/env python3

"""Unit tests for abacus_recommend module."""

import pytest
from datetime import date
from functions.abacus_recommend import DateHelper


class TestDateHelper:
    """Tests for DateHelper class."""

    def test_get_first_weekday_of_month_starts_monday(self):
        """Test month starting with Monday."""
        # January 2024 starts on Monday (Jan 1, 2024)
        result = DateHelper.get_first_weekday_of_month(date(2024, 1, 15))
        assert result == date(2024, 1, 1)

    def test_get_first_weekday_of_month_starts_saturday(self):
        """Test month starting with Saturday."""
        # June 2024 starts on Saturday (first weekday is June 3, Monday)
        result = DateHelper.get_first_weekday_of_month(date(2024, 6, 15))
        assert result == date(2024, 6, 3)

    def test_get_first_weekday_of_month_starts_sunday(self):
        """Test month starting with Sunday."""
        # December 2024 starts on Sunday (first weekday is Dec 2, Monday)
        result = DateHelper.get_first_weekday_of_month(date(2024, 12, 15))
        assert result == date(2024, 12, 2)

    def test_get_first_weekday_of_month_starts_friday(self):
        """Test month starting with Friday."""
        # March 2024 starts on Friday (first weekday is March 1)
        result = DateHelper.get_first_weekday_of_month(date(2024, 3, 15))
        assert result == date(2024, 3, 1)

    def test_get_first_weekday_january_2026(self):
        """Test January 2026 (starts Wednesday)."""
        # January 1, 2026 is Wednesday
        result = DateHelper.get_first_weekday_of_month(date(2026, 1, 19))
        assert result == date(2026, 1, 1)

    def test_get_recommendation_dates_with_date_string(self):
        """Test getting recommendation dates with specific date."""
        dates, target, first_weekday = DateHelper.get_recommendation_dates("2026-01-19")
        
        assert target == date(2026, 1, 19)
        assert first_weekday == date(2026, 1, 1)
        assert len(dates) == 2
        assert date(2026, 1, 19) in dates
        assert date(2026, 1, 1) in dates

    def test_get_recommendation_dates_when_target_is_first_weekday(self):
        """Test when target date is already the first weekday."""
        dates, target, first_weekday = DateHelper.get_recommendation_dates("2026-01-01")
        
        assert target == date(2026, 1, 1)
        assert first_weekday == date(2026, 1, 1)
        assert len(dates) == 1  # Only one date since they're the same
        assert date(2026, 1, 1) in dates

    def test_get_recommendation_dates_none_uses_today(self):
        """Test that None uses today's date."""
        dates, target, first_weekday = DateHelper.get_recommendation_dates(None)
        
        today = date.today()
        assert target == today
        assert target in dates

    def test_find_closest_trading_date_exact_match(self):
        """Test finding exact match."""
        available = [date(2026, 1, 15), date(2026, 1, 16), date(2026, 1, 17)]
        target = date(2026, 1, 16)
        
        closest, diff = DateHelper.find_closest_trading_date(target, available)
        
        assert closest == date(2026, 1, 16)
        assert diff == 0

    def test_find_closest_trading_date_one_day_before(self):
        """Test finding closest when target is between dates."""
        available = [date(2026, 1, 15), date(2026, 1, 17), date(2026, 1, 20)]
        target = date(2026, 1, 16)
        
        closest, diff = DateHelper.find_closest_trading_date(target, available)
        
        # Should find Jan 15 or Jan 17 (both 1 day away, will return first found)
        assert diff == 1
        assert closest in [date(2026, 1, 15), date(2026, 1, 17)]

    def test_find_closest_trading_date_multiple_days_away(self):
        """Test finding closest when several days away."""
        available = [date(2026, 1, 10), date(2026, 1, 20), date(2026, 1, 30)]
        target = date(2026, 1, 19)
        
        closest, diff = DateHelper.find_closest_trading_date(target, available)
        
        assert closest == date(2026, 1, 20)
        assert diff == 1

    def test_find_closest_trading_date_empty_list(self):
        """Test with empty date list."""
        closest, diff = DateHelper.find_closest_trading_date(date(2026, 1, 19), [])
        
        assert closest is None
        assert diff == 0

    def test_find_closest_trading_date_single_date(self):
        """Test with single available date."""
        available = [date(2026, 1, 15)]
        target = date(2026, 1, 20)
        
        closest, diff = DateHelper.find_closest_trading_date(target, available)
        
        assert closest == date(2026, 1, 15)
        assert diff == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
