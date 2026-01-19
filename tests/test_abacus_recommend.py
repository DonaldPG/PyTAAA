#!/usr/bin/env python3

"""Unit tests for abacus_recommend module."""

import pytest
import tempfile
import pickle
from datetime import date
from unittest.mock import Mock, MagicMock
from functions.abacus_recommend import (
    DateHelper, ConfigurationHelper, RecommendationDisplay
)


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


class TestConfigurationHelper:
    """Smoke tests for ConfigurationHelper class."""
    
    def test_get_recommendation_lookbacks_from_config(self):
        """Test getting lookbacks from config defaults."""
        config = {
            'recommendation_mode': {
                'default_lookbacks': [50, 150, 250]
            }
        }
        
        lookbacks = ConfigurationHelper.get_recommendation_lookbacks(None, config)
        
        assert lookbacks == [50, 150, 250]
    
    def test_get_recommendation_lookbacks_from_string(self):
        """Test parsing lookbacks from comma-separated string."""
        config = {'recommendation_mode': {'default_lookbacks': [50]}}
        
        lookbacks = ConfigurationHelper.get_recommendation_lookbacks("25,100,200", config)
        
        assert lookbacks == [25, 100, 200]
    
    def test_load_best_lookbacks_from_nonexistent_state(self):
        """Test loading from missing state file returns None."""
        result = ConfigurationHelper.load_best_lookbacks_from_state("nonexistent_file.pkl")
        
        assert result is None
    
    def test_load_best_lookbacks_from_valid_state(self):
        """Test loading lookbacks from valid state file."""
        # Create temporary state file
        state = {
            'best_params': {
                'lookbacks': [55, 157, 174]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(state, f)
            temp_file = f.name
        
        try:
            result = ConfigurationHelper.load_best_lookbacks_from_state(temp_file)
            assert result == [55, 157, 174]
        finally:
            import os
            os.unlink(temp_file)
    
    def test_ensure_config_defaults_adds_missing_sections(self):
        """Test that ensure_config_defaults adds missing sections."""
        config = {}
        
        ConfigurationHelper.ensure_config_defaults(config)
        
        assert 'model_selection' in config
        assert 'performance_metrics' in config['model_selection']
        assert 'metric_blending' in config
        assert 'recommendation_mode' in config
    
    def test_ensure_config_defaults_preserves_existing(self):
        """Test that existing config values are preserved."""
        config = {
            'model_selection': {
                'performance_metrics': {
                    'sharpe_ratio_weight': 2.0,
                    'sortino_ratio_weight': 1.5,
                    'max_drawdown_weight': 1.0,
                    'avg_drawdown_weight': 1.0,
                    'annualized_return_weight': 1.0
                }
            }
        }
        
        ConfigurationHelper.ensure_config_defaults(config)
        
        assert config['model_selection']['performance_metrics']['sharpe_ratio_weight'] == 2.0


class TestRecommendationDisplay:
    """Smoke tests for RecommendationDisplay class."""
    
    def test_instantiation(self):
        """Test that RecommendationDisplay can be instantiated."""
        mock_mc = Mock()
        mock_mc.CENTRAL_VALUES = {'sharpe_ratio': 1.0}
        mock_mc.STD_VALUES = {'sharpe_ratio': 0.5}
        
        display = RecommendationDisplay(mock_mc)
        
        assert display.monte_carlo == mock_mc
    
    def test_display_parameters_summary_runs_without_error(self, capsys):
        """Test that display_parameters_summary executes without errors."""
        import numpy as np
        
        mock_mc = Mock()
        mock_mc.CENTRAL_VALUES = {
            'sharpe_ratio': 1.0,
            'annual_return': 0.10,
            'max_drawdown': -0.15
        }
        mock_mc.STD_VALUES = {
            'sharpe_ratio': 0.5,
            'annual_return': 0.05,
            'max_drawdown': 0.10
        }
        
        display = RecommendationDisplay(mock_mc)
        portfolio = np.array([10000.0, 10500.0, 11000.0] + [11000.0] * 249)  # 252 days
        
        # Should not raise exception
        display.display_parameters_summary([50, 150], portfolio)
        
        captured = capsys.readouterr()
        assert "PARAMETERS SUMMARY" in captured.out
        assert "Lookback Periods" in captured.out



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
