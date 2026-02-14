"""Tests for Phase 5 modularization of TAfunctions.py

This test suite verifies:
1. All extracted functions can be imported from new ta.* modules
2. Functions produce identical results to original implementations
3. Backward compatibility is maintained
4. No circular imports exist
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestCircularImports:
    """Verify no circular imports exist in the new module structure."""
    
    def test_no_circular_imports(self):
        """All modules import without circular dependency errors."""
        # This test will fail during import if circular imports exist
        from functions.ta import utils
        from functions.ta import data_cleaning
        from functions.ta import moving_averages
        from functions.ta import channels
        from functions.ta import signal_generation
        from functions.ta import rolling_metrics
        from functions.ta import trend_analysis
        from functions.ta import ranking
        
        assert True  # If we get here, no circular imports


class TestBackwardCompatibility:
    """Verify backward compatibility - old imports still work."""
    
    def test_tafunctions_functions_still_exist(self):
        """Functions can still be imported from TAfunctions."""
        from functions import TAfunctions
        
        # Test a sampling of functions from each category
        assert hasattr(TAfunctions, 'strip_accents')
        assert hasattr(TAfunctions, 'SMA')
        assert hasattr(TAfunctions, 'SMA_2D')
        assert hasattr(TAfunctions, 'hma')
        assert hasattr(TAfunctions, 'interpolate')
        assert hasattr(TAfunctions, 'cleantobeginning')
        assert hasattr(TAfunctions, 'dpgchannel')
        assert hasattr(TAfunctions, 'percentileChannel')
        assert hasattr(TAfunctions, 'computeSignal2D')
        assert hasattr(TAfunctions, 'move_sharpe_2D')
        assert hasattr(TAfunctions, 'move_martin_2D')


class TestModuleImports:
    """Test that all functions can be imported from new modules."""
    
    def test_utils_imports(self):
        """Utils module functions import successfully."""
        from functions.ta.utils import strip_accents, normcorrcoef, nanrms
        assert callable(strip_accents)
        assert callable(normcorrcoef)
        assert callable(nanrms)
    
    def test_data_cleaning_imports(self):
        """Data cleaning module functions import successfully."""
        from functions.ta.data_cleaning import (
            interpolate, cleantobeginning, cleantoend, 
            clean_signal, cleanspikes, despike_2D
        )
        assert callable(interpolate)
        assert callable(cleantobeginning)
        assert callable(cleantoend)
        assert callable(clean_signal)
        assert callable(cleanspikes)
        assert callable(despike_2D)
    
    def test_moving_averages_imports(self):
        """Moving averages module functions import successfully."""
        from functions.ta.moving_averages import (
            SMA, SMA_2D, SMS, hma, hma_pd, SMA_filtered_2D,
            MoveMax, MoveMax_2D, MoveMin
        )
        assert callable(SMA)
        assert callable(SMA_2D)
        assert callable(SMS)
        assert callable(hma)
        assert callable(hma_pd)
        assert callable(SMA_filtered_2D)
        assert callable(MoveMax)
        assert callable(MoveMax_2D)
        assert callable(MoveMin)
    
    def test_channels_imports(self):
        """Channels module functions import successfully."""
        from functions.ta.channels import (
            percentileChannel, percentileChannel_2D,
            dpgchannel, dpgchannel_2D
        )
        assert callable(percentileChannel)
        assert callable(percentileChannel_2D)
        assert callable(dpgchannel)
        assert callable(dpgchannel_2D)
    
    def test_signal_generation_imports(self):
        """Signal generation module functions import successfully."""
        from functions.ta.signal_generation import computeSignal2D
        assert callable(computeSignal2D)
    
    def test_rolling_metrics_imports(self):
        """Rolling metrics module functions import successfully."""
        from functions.ta.rolling_metrics import (
            move_sharpe_2D, move_martin_2D, move_informationRatio
        )
        assert callable(move_sharpe_2D)
        assert callable(move_martin_2D)
        assert callable(move_informationRatio)


class TestFunctionEquivalence:
    """Test that extracted functions produce identical results to originals."""
    
    def test_strip_accents_equivalence(self):
        """Extracted strip_accents produces same results as original."""
        from functions.ta.utils import strip_accents as new_func
        from functions.TAfunctions import strip_accents as old_func
        
        test_strings = ["Café", "Zürich", "Ñoño", "Résumé", "plain text"]
        for s in test_strings:
            assert new_func(s) == old_func(s), f"Results differ for '{s}'"
    
    def test_SMA_equivalence(self):
        """Extracted SMA produces same results as original."""
        from functions.ta.moving_averages import SMA as new_func
        from functions.TAfunctions import SMA as old_func
        
        data = np.array([100.0, 102.0, 101.0, 105.0, 103.0, 106.0, 104.0, 108.0])
        periods = 3
        
        result_new = new_func(data, periods)
        result_old = old_func(data, periods)
        
        np.testing.assert_array_almost_equal(result_new, result_old)
    
    def test_SMA_2D_equivalence(self):
        """Extracted SMA_2D produces same results as original."""
        from functions.ta.moving_averages import SMA_2D as new_func
        from functions.TAfunctions import SMA_2D as old_func
        
        np.random.seed(42)
        data = np.random.rand(5, 20) * 100 + 100  # 5 stocks, 20 days
        periods = 5
        
        result_new = new_func(data, periods)
        result_old = old_func(data, periods)
        
        np.testing.assert_array_almost_equal(result_new, result_old)
    
    def test_cleantobeginning_equivalence(self):
        """Extracted cleantobeginning produces same results as original."""
        from functions.ta.data_cleaning import cleantobeginning as new_func
        from functions.TAfunctions import cleantobeginning as old_func
        
        data = np.array([np.nan, np.nan, 100.0, 101.0, np.nan, 102.0])
        
        result_new = new_func(data)
        result_old = old_func(data)
        
        np.testing.assert_array_equal(result_new, result_old)
    
    def test_interpolate_equivalence(self):
        """Extracted interpolate produces same results as original."""
        from functions.ta.data_cleaning import interpolate as new_func
        from functions.TAfunctions import interpolate as old_func
        
        data = np.array([np.nan, np.nan, 100.0, np.nan, np.nan, 105.0, np.nan, 106.0])
        
        result_new = new_func(data)
        result_old = old_func(data)
        
        np.testing.assert_array_almost_equal(result_new, result_old)
    
    def test_dpgchannel_equivalence(self):
        """Extracted dpgchannel produces same results as original."""
        from functions.ta.channels import dpgchannel as new_func
        from functions.TAfunctions import dpgchannel as old_func
        
        data = np.array([100.0, 102.0, 101.0, 105.0, 103.0, 106.0, 104.0, 108.0, 107.0, 110.0])
        
        result_new = new_func(data, 2, 5, 1)
        result_old = old_func(data, 2, 5, 1)
        
        np.testing.assert_array_almost_equal(result_new[0], result_old[0])  # minchannel
        np.testing.assert_array_almost_equal(result_new[1], result_old[1])  # maxchannel


class TestModuleFunctionality:
    """Test that functions work correctly with realistic data."""
    
    def test_SMA_basic_calculation(self):
        """SMA calculates correct moving average."""
        from functions.ta.moving_averages import SMA
        
        data = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        result = SMA(data, 3)
        
        # Last value should be average from max(0, 4-3)=1 to 4+1=5
        # So it's mean of indices 1,2,3,4 = mean of [102, 104, 106, 108] = 105
        assert pytest.approx(result[-1], rel=1e-6) == 105.0
    
    def test_hma_shape_preservation(self):
        """HMA preserves input shape."""
        from functions.ta.moving_averages import hma
        
        data = np.random.rand(5, 100) * 100 + 100  # 5 stocks, 100 days
        result = hma(data, 20)
        
        assert result.shape == data.shape
    
    def test_computeSignal2D_SMAs(self):
        """computeSignal2D generates signals with SMAs method."""
        from functions.ta.signal_generation import computeSignal2D
        
        np.random.seed(42)
        adjClose = np.cumsum(np.random.randn(10, 100) * 0.5 + 0.1, axis=1) + 100
        gainloss = np.ones_like(adjClose)
        gainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
        
        params = {
            'MA1': 50,
            'MA2': 10,
            'MA2offset': 10,
            'MA2factor': 1.0,
            'uptrendSignalMethod': 'SMAs',
            'narrowDays': [5, 20],
            'mediumDays': [10, 30],
            'wideDays': [20, 60],
            'lowPct': 20.0,
            'hiPct': 80.0
        }
        
        signal = computeSignal2D(adjClose, gainloss, params)
        
        # Signal should be 0 or 1
        assert np.all((signal == 0) | (signal == 1))
        assert signal.shape == adjClose.shape
    
    def test_move_sharpe_2D_calculation(self):
        """move_sharpe_2D calculates risk-adjusted returns."""
        from functions.ta.rolling_metrics import move_sharpe_2D
        
        np.random.seed(42)
        adjClose = np.cumsum(np.random.randn(5, 100) * 0.5 + 0.1, axis=1) + 100
        gainloss = np.ones_like(adjClose)
        gainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
        
        result = move_sharpe_2D(adjClose, gainloss, 60)
        
        # Should return finite values
        assert np.all(np.isfinite(result[:, 60:]))
        assert result.shape == adjClose.shape


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_SMA_short_period(self):
        """SMA handles period=1 correctly."""
        from functions.ta.moving_averages import SMA
        
        data = np.array([100.0, 102.0, 104.0])
        result = SMA(data, 1)
        
        # Period 1 means averaging from max(0,i-1) to i+1
        # So result[0] = mean([100]), result[1] = mean([100,102]), result[2] = mean([102,104])
        expected = np.array([100.0, 101.0, 103.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_cleantobeginning_all_nan(self):
        """cleantobeginning handles all-NaN array."""
        from functions.ta.data_cleaning import cleantobeginning
        
        data = np.array([np.nan, np.nan, np.nan])
        result = cleantobeginning(data)
        
        # Should return all NaN if no valid values
        assert np.all(np.isnan(result))
    
    def test_interpolate_no_nan(self):
        """interpolate handles array with no NaN values."""
        from functions.ta.data_cleaning import interpolate
        
        data = np.array([100.0, 101.0, 102.0, 103.0])
        result = interpolate(data)
        
        # Should return unchanged array
        np.testing.assert_array_equal(result, data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
