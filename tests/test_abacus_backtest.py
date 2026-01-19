#!/usr/bin/env python3

"""Unit tests for abacus_backtest module."""

import pytest
from unittest.mock import Mock
from functions.abacus_backtest import BacktestDataLoader


class TestBacktestDataLoader:
    """Smoke tests for BacktestDataLoader class."""
    
    def test_instantiation_without_config(self):
        """Test that BacktestDataLoader can be instantiated without config."""
        loader = BacktestDataLoader()
        
        assert loader.config == {}
    
    def test_instantiation_with_config(self):
        """Test that BacktestDataLoader can be instantiated with config."""
        config = {'models': {'base_folder': '/test/path'}}
        loader = BacktestDataLoader(config)
        
        assert loader.config == config
    
    def test_build_model_paths_backtested_format(self):
        """Test building model paths with backtested format."""
        config = {
            'models': {
                'base_folder': '/Users/test/pyTAAA_data',
                'model_choices': {
                    'cash': '',
                    'naz100_pine': '{base_folder}/naz100_pine/data_store/{data_file}',
                    'sp500_hma': '{base_folder}/sp500_hma/data_store/{data_file}'
                }
            }
        }
        
        loader = BacktestDataLoader(config)
        paths = loader.build_model_paths('backtested', '/some/json/path')
        
        assert paths['cash'] == ''
        assert 'pyTAAAweb_backtestPortfolioValue.params' in paths['naz100_pine']
        assert 'pyTAAAweb_backtestPortfolioValue.params' in paths['sp500_hma']
        assert '/Users/test/pyTAAA_data' in paths['naz100_pine']
    
    def test_build_model_paths_actual_format(self):
        """Test building model paths with actual format."""
        config = {
            'models': {
                'base_folder': '/Users/test/pyTAAA_data',
                'model_choices': {
                    'cash': '',
                    'naz100_pine': '{base_folder}/naz100_pine/data_store/{data_file}'
                }
            }
        }
        
        loader = BacktestDataLoader(config)
        paths = loader.build_model_paths('actual', '/some/json/path')
        
        assert 'PyTAAA_status.params' in paths['naz100_pine']
    
    def test_build_model_paths_legacy_mode(self):
        """Test building model paths in legacy mode (no models in config)."""
        config = {}
        
        loader = BacktestDataLoader(config)
        paths = loader.build_model_paths('backtested', None)
        
        assert 'cash' in paths
        assert 'naz100_pine' in paths
        assert 'sp500_hma' in paths
        assert paths['cash'] == ''
    
    def test_validate_model_paths_with_cash(self):
        """Test validation handles cash model (empty path) correctly."""
        loader = BacktestDataLoader()
        paths = {'cash': '', 'naz100_pine': '/nonexistent/path.params'}
        
        validated = loader.validate_model_paths(paths)
        
        assert validated['cash'] == ''
        assert 'nonexistent' in validated['naz100_pine']
    
    def test_validate_model_paths_expands_user_home(self):
        """Test that validation expands ~ in paths."""
        loader = BacktestDataLoader()
        paths = {'model1': '~/test/path.params'}
        
        validated = loader.validate_model_paths(paths)
        
        # Should expand ~ to absolute path
        assert '~' not in validated['model1']
        assert validated['model1'].startswith('/')
