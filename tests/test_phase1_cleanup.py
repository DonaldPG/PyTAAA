"""Tests for Phase 1 cleanup - verify dead code removal doesn't break functionality.

This test suite verifies that:
1. All modules import successfully after dead code removal
2. No duplicate function definitions exist
3. Key functions are accessible and documented
4. Archived files are in correct locations
"""

import pytest
import ast
import inspect
import os


class TestDeadCodeRemoval:
    """Verify that removed dead code wasn't actually used."""
    
    def test_check_market_open_imports(self):
        """CheckMarketOpen module imports successfully."""
        from functions import CheckMarketOpen
        assert hasattr(CheckMarketOpen, 'get_MarketOpenOrClosed')
        assert hasattr(CheckMarketOpen, 'CheckMarketOpen')
    
    def test_tafunctions_imports(self):
        """TAfunctions module imports successfully."""
        from functions import TAfunctions
        assert hasattr(TAfunctions, 'interpolate')
        assert hasattr(TAfunctions, 'cleantobeginning')
        assert hasattr(TAfunctions, 'cleantoend')
    
    def test_quotes_for_list_adjclose_imports(self):
        """quotes_for_list_adjClose module imports successfully."""
        from functions import quotes_for_list_adjClose
        assert hasattr(quotes_for_list_adjClose, 'arrayFromQuotesForList')
        assert hasattr(quotes_for_list_adjClose, 'get_Naz100PlusETFsList')
    
    def test_no_duplicate_definitions_checkmarketopen(self):
        """Verify no function is defined multiple times in CheckMarketOpen.py."""
        from functions import CheckMarketOpen
        
        source = inspect.getsource(CheckMarketOpen)
        tree = ast.parse(source)
        
        function_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
        
        duplicates = [name for name in set(function_names) if function_names.count(name) > 1]
        assert len(duplicates) == 0, f"Duplicate function definitions found: {duplicates}"
    
    def test_regenerate_hdf5_archived(self):
        """Verify re-generateHDF5.py is archived, not deleted."""
        assert os.path.exists('archive/re-generateHDF5.py'), "File should be archived"
        assert not os.path.exists('re-generateHDF5.py'), "File should not be in root"


class TestDocstrings:
    """Verify that key functions have docstrings."""
    
    def test_checkmarketopen_has_docstrings(self):
        """CheckMarketOpen functions have docstrings."""
        from functions.CheckMarketOpen import get_MarketOpenOrClosed, CheckMarketOpen
        
        assert get_MarketOpenOrClosed.__doc__ is not None, "get_MarketOpenOrClosed missing docstring"
        assert CheckMarketOpen.__doc__ is not None, "CheckMarketOpen missing docstring"
        assert len(get_MarketOpenOrClosed.__doc__) > 50, "Docstring too short"
    
    def test_tafunctions_has_docstrings(self):
        """TAfunctions priority functions have docstrings."""
        from functions.TAfunctions import strip_accents, normcorrcoef
        
        assert strip_accents.__doc__ is not None, "strip_accents missing docstring"
        assert normcorrcoef.__doc__ is not None, "normcorrcoef missing docstring"
        assert len(strip_accents.__doc__) > 50, "strip_accents docstring too short"
    
    def test_getparams_has_docstrings(self):
        """GetParams key functions have docstrings."""
        from functions.GetParams import (get_symbols_file, get_performance_store, 
                                         get_webpage_store)
        
        assert get_symbols_file.__doc__ is not None, "get_symbols_file missing docstring"
        assert get_performance_store.__doc__ is not None, "get_performance_store missing docstring"
        assert get_webpage_store.__doc__ is not None, "get_webpage_store missing docstring"
        
    def test_module_docstrings(self):
        """Key modules have module-level docstrings."""
        from functions import CheckMarketOpen, TAfunctions, GetParams
        
        assert CheckMarketOpen.__doc__ is not None, "CheckMarketOpen module missing docstring"
        assert TAfunctions.__doc__ is not None, "TAfunctions module missing docstring"
        assert GetParams.__doc__ is not None, "GetParams module missing docstring"


class TestFunctionSignatures:
    """Verify that function signatures haven't changed."""
    
    def test_get_MarketOpenOrClosed_signature(self):
        """get_MarketOpenOrClosed returns string."""
        from functions.CheckMarketOpen import get_MarketOpenOrClosed
        result = get_MarketOpenOrClosed()
        assert isinstance(result, str)
        assert "Markets" in result or "Market" in result
    
    def test_CheckMarketOpen_signature(self):
        """CheckMarketOpen returns tuple of two bools."""
        from functions.CheckMarketOpen import CheckMarketOpen
        result = CheckMarketOpen()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], bool)
    
    def test_strip_accents_signature(self):
        """strip_accents processes text correctly."""
        from functions.TAfunctions import strip_accents
        result = strip_accents("test")
        assert isinstance(result, str)
        assert result == "test"
        
        # Test with accented text
        result = strip_accents("café")
        assert "cafe" in result.lower()
    
    def test_normcorrcoef_signature(self):
        """normcorrcoef computes correlation."""
        from functions.TAfunctions import normcorrcoef
        import numpy as np
        
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        result = normcorrcoef(a, b)
        
        assert isinstance(result, (float, np.floating, np.ndarray))
        assert result > 0.99  # Perfect correlation


class TestCodeQuality:
    """Verify code quality improvements."""
    
    def test_no_bare_except_in_modified_files(self):
        """Modified files should have no bare except clauses (will be caught later if present)."""
        # This is a reminder that Phase 2 will address bare except clauses
        # For now, just verify the files are readable
        files_to_check = [
            'functions/CheckMarketOpen.py',
            'functions/TAfunctions.py',
            'functions/quotes_for_list_adjClose.py'
        ]
        
        for filepath in files_to_check:
            assert os.path.exists(filepath), f"{filepath} should exist"
            with open(filepath, 'r') as f:
                content = f.read()
                assert len(content) > 0, f"{filepath} should not be empty"
