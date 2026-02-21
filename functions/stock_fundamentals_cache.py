"""Cached stock fundamentals (PE ratio, sector, industry) with thread-safe updates.

This module provides local caching of stock fundamental data with:
- JSON-based persistent storage
- Thread-safe file locking for concurrent access
- Automatic refresh for stale data (>5 days old)
- Only updates symbols currently in Naz100/SP500
- Fallback to web scraping when needed
"""

import os
import json
import datetime
import fcntl
import logging
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class StockFundamentalsCache:
    """Thread-safe cache for stock fundamental data."""
    
    def __init__(self, cache_file: str = None, max_age_days: int = 5):
        """Initialize the cache.
        
        Args:
            cache_file: Path to JSON cache file. If None, uses default location.
            max_age_days: Maximum age in days before data is considered stale.
        """
        if cache_file is None:
            # Store in data directory by default
            cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, 'stock_fundamentals_cache.json')
        
        self.cache_file = cache_file
        self.max_age_days = max_age_days
        self._lock_file = cache_file + '.lock'
        
    def _acquire_lock(self, lock_fd):
        """Acquire an exclusive file lock (blocks until available)."""
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        
    def _release_lock(self, lock_fd):
        """Release the file lock."""
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        
    def _read_cache(self) -> Dict:
        """Read cache from disk with file locking."""
        # Create lock file if it doesn't exist
        lock_fd = os.open(self._lock_file, os.O_CREAT | os.O_RDWR)
        
        try:
            self._acquire_lock(lock_fd)
            
            if not os.path.exists(self.cache_file):
                return {}
            
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache: {e}")
            return {}
        finally:
            self._release_lock(lock_fd)
            os.close(lock_fd)
    
    def _write_cache(self, cache_data: Dict):
        """Write cache to disk with file locking."""
        lock_fd = os.open(self._lock_file, os.O_CREAT | os.O_RDWR)
        
        try:
            self._acquire_lock(lock_fd)
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to write cache: {e}")
        finally:
            self._release_lock(lock_fd)
            os.close(lock_fd)
    
    def _is_stale(self, timestamp_str: str) -> bool:
        """Check if cached data is stale based on timestamp."""
        try:
            cached_date = datetime.datetime.fromisoformat(timestamp_str)
            age = datetime.datetime.now() - cached_date
            return age.days > self.max_age_days
        except (ValueError, TypeError):
            return True
    
    def get_pe_ratio(self, symbol: str, force_refresh: bool = False) -> float:
        """Get PE ratio for symbol with caching.
        
        Args:
            symbol: Stock ticker symbol
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            PE ratio as float, or np.nan if unavailable
        """
        cache = self._read_cache()
        
        # Check cache if not forcing refresh
        if not force_refresh and symbol in cache:
            entry = cache[symbol]
            if not self._is_stale(entry.get('timestamp', '')):
                pe = entry.get('pe_ratio')
                if pe is not None:
                    logger.debug(f"Cache hit for {symbol} PE ratio: {pe}")
                    return float(pe) if not np.isnan(pe) else np.nan
        
        # Cache miss or stale - fetch fresh data
        logger.info(f"Fetching fresh PE ratio for {symbol}")
        pe = self._fetch_pe_ratio_from_web(symbol)
        
        # Update cache
        if symbol not in cache:
            cache[symbol] = {}
        cache[symbol]['pe_ratio'] = float(pe) if not np.isnan(pe) else None
        cache[symbol]['timestamp'] = datetime.datetime.now().isoformat()
        
        self._write_cache(cache)
        return pe
    
    def get_sector_industry(self, symbol: str, force_refresh: bool = False) -> Tuple[str, str]:
        """Get sector and industry for symbol with caching.
        
        Args:
            symbol: Stock ticker symbol
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Tuple of (sector, industry) strings
        """
        cache = self._read_cache()
        
        # Check cache if not forcing refresh
        if not force_refresh and symbol in cache:
            entry = cache[symbol]
            if not self._is_stale(entry.get('timestamp', '')):
                sector = entry.get('sector', 'unknown')
                industry = entry.get('industry', 'unknown')
                if sector != 'unknown' or industry != 'unknown':
                    logger.debug(f"Cache hit for {symbol} sector/industry: {sector}/{industry}")
                    return sector, industry
        
        # Cache miss or stale - fetch fresh data
        logger.info(f"Fetching fresh sector/industry for {symbol}")
        sector, industry = self._fetch_sector_industry_from_web(symbol)
        
        # Update cache
        if symbol not in cache:
            cache[symbol] = {}
        cache[symbol]['sector'] = sector
        cache[symbol]['industry'] = industry
        cache[symbol]['timestamp'] = datetime.datetime.now().isoformat()
        
        self._write_cache(cache)
        return sector, industry
    
    def _fetch_pe_ratio_from_web(self, symbol: str) -> float:
        """Fetch PE ratio from web (finviz or yfinance)."""
        # Try yfinance first (better, no rate limits)
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different PE ratio fields
            pe = info.get('trailingPE') or info.get('forwardPE')
            if pe is not None and not np.isnan(pe):
                logger.info(f"Got PE from yfinance for {symbol}: {pe}")
                return float(pe)
        except Exception as e:
            logger.debug(f"yfinance failed for {symbol}: {e}")
        
        # Fallback to finviz scraping
        try:
            from functions.quotes_for_list_adjClose import get_pe_finviz
            pe = get_pe_finviz(symbol, verbose=False)
            logger.info(f"Got PE from finviz for {symbol}: {pe}")
            return pe
        except Exception as e:
            logger.warning(f"Failed to fetch PE for {symbol}: {e}")
            return np.nan
    
    def _fetch_sector_industry_from_web(self, symbol: str) -> Tuple[str, str]:
        """Fetch sector and industry from web (yfinance or finviz)."""
        # Try yfinance first (better, no rate limits)
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            sector = info.get('sector', 'unknown')
            industry = info.get('industry', 'unknown')
            
            if sector != 'unknown' or industry != 'unknown':
                logger.info(f"Got sector/industry from yfinance for {symbol}: {sector}/{industry}")
                return sector, industry
        except Exception as e:
            logger.debug(f"yfinance failed for {symbol}: {e}")
        
        # Fallback to finviz scraping (existing method)
        try:
            from finvizfinance.quote import finvizfinance
            stock = finvizfinance(symbol)
            stock_fundament = stock.ticker_fundament()
            sector = stock_fundament.get("Sector", "unknown")
            industry = stock_fundament.get("Industry", "unknown")
            logger.info(f"Got sector/industry from finviz for {symbol}: {sector}/{industry}")
            return sector, industry
        except Exception as e:
            logger.warning(f"Failed to fetch sector/industry for {symbol}: {e}")
            return "unknown", "unknown"
    
    def update_for_current_symbols(self, symbols: list, force_refresh: bool = False, valid_symbols: list = None):
        """Update cache for currently active symbols only.
        
        Args:
            symbols: List of ticker symbols to update
            force_refresh: If True, refresh even if not stale
            valid_symbols: Optional list of valid symbols. If provided, only update symbols in this list.
        """
        if valid_symbols is not None:
            valid_set = set(valid_symbols)
            symbols = [s for s in symbols if s in valid_set]
            logger.info(f"Filtered to {len(symbols)} symbols that are in valid universe")
        
        logger.info(f"Updating cache for {len(symbols)} active symbols")
        
        for symbol in symbols:
            if symbol == 'CASH':
                continue
            
            try:
                # Get both PE and sector/industry
                self.get_pe_ratio(symbol, force_refresh=force_refresh)
                self.get_sector_industry(symbol, force_refresh=force_refresh)
            except Exception as e:
                logger.warning(f"Failed to update {symbol}: {e}")
                continue
    
    def prune_old_symbols(self, current_symbols: list, keep_days: int = 30):
        """Remove symbols from cache that aren't in current list and are old.
        
        Args:
            current_symbols: List of currently active ticker symbols
            keep_days: Keep symbols for this many days even if not current
        """
        cache = self._read_cache()
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        
        symbols_to_remove = []
        for symbol in cache.keys():
            if symbol not in current_symbols:
                try:
                    cached_date = datetime.datetime.fromisoformat(cache[symbol].get('timestamp', ''))
                    if cached_date < cutoff_date:
                        symbols_to_remove.append(symbol)
                except (ValueError, TypeError):
                    symbols_to_remove.append(symbol)
        
        if symbols_to_remove:
            logger.info(f"Pruning {len(symbols_to_remove)} old symbols from cache")
            for symbol in symbols_to_remove:
                del cache[symbol]
            self._write_cache(cache)


# Global cache instance
_cache_instance: Optional[StockFundamentalsCache] = None


def get_cache(cache_file: str = None, max_age_days: int = 5) -> StockFundamentalsCache:
    """Get or create the global cache instance.
    
    Args:
        cache_file: Path to cache file (only used on first call)
        max_age_days: Maximum age for cached data (only used on first call)
        
    Returns:
        StockFundamentalsCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = StockFundamentalsCache(cache_file, max_age_days)
    return _cache_instance


# Convenience functions for backward compatibility
def get_pe_cached(symbol: str) -> float:
    """Get PE ratio with caching (backward compatible interface).
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        PE ratio as float, or np.nan if unavailable
    """
    cache = get_cache()
    return cache.get_pe_ratio(symbol)


def get_sector_industry_cached(symbol: str) -> Tuple[str, str]:
    """Get sector and industry with caching (backward compatible interface).
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Tuple of (sector, industry) strings
    """
    cache = get_cache()
    return cache.get_sector_industry(symbol)
