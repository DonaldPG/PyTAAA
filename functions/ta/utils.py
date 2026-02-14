"""Utility functions for technical analysis.

This module contains basic utility functions used throughout the PyTAAA
technical analysis system, including text processing and correlation calculations.
"""

import numpy as np
from numpy.typing import NDArray


def strip_accents(text: str) -> str:
    """Remove accent marks and diacritics from Unicode text.
    
    Converts text to NFD normalized form, encodes to ASCII (ignoring
    non-ASCII characters), and decodes back to UTF-8. This is useful
    for cleaning company names and symbols that may contain accents.
    
    Args:
        text: Input text string (may contain Unicode accents)
        
    Returns:
        str: Text with all accent marks removed
        
    Example:
        >>> strip_accents("Café")
        'Cafe'
        >>> strip_accents("Zürich")
        'Zurich'
    """
    import unicodedata
    try:
        text = unicode(text, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)


def normcorrcoef(a: NDArray[np.floating], b: NDArray[np.floating]) -> np.floating:
    """Compute normalized correlation coefficient between two arrays.
    
    This is a more numerically stable version of correlation computation
    than the standard formula, using numpy's correlate function.
    
    Args:
        a: First numpy array
        b: Second numpy array (must have same length as a)
        
    Returns:
        float: Normalized correlation coefficient between a and b
        
    Note:
        This differs from numpy.corrcoef in that it uses numpy.correlate
        for the computation, which can be more stable for certain inputs.
    """
    return np.correlate(a, b) / np.sqrt(np.correlate(a, a) * np.correlate(b, b))[0]


def nanrms(x: NDArray[np.floating], axis: int | None = None) -> float | NDArray[np.floating]:
    """Calculate root mean square ignoring NaN values.
    
    Computes the RMS (root mean square) of an array while safely handling
    NaN values by excluding them from the calculation.
    
    Args:
        x: Numpy array (any shape) containing data with potential NaN values
        axis: Axis along which to compute RMS. If None, compute over flattened array
        
    Returns:
        float or NDArray: Root mean square value(s)
        
    Example:
        >>> data = np.array([1.0, 2.0, np.nan, 3.0])
        >>> rms = nanrms(data)
        >>> # Returns sqrt(mean([1, 4, 9])) = sqrt(14/3)
        
    Note:
        - Uses bottleneck.nanmean for efficient NaN handling
        - RMS = sqrt(mean(x^2))
        - Useful for volatility and risk calculations
    """
    from bottleneck import nanmean
    return np.sqrt(nanmean(x**2, axis=axis))
