"""Market status checking utilities.

This module provides functions to determine if US stock markets are currently
open or closed, accounting for trading hours, weekends, and holidays.

Functions:
    get_MarketOpenOrClosed: Check current market status
    CheckMarketOpen: Check market status and last trading day of month
"""

import datetime
from typing import Tuple


def get_MarketOpenOrClosed() -> str:
    """Check if US stock markets are currently open or closed.
    
    Uses Eastern Time timezone and the US holidays calendar to determine
    market status based on time of day, day of week, and holidays.
    
    Market hours: Monday-Friday, 9:30 AM - 4:00 PM ET
    Closed on: Weekends and US federal holidays
    
    Returns:
        str: " Markets are open" if currently within trading hours,
             " Markets are closed" otherwise.
             
    Example:
        >>> status = get_MarketOpenOrClosed()
        >>> print(status)
        ' Markets are closed'
    """
    import datetime, pytz, holidays

    tz = pytz.timezone('US/Eastern')
    us_holidays = holidays.US()

    after_hours = False

    now = datetime.datetime.now(tz)
    openTime = datetime.time(hour = 9, minute = 30, second = 0)
    closeTime = datetime.time(hour = 16, minute = 0, second = 0)
    # If a holiday
    if now.strftime('%Y-%m-%d') in us_holidays:
        after_hours = True
    # If before 0930 or after 1600
    if (now.time() < openTime) or (now.time() > closeTime):
        after_hours = True
    # If it's a weekend
    if now.date().weekday() > 4:
        after_hours = True

    if after_hours:
        status = " Markets are closed"
    else:
        status = " Markets are open"

    return status


def CheckMarketOpen() -> Tuple[bool, bool]:
    """Check if markets are open and if today is last trading day of month.
    
    Performs two checks:
    1. Whether markets are currently open (based on get_MarketOpenOrClosed)
    2. Whether today is the last trading day of the month (after 1 PM)
    
    The last trading day check accounts for weekends by looking ahead to
    determine if the next trading day falls in a different month.
    
    Returns:
        tuple: (marketOpen, lastDayOfMonth)
            marketOpen (bool): True if markets are open, False otherwise
            lastDayOfMonth (bool): True if today is last trading day of month
                (after 1 PM), False otherwise
                
    Example:
        >>> is_open, is_month_end = CheckMarketOpen()
        >>> if is_open:
        ...     print("Markets are open")
        >>> if is_month_end:
        ...     print("Last trading day of month")
    """

    today = datetime.datetime.now()
    hourOfDay = today.hour
    dayOfWeek = today.weekday()
    dayOfMonth = today.day
    monthOfYear = today.month

    # Use simple checks to suppress execcessive execution during closed markets
    market_status = get_MarketOpenOrClosed()
    #if market_status == 'open':
    if market_status == 'close in':
        marketOpen = True
    else:
        marketOpen = False


    # Use simple checks to tell is this is the last trading day of month
    if dayOfWeek < 4:
        tomorrow = today + datetime.timedelta( days=1 )
    elif dayOfWeek == 4:
        tomorrow = today + datetime.timedelta( days=3 )
    elif dayOfWeek == 5:
        tomorrow = today + datetime.timedelta( days=2 )
    elif dayOfWeek == 6:
        tomorrow = today + datetime.timedelta( days=1 )

    lastDayOfMonth = False
    if monthOfYear != tomorrow.month and hourOfDay > 13 :
        lastDayOfMonth = True

    return marketOpen, lastDayOfMonth
