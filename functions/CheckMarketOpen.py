import datetime


#############################################################################
# Perform a check to see if the stock market is open.
# Purpose is to stop calculating and sending emails when nothing has changed.
#############################################################################


def get_MarketOpenOrClosed():

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


def CheckMarketOpen() :

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
