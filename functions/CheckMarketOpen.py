import datetime

###
### Perform a check to see if the stock market is open
### - purpose is to stop calculating and sending emails when nothing has changed
###

def CheckMarketOpen() :

    today = datetime.datetime.now()
    hourOfDay = today.hour
    dayOfWeek = today.weekday()
    dayOfMonth = today.day
    monthOfYear = today.month

    # Use simple checks to suppress execcessive execution during closed markets
    marketOpen = False
    if dayOfWeek < 5:
        if hourOfDay > 8 and hourOfDay < 18 :
            marketOpen = True

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
