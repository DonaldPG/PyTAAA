

import datetime

import nose
import bottleneck as bn
import la
import h5py
import os

from scipy.stats import gmean
from la.external.matplotlib import quotes_historical_yahoo

## local imports
from functions.quotes_for_list_adjCloseVol import *
from functions.TAfunctions import *
from functions.readSymbols import *

def loadQuotes_fromHDF( symbols_file ):

    (directory_name, file_name) = os.path.split(symbols_file)
    (shortname, extension) = os.path.splitext( file_name )

    print "file name for symbols = ","_"+shortname+"_"
    print "file type for symbols = ",extension

    # set up to write quotes to disk.

    if shortname == "symbols" :
        listname = "TAA-Symbols"
    elif shortname == "cmg_symbols" :
        listname = "CMG-Symbols"
    elif shortname == "Naz100_symbols" :
        listname = "Naz100-Symbols"
    elif shortname == "biglist" :
        listname = "biglist-Symbols"
    elif shortname == "ETF_symbols" :
        listname = "ETF-Symbols"
    elif shortname == "ProvidentFundSymbols" :
        listname = "ProvidentFund-Symbols"
    elif shortname == "sp500_symbols" :
        listname = "SP500-Symbols"
    else :
        listname = shortname

    hdf5_directory = os.path.join( os.getcwd(), "symbols" )
    hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")

    print ""
    print ""
    print "symbol_directory = ", directory_name
    print "symbols_file = ", symbols_file
    print "shortname, extension = ",shortname, extension
    print "hdf5filename = ",hdf5filename


    try:
        io = la.IO(hdf5filename)
        quote = io[listname][:]
        x=quote.copyx()
        date = quote.getlabel(1)
        symbols = quote.getlabel(0)
        dates=[]
        for i in range(len(date)):
            datestr = date[i]
            date_newformat = datetime.date(*[int(val) for val in datestr.split('-')])
            dates.append(date_newformat)
    except:
        createHDF( hdf5_directory, symbols_file, listname )
        io = la.IO(hdf5filename)
        quote = io[listname][:]
        x=quote.copyx()
        date = quote.getlabel(1)
        symbols = quote.getlabel(0)
        dates=[]
        for i in range(len(date)):
            datestr = date[i]
            date_newformat = datetime.date(*[int(val) for val in datestr.split('-')])
            dates.append(date_newformat)

    return x, symbols, dates, quote, listname

def getLastDateFromHDF5( symbol_directory, symbols_file ) :
    filename = os.path.join(symbol_directory, symbols_file)
    adjClose, symbols, datearray, quote, _ = loadQuotes_fromHDF( filename )
    import numpy as np
    symbols2 = quote.getlabel(0)
    for i in range(len(symbols)):
        numisnans = adjClose[i,:].copy()
        print "...inside getLastDateFromHDF5...  i, numisnans = ", i, numisnans
        print "...inside getLastDateFromHDF5...  i, symbols[i], symbols2[i] = ", i, symbols[i],symbols2[i], numisnans[np.isnan(numisnans)].shape
    numdates = adjClose.shape[1]
    print "...inside getLastDateFromHDF5...  numdates, quote.shape = ", numdates, adjClose.shape
    for i in range(adjClose.shape[0]):
        for j in range(adjClose.shape[1]):
            if isnan(adjClose[i,numdates-j-1])  :
                print "...inside getLastDateFromHDF5...  i, j, numdates, numdates-j, quote[i,numdates-j] = ", i, j, numdates, numdates-j-1, adjClose[i,numdates-j-1]
                lastindex = numdates-j-1
    return datearray[lastindex]


def getLastDateFromHDF5( symbol_directory, symbols_file ) :
    filename = os.path.join(symbol_directory, symbols_file)
    _, _, datearray, _, _ = loadQuotes_fromHDF( filename )

    import datetime
    today = datetime.datetime.now()
    hourOfDay = today.hour
    dayOfWeek = today.weekday()
    dayOfMonth = today.day
    tomorrow = today + datetime.timedelta( days=1 )
    tomorrowDayOfMonth = tomorrow.day

    # set up to return current day's quotes.
    # - Except late Friday nights and at end of month, when quotes are updated for entire history.
    # - This logic ensures dividends and splits are accounted for.
    # TODO: check if there's a split or deividend and only get entire history if 'yes'.
    if  hourOfDay >= 22 :
        return datearray[0]
    else:
        return datearray[-1]


def UpdateHDF5( symbol_directory, symbols_file ):

    ##
    ##  Update symbols in 'symbols_file' with quotes more recent than last update.
    ##

    filename = os.path.join(symbol_directory, symbols_file)

    x, symbols, datearray, quote, listname = loadQuotes_fromHDF( filename )

    # get last date in hdf5 archive
    from datetime import datetime

    date = quote.getlabel(1)
    lastdate = datetime.strptime(date[-1], '%Y-%m-%d')
    lastdatetuple = (lastdate.year,lastdate.month,lastdate.day)
    print "lastdatetuple = ", lastdatetuple

    lastdate = getLastDateFromHDF5( symbol_directory, symbols_file )
    lastdatetuple = (lastdate.year,lastdate.month,lastdate.day)
    print "lastdatetuple = ", lastdatetuple

    ##
    ## Get quotes for each symbol in list
    ## process dates.
    ## Clean up quotes.
    ## Make a plot showing all symbols in list
    ##

    # locate symbols added to list that aren't in HDF5 file
    symbols_in_list = readSymbolList( filename, verbose=False)
    symbols_in_HDF5 = quote.getlabel(0)
    new_symbols = [x for x in symbols_in_list if x  not in symbols_in_HDF5]

    # write new symbols to temporary file
    if len(new_symbols) > 0:
        # write new symbols to temporary file
        tempfilename = os.path.join(symbol_directory, "newsymbols_tempfile.txt")
        OUTFILE = open(tempfilename,"w",0)
        for i,isymbol in enumerate(new_symbols):
            print "new symbol = ", isymbol
            OUTFILE.write(str(isymbol) + "\n")

        newquotesfirstdate = (1991,1,1)
        newquoteslastdate = (datetime.now().year,datetime.now().month,datetime.now().day)

        # print dates to be used
        print "dates for new symbol found = ", newquotesfirstdate, newquoteslastdate

        newadjClose, symbols, newdatearray = arrayFromQuotesForList(tempfilename, newquotesfirstdate, newquoteslastdate)

        print " security values check: ",newadjClose[isnan(newadjClose)].shape

        newdates = []
        for i in range(newdatearray.shape[0]):
            newdates.append(str(newdatearray[i]))
        quotes_NewSymbols = la.larry(newadjClose, [symbols,newdates], dtype=float)

    # add CASH as symbol (with value of 1.0 for all dates)
    CASHsymbols = ['CASH']
    if len(CASHsymbols) > 0:
        # write CASH symbols to temporary file
        tempfilename = os.path.join(symbol_directory, "CASHsymbols_tempfile.txt")
        OUTFILE = open(tempfilename,"w",0)
        for i,isymbol in enumerate(CASHsymbols):
            print "new symbol = ", isymbol
            OUTFILE.write(str(isymbol) + "\n")

        date = quote.getlabel(1)
        CASHdatearray = np.array( date )
        CASHadjClose = np.ones( (1,len(CASHdatearray)), float ) * 100.
        for i in range(CASHadjClose.shape[0]):
            if i%10 == 0:
                CASHadjClose[i] = CASHadjClose[i-1] + .01
            else:
                CASHadjClose[i] = CASHadjClose[i-1]

        print " security values check: ",CASHadjClose[isnan(CASHadjClose)].shape
        print "CASHsymbols = ", CASHsymbols

        CASHdates = []
        for i in range(CASHdatearray.shape[0]):
            CASHdates.append(str(CASHdatearray[i]))
        print "sizes of CASHadjClose, CASHdates = ",CASHadjClose.shape, len(CASHdates)
        quotes_CASHSymbols = la.larry(CASHadjClose, [CASHsymbols,CASHdates], dtype=float)

    ##
    ## Get quotes for each symbol in list
    ## process dates.
    ## Clean up quotes.
    ## Make a plot showing all symbols in list
    ##

    newquotesfirstdate =lastdate
    newquoteslastdate = (datetime.now().year,datetime.now().month,datetime.now().day)
    print " newquotesfirstdate, newquoteslastdate = ", lastdate, newquoteslastdate

    newadjClose, symbols, newdatearray = arrayFromQuotesForList(filename, newquotesfirstdate, newquoteslastdate)

    print " security values check: ",newadjClose[isnan(newadjClose)].shape

    newdates = []
    for i in range(newdatearray.shape[0]):
        newdates.append(str(newdatearray[i]))
    quoteupdate = la.larry(newadjClose, [symbols,newdates], dtype=float)

    updatedquotes = quote.merge(quoteupdate, update=True)
    if len(new_symbols) > 0:
        updatedquotes = updatedquotes.merge(quotes_NewSymbols, update=True)
    if len(CASHsymbols) > 0:
        updatedquotes = updatedquotes.merge(quotes_CASHSymbols, update=True)

    # set up to write quotes to disk.
    dirname = os.path.join( os.getcwd(), "symbols" )

    hdf5filename = os.path.join( dirname, listname + "_.hdf5" )
    print "hdf5 filename = ",hdf5filename
    io = la.IO(hdf5filename)
    io[listname] = updatedquotes

    return

def createHDF( hdf5_directory, symbol_file, listname ):

    import os
    import numpy as np
    from matplotlib.pylab import *
    import matplotlib.gridspec as gridspec

    from datetime import datetime

    import nose
    import bottleneck as bn
    import la
    import h5py

    from la.external.matplotlib import quotes_historical_yahoo

    ## local imports
    from functions.quotes_for_list_adjCloseVol import *
    from functions.TAfunctions import *

    hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")
    (shortname, extension) = os.path.splitext( symbol_file )
    symbol_directory = hdf5_directory

    if shortname == "symbols" :
        listname = "TAA-Symbols"
    elif shortname == "cmg_symbols" :
        listname = "CMG-Symbols"
    elif shortname == "Naz100_symbols" :
        listname = "Naz100-Symbols"
    elif shortname == "biglist" :
        listname = "biglist-Symbols"
    elif shortname == "ETF_symbols" :
        listname = "ETF-Symbols"
    elif shortname == "ProvidentFundSymbols" :
        listname = "ProvidentFund-Symbols"
    elif shortname == "sp500_symbols" :
        listname = "SP500-Symbols"
    else :
        listname = shortname

    print "symbol_directory = ",symbol_directory
    print "symbol_file = ",symbol_file
    print "shortname, extension = ",shortname, extension
    print "hdf5filename = ",hdf5filename

    ##
    ## Get quotes for each symbol in list
    ## process dates.
    ## Clean up quotes.
    ## Make a plot showing all symbols in list
    ##

    firstdate=(1991,1,1)
    lastdate=(2011,11,30)
    lastdate=(2013,6,1)
    import datetime
    today = datetime.datetime.now()
    lastdate = ( today.year, today.month, today.day )

    filename = os.path.join( symbol_directory, symbol_file )
    print "filename with list of symbols = ", filename

    adjClose, symbols, datearray = arrayFromQuotesForList(filename, firstdate, lastdate)
    Close = adjClose
    print " security values check (adjClose): ",adjClose[isnan(adjClose)].shape
    print " security values check (Close): ",Close[isnan(Close)].shape

    dates = []
    for i in range(datearray.shape[0]):
        dates.append(str(datearray[i]))
    quotetype = []
    quotetype.append('adjClose')
    quotetype.append('Volume')
    quotetype.append('Close')

    quotesarray = np.zeros( (adjClose.shape[0], 3, adjClose.shape[1] ), float )
    quotesarray[:,0,:] = adjClose
    quotesarray[:,2,:] = Close
    quoteupdate = la.larry(quotesarray, [symbols,quotetype,dates], dtype=float)

    io = la.IO(hdf5filename)
    io[listname] = quoteupdate

    return
