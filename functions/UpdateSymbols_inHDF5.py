
#import time, threading

#import numpy as np
#from matplotlib.pylab import *
#import matplotlib.gridspec as gridspec

import datetime
#from scipy import random
#from random import choice
#from scipy.stats import rankdata
#import scipy as sp

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

    hdf5_directory = os.getcwd()+"\\symbols"
    hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")

    print ""
    print ""
    print "symbol_directory = ", directory_name
    print "symbols_file = ", symbols_file
    print "shortname, extension = ",shortname, extension
    print "hdf5filename = ",hdf5filename


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


def UpdateHDF5( symbol_directory, symbols_file ):

    ##
    ##  Update symbols in 'symbols_file' with quotes more recent than last update.
    ##

    """# create list of symbols files on disk.
    symbols_file = []
    symbols_file.append("RDSA_symbols.txt")
    symbols_file.append("cmg_symbols.txt")
    symbols_file.append("sp500_symbols.txt")
    symbols_file.append("Naz100_symbols.txt")                     # plotmax = 1.e10, runnum = 902
    symbols_file.append("symbols.txt")                            # plotmax = 1.e5, runnum = 901
    symbols_file.append("biglist.txt")                            # plotmax = 1.e10, runnum = 903
    symbols_file.append("ETF_symbols.txt")                        # plotmax = 1.e6, runnum = 904
    """

    filename = os.path.join(symbol_directory, symbols_file)

    """
    (shortname, extension) = os.path.splitext(symbols_file)

    # set up to write quotes to disk.

    if shortname == "symbols" :
        listname = "TAA-Symbols"
    if shortname == "cmg_symbols" :
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
        listname = "Favs-Symbols"

    hdf5_directory = r'C:\users\don\Naz100_stats\quotes_data'
    hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")

    print ""
    print ""
    print "symbol_directory = ",symbol_directory
    print "symbols_file = ",symbols_file
    print "shortname, extension = ",shortname, extension
    print "hdf5filename = ",hdf5filename


    io = la.IO(hdf5filename)
    quote = io[listname][:]
    x=quote.copyx()
    date = quote.getlabel(1)
    datearray=array(date)
    """

    #x, symbols, datearray, quote, listname = loadQuotes_fromHDF( symbols_file )
    x, symbols, datearray, quote, listname = loadQuotes_fromHDF( filename )

    '''
    ## local imports
    from functions.quotes_for_list_adjCloseVol import *
    from functions.TAfunctions import *
    '''

    # get last date in hdf5 archive
    from datetime import datetime

    date = quote.getlabel(1)
    lastdate = datetime.strptime(date[-1], '%Y-%m-%d')
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

        #updatedquotes = quotes_new-symbols.merge(quote, update=True)

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
        CASHdatearray=array(date)
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

    newadjClose, symbols, newdatearray = arrayFromQuotesForList(filename, newquotesfirstdate, newquoteslastdate)

    print " security values check: ",newadjClose[isnan(newadjClose)].shape

    newdates = []
    for i in range(newdatearray.shape[0]):
        newdates.append(str(newdatearray[i]))
    quoteupdate = la.larry(newadjClose, [symbols,newdates], dtype=float)

    updatedquotes = quoteupdate.merge(quote, update=True)
    if len(new_symbols) > 0:
        updatedquotes = updatedquotes.merge(quotes_NewSymbols, update=True)
    if len(CASHsymbols) > 0:
        updatedquotes = updatedquotes.merge(quotes_CASHSymbols, update=True)

    # set up to write quotes to disk.
    dirname = os.getcwd()+"\\symbols"

    #hdf5filename = dirname + listname + "_.hdf5"
    hdf5filename = os.path.join( dirname, listname + "_.hdf5" )
    print "hdf5 filename = ",hdf5filename
    io = la.IO(hdf5filename)
    io[listname] = updatedquotes

    return

