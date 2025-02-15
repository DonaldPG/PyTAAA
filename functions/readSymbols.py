'''
Created on May 12, 202

@author: donaldpg
'''
import os
import datetime
import pandas as pd

class webpage_companies_extractor:
    Url = None

    def __init__(self, url):
        self.__url = url

    def get_companies_list(self, current_portfolio=None):
        dfs = pd.read_html(self.__url, header=0)
        first_table = dfs[2]
        company_names = first_table
        symbols = list(company_names['Ticker'].values)
        companyNames = list(company_names['Company'].values)
        return symbols, companyNames


def strip_accents(text):
    import unicodedata
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass
    try:
        text = text.decode('ascii')
    except:
        pass
    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")
    return str(text)


# '''
# def readSymbolList(filename, verbose=False):
#     # Get the Data
#     from functions.quotes_for_list_adjClose import get_Naz100List, get_SP500List
#     try:
#         print(" ...inside readSymbolList... filename = ", filename)
#         infile = open(filename,"r")
#     except:
#         symbol_directory = os.path.join( os.getcwd(), "symbols" )
#         # the symbols list doesn't exist. generate from the web.
#         if 'SP500' in filename:
#             symbol_file = "SP500_Symbols.txt"
#             symbols_file = os.path.join( symbol_directory, symbol_file )
#             open(symbols_file, 'a').close()
#             symbolList, _, _ = get_SP500List( verbose=True )
#             infile.close()
#             infile = open(filename,"r")
#         elif 'Naz100' in filename:
#             symbol_file = "Naz100_Symbols.txt"
#             symbols_file = os.path.join( symbol_directory, symbol_file )
#             open(symbols_file, 'a').close()
#             symbolList, _, _ = get_Naz100List( verbose=True )

#     symbols = []

#     content = infile.read()
#     number_lines = len(content.split("\n"))
#     if number_lines == 0:
#         symbol_directory = os.path.join( os.getcwd(), "symbols" )
#         # the symbols list doesn't exist. generate from the web.
#         if 'SP500' in filename:
#             symbol_file = "SP500_Symbols.txt"
#             symbols_file = os.path.join( symbol_directory, symbol_file )
#             open(symbols_file, 'a').close()
#             symbolList, _, _ = get_SP500List( verbose=True )
#         elif 'Naz100' in filename:
#             symbol_file = "Naz100_Symbols.txt"
#             symbols_file = os.path.join( symbol_directory, symbol_file )
#             open(symbols_file, 'a').close()
#             symbolList, _, _ = get_Naz100List( verbose=True )

#     infile.close()
#     infile = open(filename,"r")

#     while infile:
#         line = infile.readline()
#         s = line.split()
#         n = len(s)
#         if n != 0:
#             for i in range(len(s)):
#                 s[i] = s[i].replace('.','-')
#                 symbols.append(s[i])
#         else:
#             break

#     # ensure that there are no duplicate tickers
#     symbols = list( set( symbols ) )

#     # sort list of symbols
#     symbols.sort()

#     # print list of symbols
#     if verbose:
#         print("number of symbols is ",len(symbols))
#         print(symbols)

#     return symbols
# '''


def readSymbolList(filename, json_fn, verbose=False):
    # Get the Data
    from functions.GetParams import get_symbols_file
    existing_symbols_file = get_symbols_file(json_fn)
    try:
        print(" ...inside readSymbolList... filename = ", filename)
        infile = open(filename,"r")
    except:
        from functions.quotes_for_list_adjClose import get_Naz100List, get_SP500List
        symbol_directory = os.path.join( os.getcwd(), "symbols" )
        # the symbols list doesn't exist. generate from the web.
        if 'SP500' in filename:
            symbol_file = "SP500_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_SP500List(existing_symbols_file, verbose=True )
            infile.close()
            infile = open(filename,"r")
        elif 'Naz100' in filename:
            symbol_file = "Naz100_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_Naz100List(existing_symbols_file, verbose=True )

    symbols = []

    content = infile.read()
    number_lines = len(content.split("\n"))
    if number_lines == 0:
        from quotes_for_list_adjClose import get_Naz100List, get_SP500List
        symbol_directory = os.path.join( os.getcwd(), "symbols" )
        # the symbols list doesn't exist. generate from the web.
        if 'SP500' in filename:
            symbol_file = "SP500_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_SP500List(existing_symbols_file, verbose=True )
        elif 'Naz100' in filename:
            symbol_file = "Naz100_Symbols.txt"
            symbols_file = os.path.join( symbol_directory, symbol_file )
            open(symbols_file, 'a').close()
            symbolList, _, _ = get_Naz100List(existing_symbols_file, verbose=True )

    infile.close()
    infile = open(filename,"r")

    while infile:
        line = infile.readline()
        s = line.split()
        n = len(s)
        if n != 0:
            for i in range(len(s)):
                s[i] = s[i].replace('.','-')
                symbols.append(s[i])
        else:
            break

    # ensure that there are no duplicate tickers
    symbols = list( set( symbols ) )

    # sort list of symbols
    symbols.sort()

    # print list of symbols
    if verbose:
        print("number of symbols is ",len(symbols))
        print(symbols)

    return symbols


def read_symbols_list_local(json_fn, verbose=False):

    # Get a list of stock tickers stored locally
    from functions.GetParams import get_symbols_file
    existing_symbols_file = get_symbols_file(json_fn)

    print(" ...inside readSymbolList... existing_symbols_file = ", existing_symbols_file)
    with open(existing_symbols_file, "r") as infile:

        symbols = []
        while infile:
            line = infile.readline()
            s = line.split()
            n = len(s)
            if n != 0:
                for i in range(len(s)):
                    s[i] = s[i].replace('.','-')
                    symbols.append(s[i])
            else:
                break

    # ensure that there are no duplicate tickers
    symbols = list( set( symbols ) )

    # remove CASH from the list, if present
    symbols = [symbol for symbol in symbols if symbol.lower() != 'cash']

    # sort list of symbols
    symbols.sort()

    # print list of symbols
    if verbose:
        print("number of symbols is ",len(symbols))
        print(symbols)

    return symbols


def read_company_names_local(json_fn, verbose=False):

    # Get a list of stock tickers stored locally
    from functions.GetParams import get_symbols_file
    existing_company_names_fn = get_symbols_file(json_fn).replace(
            "Naz100_symbols", "companyNames"
        ).replace(
            "SP500_Symbols", "companyNames"
    )

    print(" ...inside readSymbolList... existing_company_names_fn = ", existing_company_names_fn)
    with open(existing_company_names_fn, "r") as infile:
        lines = infile.readlines()
    lines = [x.split("\n")[0] for x in lines]
    lines.sort()

    symbols = []
    company_names = []
    for i in range(len(lines)):
        s = lines[i].split(";")
        symbols.append(s[0])
        company_names.append(s[1])

    # ensure that there are no duplicate tickers
    symbols = list( set( symbols ) )
    company_names = list( set( company_names ) )

    # print list of symbols
    if verbose:
        print("number of symbols is ",len(symbols))
        print(symbols)

    return symbols, company_names


def read_symbols_list_web(json_fn, verbose=True ):

    import os
    # determine if SP500 or Naz100
    from functions.GetParams import get_symbols_file
    existing_symbols_file = get_symbols_file(json_fn)
    symbol_folder, symbol_fn = os.path.split(existing_symbols_file)


    if "Naz100_symbols".lower() in existing_symbols_file.lower():

        ###
        ### Query nasdaq.com for updated list of stocks in Nasdaq 100 index.
        ### Return list with stock tickers.
        ###
        import urllib.request, urllib.parse, urllib.error
        from bs4 import BeautifulSoup

        ###
        ### get current symbol list from wikipedia website
        ###
        try:
            base_url ='https://en.wikipedia.org/wiki/NASDAQ-100#Components'
            soup = BeautifulSoup( urllib.request.urlopen(base_url).read(), "lxml" )
            t = soup.find("table", {"id": "constituents"}) # 2024-10-05

            print("... got web content")
            print("... ran beautiful soup on web content")

            symbolList = [] # store all of the records in this list
            companyNamesList = []
            data=[]
            for row in t.find_all('tr'):
                if str(row)==[]:
                    continue
                if str(row)=="\n":
                    continue
                col = row.find_all('td')
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data.append([ele for ele in cols if ele])
                if col==[]:
                    continue
                company_name = data[-1][0].encode("utf8")
                if company_name[:3]=='MON':
                    break

                company_name = strip_accents(data[-1][0])
                symbol_name = data[-1][1]

                print(" ...symbol_name="+'{0: <5}'.format(symbol_name)+" ...company_name="+company_name)
                companyNamesList.append(company_name)
                symbolList.append(symbol_name)
            print("... retrieved Naz100 companies lists from internet")
            print("\nsymbolList = ", symbolList)

        except:
            ###
            ### something didn't wor. print message and return old list.
            ###
            print("\n\n\n")
            print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
            print(" Nasdaq sysmbols list did not get updated from web.")
            print(" ... check quotes_for_list_adjCloseVol.py in function 'get_Naz100List' ")
            print(" ... also check web at https://en.wikipedia.org/wiki/NASDAQ-100#Components")
            print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
            print("\n\n\n")

            symbolList, companyNamesList = read_company_names_local(json_fn)

    return companyNamesList, symbolList



def get_symbols_changes(json_fn, verbose=False):

    from functions.GetParams import get_symbols_file

    existing_symbols_file = get_symbols_file(json_fn)
    symbol_folder, symbol_fn = os.path.split(existing_symbols_file)
    json_dir = os.path.split(json_fn)[0]

    # get local symbols and company_name lists
    l_symbolList, l_companyNamesList = read_company_names_local(json_fn)

    # get current symbols and company_name lists
    w_companyNamesList, w_symbolList = read_symbols_list_web(json_fn)

    # update local list of company names with version from web
    companyName_file = os.path.join( symbol_folder, "companyNames.txt" )
    with open( companyName_file, "w" ) as f:
        for i in range( len(w_symbolList) ) :
            f.write( w_symbolList[i] + ";" + w_companyNamesList[i] + "\n" )

    ###
    ### compare old list with new list and print changes, if any
    ###

    # file for index changes history
    symbol_change_file = "Naz100_symbolsChanges.txt"
    symbols_changes_file = os.path.join( symbol_folder, symbol_change_file )
    with open(symbols_changes_file, "r+") as f:
        old_symbol_changesList = f.readlines()
    old_symbol_changesListText = ''
    for i in range( len(old_symbol_changesList) ):
        old_symbol_changesListText = old_symbol_changesListText + old_symbol_changesList[i]

    # parse date
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    dateToday = str(year)+"-"+str(month)+"-"+str(day)

    # compare lists to check for tickers removed from the index
    # - printing will be suppressed if "verbose = False"
    removedTickers = []
    print("")
    for i, ticker in enumerate( l_symbolList ):
        if i == 0:
            removedTickersText = ''
        if ticker not in w_symbolList:
            removedTickers.append( ticker )
            if verbose:
                print(" Ticker ", ticker, " has been removed from the Nasdaq100 index")
            removedTickersText = removedTickersText + "\n" + dateToday + " Remove " + ticker

    # compare lists to check for tickers added to the index
    # - printing will be suppressed if "verbose = False"
    addedTickers = []
    print("")
    for i, ticker in enumerate( w_symbolList ):
        if i == 0:
            addedTickersText = ''
        if ticker not in l_symbolList:
            addedTickers.append( ticker )
            if verbose:
                print(" Ticker ", ticker, " has been added to the Nasdaq100 index")
            addedTickersText = addedTickersText + "\n" + dateToday + " Add    " + ticker

    print("")
    with open(symbols_changes_file, "w") as f:
        f.write(addedTickersText)
        f.write(removedTickersText)
        f.write("\n")
        f.write(old_symbol_changesListText)

    print("****************")
    print("addedTickers = ", addedTickers)
    print("removedTickers = ", removedTickers)
    print("****************")
    ###
    ### update symbols file with current list. Keep copy of of list.
    ###

    if removedTickers != [] or addedTickers != []:

        # make copy of previous symbols list file
        symbol_directory = os.path.join( json_dir, "symbols" )
        symbol_file = "Naz100_Symbols.txt"
        archive_symbol_file = "Naz100_symbols__" + str(datetime.date.today()) + ".txt"
        symbols_file = os.path.join( symbol_directory, symbol_file )
        archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

        with open( archive_symbols_file, "w" ) as f:
            for i in range( len(l_symbolList) ) :
                f.write( l_symbolList[i] + "\n" )

        # make new symbols list file
        with open( symbols_file, "w" ) as f:
            for i in range( len(w_symbolList) ) :
                f.write( w_symbolList[i] + "\n" )
        # except:
        #     ###
        #     ### something didn't wor. print message and return old list.
        #     ###
        #     print("\n\n\n")
        #     print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
        #     print(" Nasdaq sysmbols list did not get updated from web.")
        #     print(" ... check quotes_for_list_adjCloseVol.py in function 'get_Naz100List' ")
        #     print(" ... also check web at https://en.wikipedia.org/wiki/NASDAQ-100#Components")
        #     print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
        #     print("\n\n\n")

        #     symbolList = l_symbolList
        #     removedTickers = []
        #     addedTickers = []

    return w_symbolList, removedTickers, addedTickers

