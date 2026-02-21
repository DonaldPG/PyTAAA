'''
Created on May 12, 202

@author: donaldpg
'''
import os
import datetime
from typing import Optional, Tuple
import pandas as pd
import pandas_market_calendars as mcal

class webpage_companies_extractor:
    Url = None

    def __init__(self, url: str) -> None:
        self.__url = url

    def get_companies_list(self, current_portfolio: Optional[list] = None) -> Tuple[list, list]:
        dfs = pd.read_html(self.__url, header=0)
        first_table = dfs[2]
        company_names = first_table
        symbols = list(company_names['Ticker'].values)
        companyNames = list(company_names['Company'].values)
        return symbols, companyNames


def strip_accents(text: str) -> str:
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


def readSymbolList(filename: str, json_fn: str, verbose: bool = False) -> list:
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


def read_symbols_list_local(json_fn: str, verbose: bool = False) -> list:

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


def read_company_names_local(json_fn: str, verbose: bool = False) -> Tuple[list, list]:

    # Get a list of stock tickers stored locally
    from functions.GetParams import get_symbols_file
    existing_company_names_fn = get_symbols_file(json_fn).replace(
            "Naz100_symbols", "companyNames"
        ).replace(
            "SP500_Symbols", "companyNames"
    )
    symbol_folder = os.path.split(get_symbols_file(json_fn))[0]
    if "Naz100".lower() in symbol_folder.lower():
        existing_company_names_fn = os.path.join( symbol_folder, "Naz100_companyNames.txt" )
    elif "SP500".lower() in symbol_folder.lower():
        existing_company_names_fn = os.path.join( symbol_folder, "SP500_companyNames.txt" )

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

    # ensure that there are no duplicate tickers while maintaining symbol-company correspondence
    seen_symbols = set()
    unique_symbols = []
    unique_company_names = []
    
    for symbol, company_name in zip(symbols, company_names):
        if symbol not in seen_symbols:
            seen_symbols.add(symbol)
            unique_symbols.append(symbol)
            unique_company_names.append(company_name)
    
    symbols = unique_symbols
    company_names = unique_company_names

    # print list of symbols
    if verbose:
        print("number of symbols is ",len(symbols))
        print(symbols)

    return symbols, company_names


def read_symbols_list_web(json_fn: str, verbose: bool = True) -> Tuple[list, list]:

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
        symbolList = [] # store all of the records in this list
        companyNamesList = []
        data=[]

        try:

            '''
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
                if "<th>" in str(row):
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

                if verbose:
                    print(
                        " ...symbol_name="+'{0: <5}'.format(symbol_name) + \
                        " ...company_name=" + company_name
                    )
                companyNamesList.append(company_name)
                symbolList.append(symbol_name)
            '''
            import pandas as pd

            url = "https://yfiua.github.io/index-constituents/constituents-nasdaq100.csv"
            df = pd.read_csv(url)
            # Sort by 'Age' in descending order
            df = df.sort_values(by='Symbol', ascending=True)
            symbolList = df["Symbol"].values
            companyNamesList = df["Name"].values

            if verbose:
                for _symbol, _company in zip(symbolList, companyNamesList):
                       print(
                           " ...symbol_name="+'{0: <5}'.format(_symbol) + \
                           " ...company_name=" + _company
                       )

            print(f"... retrieved {symbolList.size} Naz100 companies lists from internet")
            print("\nsymbolList = ", symbolList)

            symbolList = list(symbolList)
            companyNamesList = list(companyNamesList)

        except:

            base_url ='https://en.wikipedia.org/wiki/NASDAQ-100#Components'
            # Add User-Agent header to avoid 403 Forbidden errors from Wikipedia
            req = urllib.request.Request(
                base_url,
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
            )
            soup = BeautifulSoup( urllib.request.urlopen(req, timeout=10).read(), "lxml" )
            t = soup.find("table", {"id": "constituents"}) # 2024-10-05

            print("... got web content")
            print("... ran beautiful soup on web content")

            symbolList = [] # store all of the records in this list
            companyNamesList = []
            data=[]
            
            # Check if table was found before attempting to parse
            if t is not None:
                for row in t.find_all('tr'):
                    if str(row)==[]:
                        continue
                    if str(row)=="\n":
                        continue
                    if "<th>" in str(row):
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

                    if verbose:
                        print(
                            " ...symbol_name="+'{0: <5}'.format(symbol_name) + \
                            " ...company_name=" + company_name
                        )
                    companyNamesList.append(company_name)
                    symbolList.append(symbol_name)
            else:
                print("... WARNING: Could not find Nasdaq-100 table on Wikipedia. Will use empty list.")



            # import requests
            # base_url ='https://www.liberatedstocktrader.com/nasdaq-100-companies-list-sector-market-cap/'
            # headers = {
            #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            # }
            # response = requests.get(base_url, headers=headers)
            # soup = BeautifulSoup(response.text, "html.parser")

            # table_list = []
            # for table in soup.find_all("table"):
            #     table_list.append(table)
            # t1 = table_list[1]

            # print("... got web content")
            # print("... ran beautiful soup on web content")

            # symbolList = [] # store all of the records in this list
            # companyNamesList = []
            # data=[]
            # for row in t1.find_all('tr'):
            #     if str(row)==[]:
            #         continue
            #     if str(row)=="\n":
            #         continue
            #     if "<tr stye>" in str(row):
            #         continue
            #     col = row.find_all('td')
            #     cols = row.find_all('td')
            #     cols = [ele.text.strip() for ele in cols]
            #     data.append([ele for ele in cols if ele])
            #     if col==[]:
            #         continue
            #     company_name = data[-1][0].encode("utf8")
            #     if company_name[:3]=='MON':
            #         break

            #     company_name = strip_accents(data[-1][1])
            #     symbol_name = data[-1][0]

            #     if symbol_name.lower() == "ticker":
            #         continue

            #     if verbose:
            #         print(
            #             " ...symbol_name="+'{0: <5}'.format(symbol_name) + \
            #             " ...company_name=" + company_name
            #         )
            #     companyNamesList.append(company_name)
            #     symbolList.append(symbol_name)
            # print("... retrieved Naz100 companies lists from internet")
            # print("\nsymbolList = ", symbolList)

        '''
        except:
            import requests
            url_list = [
                #"https://markets.businessinsider.com/index/components/nasdaq_100",
                #"https://markets.businessinsider.com/index/components/nasdaq_100?p=1",
                "https://markets.businessinsider.com/index/components/nasdaq_100?p=2",
                #"https://markets.businessinsider.com/index/components/nasdaq_100?p=3",
            ]

            symbolList = [] # store all of the records in this list
            companyNamesList = []
            data=[]

            base_url = "https://markets.businessinsider.com/index/components/nasdaq_100"
            for base_url in url_list:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(base_url, headers=headers)
                soup = BeautifulSoup(response.text, "html.parser")

                table_list = []
                for table in soup.find_all("table"):
                    table_list.append(table)
                t1 = table_list[0]

                print("... got web content")
                print("... ran beautiful soup on web content")

                for row in t1.find_all('tr'):
                    if str(row)==[]:
                        continue
                    if str(row)=="\n":
                        continue
                    if "<th class>" in str(row):
                        continue
                    col = row.find_all('td')
                    cols = row.find_all('td')
                    cols = [ele.text.strip() for ele in cols]
                    data.append([ele for ele in cols if ele])
                    ticker = str(row.find_all('a')).split("stocks/")[-1].split("-")[0]
                    if col==[]:
                        continue
                    company_name = data[-1][0].encode("utf8")
                    if company_name[:3]=='MON':
                        break

                    company_name = strip_accents(data[-1][0])
                    symbol_name = ticker.upper()

                    if symbol_name.lower() == "ticker":
                        continue

                    if verbose:
                        print(
                            " ...symbol_name="+'{0: <5}'.format(symbol_name) + \
                            " ...company_name=" + company_name
                        )
                    companyNamesList.append(company_name)
                    symbolList.append(symbol_name)
            print("... retrieved Naz100 companies lists from internet")
            print("\nsymbolList = ", symbolList)

        '''


        if len(companyNamesList) == 0:
            ###
            ### something didn't work. print message and return old list.
            ###
            print("\n\n\n")
            print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
            print(" Nasdaq sysmbols list did not get updated from web.")
            print(" ... check quotes_for_list_adjCloseVol.py in function 'get_Naz100List' ")
            print(" ... also check web at https://en.wikipedia.org/wiki/NASDAQ-100#Components")
            print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
            print("\n\n\n")

            symbolList, companyNamesList = read_company_names_local(json_fn)

    elif "SP500_symbols".lower() in existing_symbols_file.lower():

        ###
        ### Query wikipedia.com for updated list of stocks in S&P 500 index.
        ### Return list with stock tickers.
        ###
        #import urllib2
        import urllib.request, urllib.error, urllib.parse
        import requests
        import re
        from bs4 import BeautifulSoup
        import os
        import datetime
        import unicodedata

        ###
        ### get symbol list from previous period
        ###
        # symbol_directory = os.path.join( os.getcwd(), "symbols" )

        # symbol_file = "SP500_Symbols.txt"
        # symbols_file = os.path.join( symbol_directory, symbol_file )
        existing_symbols_file = get_symbols_file(json_fn)
        symbol_directory, symbol_file = os.path.split(existing_symbols_file)

        with open(existing_symbols_file, "r+") as f:
            old_symbolList = f.readlines()
        for i in range( len(old_symbolList) ) :
            old_symbolList[i] = old_symbolList[i].replace("\n","")

        ###
        ### get current symbol list from wikipedia website
        ###

        # base_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # soup = BeautifulSoup( urllib.request.urlopen(base_url), "lxml" )
        # t = soup.find(
        #     "table",
        #     {
        #         "class" : "wikitable sortable sticky-header",
        #         "id": "constituents"
        #     }
        # )

        # print("... got web content")
        # print("... ran beautiful soup on web content")

        # symbolList = [] # store all of the records in this list
        # companyNamesList = []
        # data=[]
        # n_cos = 0
        # for row in t.find_all('tr'):
        #     if str(row)==[]:
        #         continue
        #     col = row.find_all('td')
        #     cols = row.find_all('td')
        #     cols = [ele.text.strip() for ele in cols]
        #     data.append([ele for ele in cols if ele])
        #     if col==[]:
        #         continue
        #     company_name = data[-1][1].encode("utf8")
        #     # if company_name[:3]=='MON':
        #     #     break
        #     company_name = unicodedata.normalize('NFD',data[-1][1]).encode('ascii', 'ignore').decode('ascii')
        #     symbol_name = data[-1][0].encode("utf8").decode('ascii')
        #     symbol_name = symbol_name.replace(".", "-")

        #     #print(str(row)+"\n ...company_name = "+company_name+"\n ...symbol_name="+symbol_name+"\n")
        #     print("  ...symbol_name=" + f'{symbol_name:<6}'+ " ...company_name = "+company_name)
        #     companyNamesList.append(company_name)
        #     symbolList.append(symbol_name)
        #     n_cos += 1
        # print("... retrieved " + str(n_cos) + " SP500 companies lists from internet")


        ### -----------------
        ### start of revised code
        ### -----------------
        ###
        ### Query wikipedia.com for updated list of stocks in S&P 500 index.
        ### Return list with stock tickers.
        ###
        #import urllib2
        import urllib.request, urllib.error, urllib.parse
        import requests
        import re
        from bs4 import BeautifulSoup
        import os
        import datetime
        import unicodedata

        ###
        ### get symbol list from previous period
        ###
        symbol_directory = os.path.split(existing_symbols_file)[0]

        symbol_file = "SP500_Symbols.txt"
        symbols_file = os.path.join( symbol_directory, symbol_file )
        with open(symbols_file, "r+") as f:
            old_symbolList = f.readlines()
        for i in range( len(old_symbolList) ) :
            old_symbolList[i] = old_symbolList[i].replace("\n","")

        ###
        ### get current symbol list from wikipedia website with enhanced error handling
        ###
        base_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Add User-Agent header to avoid 403 Forbidden errors from Wikipedia
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            req = urllib.request.Request(base_url, headers=headers)
            response = urllib.request.urlopen(req, timeout=15)
            soup = BeautifulSoup(response.read(), "lxml")
            print("... got web content (with headers)")
        except:
            # Fallback to basic request
            try:
                soup = BeautifulSoup( urllib.request.urlopen(base_url), "lxml" )
                print("... got web content (basic request)")
            except Exception as e:
                print(f"... ERROR: Could not fetch Wikipedia page: {e}")
                print("... Using local symbol list instead")
                symbolList = old_symbolList
                companyNamesList = []
                soup = None
        
        if soup is not None:
            print("... ran beautiful soup on web content")
            
            # Try multiple table selection strategies
            t = soup.find("table", {"class" : "wikitable sortable sticky-header", "id": "constituents"})
            
            # If table not found with specific class, try simpler search.
            if t is None:
                print("... table not found with specific class. Trying alternative search...")
                t = soup.find("table", {"id": "constituents"})
            
            # Try even simpler search if still not found
            if t is None:
                print("... trying fallback table search...")
                tables = soup.find_all("table", {"class": "wikitable"})
                for table in tables:
                    table_text = str(table).lower()
                    if 'symbol' in table_text and 'security' in table_text:
                        rows = table.find_all('tr')
                        if len(rows) > 400:  # S&P 500 should have ~500 rows
                            t = table
                            print(f"... found table with {len(rows)} rows using content detection")
                            break

            symbolList = [] # store all of the records in this list
            companyNamesList = []
            data=[]

            # Check if table was found before attempting to parse
            if t is not None:
                for row in t.find_all('tr'):
                    if str(row)==[]:
                        continue
                    col = row.find_all('td')
                    cols = row.find_all('td')
                    cols = [ele.text.strip() for ele in cols]
                    data.append([ele for ele in cols if ele])
                    if col==[]:
                        continue
                    
                    # Skip if not enough columns
                    if len(data[-1]) < 2:
                        continue
                    
                    try:
                        company_name = unicodedata.normalize('NFD',data[-1][1]).encode('ascii', 'ignore').decode('ascii')
                        symbol_name = data[-1][0].encode("utf8").decode('ascii')
                        symbol_name = symbol_name.replace(".", "-")

                        # Skip empty or header rows
                        if not symbol_name or symbol_name.lower() in ['symbol', 'ticker', '', 'company']:
                            continue
                        
                        if verbose:
                            print(
                                " ...symbol_name="+'{0: <5}'.format(symbol_name) + \
                                " ...company_name=" + company_name
                            )
                        companyNamesList.append(company_name)
                        symbolList.append(symbol_name)
                    except (IndexError, UnicodeDecodeError, AttributeError) as e:
                        if verbose:
                            print(f"... error processing row: {e}")
                        continue
            else:
                print("... WARNING: Could not find table on Wikipedia page. Using local list.")
                symbolList = old_symbolList
            
            if len(symbolList) > 0:
                print(f"... retrieved {len(symbolList)} SP500 companies from internet")
            else:
                print("... WARNING: No symbols retrieved, using local list")
                symbolList = old_symbolList

        companyName_file = os.path.join( symbol_directory, "SP500_companyNames.txt" )
        with open( companyName_file, "w" ) as f:
            for i in range( len(symbolList) ) :
                if i < len(companyNamesList):
                    f.write( symbolList[i] + ";" + companyNamesList[i] + "\n" )
                else:
                    f.write( symbolList[i] + ";\n" )
        print("... wrote SP500_companyNames.txt")

        ### -----------------
        ### end of revised code
        ### -----------------

    return companyNamesList, symbolList



def get_symbols_changes(json_fn: str, verbose: bool = False) -> Tuple[list, list, list]:

    from functions.GetParams import get_symbols_file

    existing_symbols_file = get_symbols_file(json_fn)
    symbol_folder, symbol_fn = os.path.split(existing_symbols_file)
    json_dir = os.path.split(json_fn)[0]

    # get local symbols and company_name lists
    l_symbolList, l_companyNamesList = read_company_names_local(json_fn)

    # get current symbols and company_name lists
    w_companyNamesList, w_symbolList = read_symbols_list_web(
        json_fn, verbose=verbose
    )

    # update local list of company names with version from web
    if "Naz100".lower() in symbol_folder.lower():
        companyName_file = os.path.join( symbol_folder, "Naz100_companyNames.txt" )
    elif "SP500".lower() in symbol_folder.lower():
        companyName_file = os.path.join( symbol_folder, "SP500_companyNames.txt" )
    with open( companyName_file, "w" ) as f:
        for i in range( len(w_symbolList) ) :
            f.write( w_symbolList[i] + ";" + w_companyNamesList[i] + "\n" )

    ###
    ### compare old list with new list and print changes, if any
    ###

    # file for index changes history
    if "Naz100".lower() in symbol_fn.lower():
        symbol_change_file = "Naz100_symbolsChanges.txt"
        stock_index = "SP500"
    elif "SP500".lower() in symbol_fn.lower():
        symbol_change_file = "SP500_symbolsChanges.txt"
        stock_index = "SP500"
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
                print(" Ticker ", ticker, " has been removed from the "+ stock_index + " index")
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
        # symbol_directory = os.path.join( json_dir, "symbols" )
        symbol_directory = symbol_folder
        if "Naz100".lower() in symbol_fn.lower():
            symbol_file = "Naz100_Symbols.txt"
            symbol_change_file = "Naz100_symbolsChanges.txt"
            fn_stem = "Naz100_symbols__"
        elif "SP500".lower() in symbol_fn.lower():
            symbol_file = "SP500_Symbols.txt"
            symbol_change_file = "SP500_symbolsChanges.txt"
            fn_stem = "SP500_symbols__"
        symbols_changes_file = os.path.join( symbol_folder, symbol_change_file )

        archive_symbol_file = fn_stem + str(datetime.date.today()) + ".txt"
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


def is_last_trade_day_in_month() -> bool:
    from datetime import datetime, timedelta

    # Get the NYSE calendar
    nyse = mcal.get_calendar('NYSE')

    # Get today's date
    today = datetime.now().date()

    # Get the schedule for the current month
    current_month_schedule = nyse.schedule(
        start_date=today.replace(day=1),
        end_date=today.replace(day=28) + timedelta(days=4)
    )  # Ensure end_date is safely in the next month

    # Get the last trading day of the month
    last_trading_day = current_month_schedule.index[-1].date()

    # Check if today is the last trading day of the month
    is_last_trading_day_of_month = today == last_trading_day

    if is_last_trading_day_of_month:
        print("Today is the last trading day of the month.")
        last_day_of_month = True
    else:
        print("Today is not the last trading day of the month.")
        last_day_of_month = False

    return last_day_of_month