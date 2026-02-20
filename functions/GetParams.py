"""Configuration parameter extraction from JSON and config files.

This module provides functions to read and extract various configuration
parameters from JSON configuration files used throughout the PyTAAA system.
It handles paths, FTP settings, valuation parameters, and status tracking.

Key Functions:
    get_json_params: Load main valuation parameters from JSON config
    get_symbols_file: Get path to stock symbols list file
    get_performance_store: Get path to performance history storage
    get_webpage_store: Get path to webpage output directory
    get_web_output_dir: Get web output directory path
    get_holdings: Extract portfolio holdings from params file
    get_status: Read cumulative status tracking data
    put_status: Write cumulative status tracking data

Legacy Functions (deprecated, use JSON-based equivalents):
    GetParams: Legacy config file reader
    GetHoldings: Legacy holdings reader
    GetStatus: Legacy status reader
"""

import os
import numpy as np
import configparser
import json
import re
from typing import Tuple, Dict, Optional


def from_config_file(config_filename: str):
    """Load configuration from an INI-style config file.
    
    Args:
        config_filename: Path to configuration file
        
    Returns:
        ConfigParser object with parsed configuration
        
    Note:
        This is a legacy function. New code should use get_json_params().
    """
    with open(config_filename, "r") as fid:
        config = configparser.ConfigParser(strict=False)
        params = config.read_file(fid)
    return params


def get_symbols_file(json_fn: str) -> str:
    """Get path to file containing list of stock symbols to process.
    
    Reads the JSON configuration to determine which symbol list to use
    (Naz100 or SP500) and constructs the full path to the symbols file.
    
    Args:
        json_fn: Path to JSON configuration file
        
    Returns:
        str: Full path to symbols file (e.g., "symbols/Naz100_Symbols.txt")
        
    Example:
        >>> symbols_file = get_symbols_file("config/pytaaa_naz100_pine.json")
        >>> print(symbols_file)
        '/path/to/symbols/Naz100_Symbols.txt'
    """
    ######################
    ### get filename where list of symbols is stored
    ######################
    ##
    ##  Import list of symbols to process.
    ##

    params = get_json_params(json_fn)
    stockList = params['stockList']

    if "symbols_file" in params.keys():
        symbols_file = params["symbols_file"]
    else:
        # read list of symbols from disk.
        top_dir = os.path.split(json_fn)[0]
        symbol_directory = os.path.join( top_dir, "symbols" )
        if stockList == 'Naz100':
            symbol_file = "Naz100_Symbols.txt"
        elif stockList == 'SP500':
            symbol_file = "SP500_Symbols.txt"
        symbols_file = os.path.join( symbol_directory, symbol_file )

    return symbols_file


def get_performance_store(json_fn: str) -> str:
    """Get path to directory where performance history files are stored.
    
    Performance history files (*.params files) contain backtest results,
    portfolio allocations, and trading history for each model configuration.
    
    Args:
        json_fn: Path to JSON configuration file
        
    Returns:
        str: Path to performance_store directory from config
        
    Raises:
        FileNotFoundError: If json_fn doesn't exist
        KeyError: If Valuation section or performance_store key missing
        
    Example:
        >>> store = get_performance_store("config/pytaaa_sp500_pine.json")
        >>> print(store)
        '/Users/user/pyTAAA_data_static/sp500_pine/data_store'
    """
    ######################
    ### get folder where performance history files (*.params) are stored
    ######################
    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)

    # Access and print different sections
    valuation_section = config.get('Valuation')
    p_store = valuation_section["performance_store"]

    return p_store


def get_webpage_store(json_fn: str) -> str:
    """Get path to directory where updated webpage files are created.
    
    The webpage directory contains HTML files, plots, and other assets
    for displaying portfolio recommendations and performance metrics.
    
    Args:
        json_fn: Path to JSON configuration file
        
    Returns:
        str: Path to webpage directory from config
        
    Raises:
        FileNotFoundError: If json_fn doesn't exist
        KeyError: If Valuation section or webpage key missing
        
    Example:
        >>> webpage = get_webpage_store("config/pytaaa_naz100_hma.json")
        >>> print(webpage)
        '/Users/user/pyTAAA_data_static/naz100_hma/webpage'
    """
    ######################
    ### get folder where updated webpage is created
    ######################

    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)

    # Access and print different sections
    valuation_section = config.get('Valuation')
    w_store = valuation_section["webpage"]

    return w_store


def get_web_output_dir(json_fn: str) -> str:
    """
    Get web output directory from JSON configuration.
    
    Args:
        json_fn (str): Path to the JSON configuration file.
        
    Returns:
        str: The web output directory path.
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        KeyError: If the web_output_dir key is missing.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)
    
    if 'web_output_dir' not in config:
        raise KeyError("'web_output_dir' key not found in JSON configuration")
    
    return config['web_output_dir']


def get_central_std_values(json_fn: str) -> Dict[str, Dict[str, float]]:
    """
    Get normalization values (central_values and std_values) from JSON configuration.
    
    Args:
        json_fn (str): Path to the JSON configuration file.
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing central_values and 
                                    std_values for normalization calculations.
                                    
    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        KeyError: If required normalization keys are missing.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)
    
    # Navigate through the nested JSON structure
    model_selection = config.get('model_selection')
    if model_selection is None:
        raise KeyError("'model_selection' section not found in JSON configuration")
    
    normalization = model_selection.get('normalization')
    if normalization is None:
        raise KeyError("'normalization' section not found in model_selection")
    
    central_values = normalization.get('central_values')
    std_values = normalization.get('std_values')
    
    if central_values is None:
        raise KeyError("'central_values' not found in normalization section")
    if std_values is None:
        raise KeyError("'std_values' not found in normalization section")
    
    return {
        'central_values': central_values,
        'std_values': std_values
    }


def get_json_ftp_params(json_fn, verbose=False):
    ######################
    ### Input FTP parameters from json file with multiple sections
    ######################

    # set default values
    ftpparams = {}

    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)

    # Access and print different sections
    ftp_section = config.get('FTP')

    if verbose:
        print("\nFTP Section:")
        print(ftp_section)

    ftpHostname = config.get("FTP")["hostname"]
    ftpUsername = config.get("FTP")["username"]
    ftpPassword = config.get("FTP")["password"]
    ftpRemotePath = config.get("FTP")["remotepath"]
    ftpRemoteIP   = config.get("FTP")["remoteIP"]

    # put params in a dictionary
    ftpparams['ftpHostname'] = str( ftpHostname )
    ftpparams['ftpUsername'] = str( ftpUsername )
    ftpparams['ftpPassword'] = str( ftpPassword )
    ftpparams['remotepath'] = str( ftpRemotePath )
    ftpparams['remoteIP'] = str( ftpRemoteIP )

    return ftpparams


def get_holdings(json_fn):
    ######################
    ### Input current holdings and cash
    ######################

    # set default values
    holdings = {}

    # read the parameters form the configuration file
    # params = get_json_params(json_fn)
    params_folder = os.path.split(json_fn)[0]
    p_store = get_performance_store(json_fn)
    config_filename = os.path.join(p_store, "PyTAAA_holdings.params")

    config = configparser.ConfigParser(strict=False)
    configfile = open(config_filename, "r")
    config.read_file(configfile)

    # put params in a dictionary
    holdings['stocks'] = config.get("Holdings", "stocks").split()
    holdings['shares'] = config.get("Holdings", "shares").split()
    holdings['buyprice'] = config.get("Holdings", "buyprice").split()
    holdings['cumulativecashin'] = config.get("Holdings", "cumulativecashin").split()

    # get rankings for latest dates for all stocks in index
    # read the parameters form the configuration file
    print(" ...inside GetHoldings...  p_store = ", p_store)
    config_filename = os.path.join(p_store, "PyTAAA_ranks.params")
    configfile = open(config_filename, "r")
    config.read_file(configfile)
    symbols = config.get("Ranks", "symbols").split()
    ranks = config.get("Ranks", "ranks").split()
    # put ranks params in dictionary
    holdings_ranks = []
    print("\n\n********************************************************")
    print(" ...inside GetParams/GetHoldings...")
    for i, holding in enumerate(holdings['stocks']):
        for j,symbol in enumerate(symbols):
            # print("... j, symbol, rank = ", j, symbol, ranks[j])
            if symbol == holding:
                print("                                       MATCH ... i, symbol, rank = ", i, holding, symbols[j], ranks[j])
                holdings_ranks.append( ranks[j] )
                break
    holdings['ranks'] = holdings_ranks
    print("\n\n********************************************************")

    return holdings


def get_json_params(json_fn, verbose=False):

    ######################
    ### Input parameters from json file with multiple sections
    ######################

    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)

    # Access and print different sections
    email_section = config.get('Email')
    text_from_email_section = config.get('Text_from_email')
    ftp_section = config.get('FTP')
    stock_server_section = config.get('stock_server')
    setup_section = config.get('Setup')
    valuation_section = config.get('Valuation')

    if verbose:
        print("Email Section:")
        print(email_section)

        print("\nText from Email Section:")
        print(text_from_email_section)

        print("\nFTP Section:")
        print(ftp_section)

        print("\nStock Server Section:")
        print(stock_server_section)

        print("\nSetup Section:")
        print(setup_section)

        print("\nValuation Section:")
        print(valuation_section)

    # set default values
    params = {}

    toaddrs = config.get("Email")["To"]
    fromaddr = config.get("Email")["From"]
    toSMS = config.get("Text_from_email")["phoneEmail"]
    send_texts = config.get("Text_from_email")["send_texts"]
    pw = config.get("Email")["PW"]
    runtime = config.get("Setup")["runtime"]
    pausetime = config.get("Setup")["pausetime"]

    quote_server = config.get("stock_server")["quote_download_server"]

    if len(runtime) == 1:
        runtime.join('days')
    if len(pausetime) == 1:
        pausetime.join('hours')

    if runtime.split(" ")[1] == 'seconds':
        factor = 1
    elif runtime.split(" ")[1] == 'minutes':
        factor = 60
    elif runtime.split(" ")[1] == 'hours':
        factor = 60*60
    elif runtime.split(" ")[1] == 'days':
        factor = 60*60*24
    elif runtime.split(" ")[1] == 'months':
        factor = 60*60*24*30.4
    elif runtime.split(" ")[1] == 'years':
        factor = 6060*60*60*24*365.25
    else:
        # assume days
        factor = 60*60*24

    max_uptime = int(runtime.split(" ")[0]) * factor

    if pausetime.split(" ")[1] == 'seconds':
        factor = 1
    elif pausetime.split(" ")[1] == 'minutes':
        factor = 60
    elif pausetime.split(" ")[1] == 'hours':
        factor = 60*60
    elif pausetime.split(" ")[1] == 'days':
        factor = 60*60*24
    elif pausetime.split(" ")[1] == 'months':
        factor = 60*60*24*30.4
    elif pausetime.split(" ")[1] == 'years':
        factor = 60*60*60*24*365.25
    else:
        # assume hour
        factor = 60*60

    seconds_between_runs = int(pausetime.split(" ")[0]) * factor

    # put params in a dictionary
    params['fromaddr'] = str(fromaddr)
    params['toaddrs'] = str(toaddrs)
    params['toSMS'] = toSMS
    if str(send_texts).lower() == 'true':
        params['send_texts'] = True
    elif str(send_texts).lower() == 'false':
        params['send_texts'] = False
    params['PW'] = str(pw)
    params['runtime'] = max_uptime
    params['pausetime'] = seconds_between_runs
    params['quote_server'] = quote_server
    params['numberStocksTraded'] = int(config.get("Valuation")["numberStocksTraded"])
    params['trade_cost'] = float(config.get("Valuation")["trade_cost"])
    params['monthsToHold'] = int(config.get("Valuation")["monthsToHold"])
    params['LongPeriod'] = int( config.get("Valuation")["LongPeriod"])
    params['stddevThreshold'] = float( config.get("Valuation")["stddevThreshold"])
    params['MA1'] = int( config.get("Valuation")["MA1"])
    params['MA2'] = int( config.get("Valuation")["MA2"])
    params['MA3'] = int( config.get("Valuation")["MA3"])
    params['MA2offset'] = params['MA3'] - params['MA2']
    params['MA2factor'] = float( config.get("Valuation")["sma2factor"])
    params['rankThresholdPct'] = float( config.get("Valuation")["rankThresholdPct"])
    params['riskDownside_min'] = float( config.get("Valuation")["riskDownside_min"])
    params['riskDownside_max'] = float( config.get("Valuation")["riskDownside_max"])
    params['narrowDays'] = [
        float(config.get("Valuation")["narrowDays_min"]),
        float(config.get("Valuation")["narrowDays_max"])
    ]
    params['mediumDays'] = [
        float(config.get("Valuation")["mediumDays_min"]),
        float(config.get("Valuation")["mediumDays_max"])
    ]
    params['wideDays'] = [
        float(config.get("Valuation")["wideDays_min"]),
        float(config.get("Valuation")["wideDays_max"])
    ]
    params['uptrendSignalMethod'] = config.get("Valuation")["uptrendSignalMethod"]
    params['lowPct'] = config.get("Valuation")["lowPct"]
    params['hiPct'] = config.get("Valuation")["hiPct"]

    valuation_section = config.get("Valuation")
    params['minperiod'] = int( valuation_section.get("minperiod", 10))
    params['maxperiod'] = int( valuation_section.get("maxperiod", 100))
    params['incperiod'] = int( valuation_section.get("incperiod", 10))
    params['numdaysinfit'] = int( valuation_section.get("numdaysinfit", 100))
    params['numdaysinfit2'] = int( valuation_section.get("numdaysinfit2", 200))
    params['offset'] = int( valuation_section.get("offset", 0))

    params['stockList'] = config.get("Valuation")["stockList"]
    params['symbols_file'] = config.get("Valuation")["symbols_file"]

    # Rolling window data-quality filter settings.
    # These keys exist in the JSON Valuation section but were never read,
    # causing dailyBacktest.py to skip the filter unconditionally
    # (it defaulted enable_rolling_filter to False).
    params['enable_rolling_filter'] = bool(
        valuation_section.get('enable_rolling_filter', False)
    )
    params['window_size'] = int(valuation_section.get('window_size', 50))

    return params


def get_json_status(json_fn):
    ######################
    ### Input current cumulative value
    ######################

    # read the parameters form the configuration file
    # json_folder = os.path.split(json_fn)[0]
    json_folder = get_performance_store(json_fn)
    status_filename = os.path.join(json_folder, "PyTAAA_status.params")

    config = configparser.ConfigParser(strict=False)
    configfile = open(status_filename, "r")
    config.read_file(configfile)

    # put params in a dictionary
    status = config.get("Status", "cumu_value").split()[-3]

    return status


def compute_long_hold_signal(json_fn):
    ######################
    ### compute signal based on MA of system portfolio value
    ######################

    import numpy as np
    import datetime
    from functions.TAfunctions import dpgchannel, SMA

    def uniqueify2lists(seq, seq2):
       # order preserving
       # uniqueness and order determined by seq
       seen = {}
       result = []
       result2 = []
       for i,item in enumerate(seq):
           marker = item
           # in old Python versions:
           # if seen.has_key(marker)
           # but in new ones:
           if marker in seen: continue
           seen[marker] = 1
           result.append(item)
           result2.append(seq2[i])
       return result,result2

    # json_dir = os.path.split(json_fn)[0]
    # filepath = os.path.join( json_dir, "PyTAAA_status.params" )
    json_folder = get_performance_store(json_fn)
    filepath = os.path.join(json_folder, "PyTAAA_status.params")

    date = []
    value = []
    #try:
    with open( filepath, "r" ) as f:
        # get number of lines in file
        lines = f.read().split("\n")
        numlines = len (lines)
        for i in range(numlines):
            #try:
            statusline = lines[i]
            statusline_list = (statusline.split("\r")[0]).split(" ")
            if len( statusline_list ) >= 4:
                date.append( datetime.datetime.strptime( statusline_list[1], '%Y-%m-%d') )
                value.append( float(statusline_list[3]) )
            #except:
            #   break

            #print "rankingMessage -----"
            #print rankingMessage
    # '''
    # except:
    #     print " Error: unable to read updates from PyTAAA_status.params"
    #     print ""
    # '''

    value = np.array( value ).astype('float')

    #print "\n\n\ndate = ", date
    #print "\n\n\nvalue = ", value

    # '''
    # # calculate mid-channel and compare to MA
    # dailyValue = [ value[-1] ]
    # dailyDate = [ date[-1] ]
    # for ii in range( len(value)-2, 0, -1 ):
    #     if date[ii] != date[ii+1]:
    #         dailyValue.append( value[ii] )
    #         dailyDate.append( date[ii] )
    # '''
    # '''
    # sortindices = (np.array( dailyDate )).argsort()
    # sortedDailyValue = (np.array( dailyValue ))[ sortindices ]
    # sortedDailyDate = (np.array( dailyDate ))[ sortindices ]
    # print 'sortindices = ', sortindices
    # '''
    # reverse date and value to keep most recent values
    sortedDailyDate, sortedDailyValue = uniqueify2lists(date[::-1],value[::-1])
    # reverse results so most recent are last
    sortedDailyDate, sortedDailyValue = sortedDailyDate[::-1], sortedDailyValue[::-1]

    #print "\n\n\nsortedDailyValue = ", sortedDailyValue

    minchannel, maxchannel = dpgchannel( sortedDailyValue, 5, 18, 4 )
    midchannel = ( minchannel + maxchannel )/2.
    MA_midchannel = SMA( midchannel, 5 )

    # create signal 11,000 for 'long' and 10,001 for 'cash'
    signal = np.ones_like( sortedDailyValue ) * 11000.
    signal[ MA_midchannel > midchannel ] = 10001
    signal[0] = 11000.

    # apply trading signal to portfolio values
    _gainloss = np.array(sortedDailyValue)[1:] / np.array(sortedDailyValue)[:-1]
    _gainloss = np.hstack(( (1.), _gainloss ))
    _gainloss -= 1.
    last_signal = (signal/11000.).astype('int')
    _gainloss *= last_signal.astype('float')
    _gainloss += 1
    _gainloss[0]=sortedDailyValue[0]
    traded_values = np.cumprod(_gainloss)

    #return dailyDate[-1], format(int(signal[-1]/11000.),'-2d')
    return sortedDailyDate, traded_values, sortedDailyValue, (signal/11000.).astype('int')


def get_status(json_fn):
    ######################
    ### Input current cumulative value
    ######################

    # read the parameters form the configuration file
    # json_dir = os.path.split(json_fn)[0]
    p_store = get_performance_store(json_fn)
    status_filename = os.path.join(p_store, "PyTAAA_status.params")

    config = configparser.ConfigParser(strict=False)
    configfile = open(status_filename, "r")
    config.read_file(configfile)

    # put params in a dictionary
    status = config.get("Status", "cumu_value").split()[-3]

    return status


def put_status(cumu_status, json_fn):
    ######################
    ### Input current cumulative value
    ######################

    import datetime

    # read the parameters form the configuration file
    # json_dir = os.path.split(json_fn)[0]
    p_store = get_performance_store(json_fn)
    status_filename = os.path.join(p_store, "PyTAAA_status.params")

    # check last value written to file for comparison with current cumu_status. Update if different.
    with open(status_filename, 'r') as f:
        lines = f.read()
    old_cumu_status = lines.split("\n")[-2]
    #old_cumu_status = old_cumu_status.split(" ")[-1]
    old_cumu_status = old_cumu_status.split(" ")[-3]

    old_cumu_signal = lines.split("\n")[-2]
    old_cumu_signal = old_cumu_signal.split(" ")[-2]

    # check current signal based on system protfolio value trend
    # _, traded_values, _, last_signal = computeLongHoldSignal()
    _, traded_values, _, last_signal = compute_long_hold_signal(json_fn)

    print("cumu_status = ", str(cumu_status))
    print("old_cumu_status = ", old_cumu_status)
    print("last_signal[-1] = ", last_signal[-1])
    print("old_cumu_signal = ", old_cumu_signal)
    print(str(cumu_status)== old_cumu_status, str(last_signal[-1])== old_cumu_signal)
    if str(cumu_status) != str(old_cumu_status) or str(last_signal[-1]) != str(old_cumu_signal):
        with open(status_filename, 'a') as f:
            f.write( "cumu_value: "+\
                     str(datetime.datetime.now())+" "+\
                     str(cumu_status)+" "+\
                     str(last_signal[-1])+" "+\
                     str(traded_values[-1])+"\n" )

    return



def computeLongHoldSignal():
    ######################
    ### compute signal based on MA of system portfolio value
    ######################

    import numpy as np
    import datetime
    from functions.TAfunctions import dpgchannel, SMA

    def uniqueify2lists(seq, seq2):
       # order preserving
       # uniqueness and order determined by seq
       seen = {}
       result = []
       result2 = []
       for i,item in enumerate(seq):
           marker = item
           # in old Python versions:
           # if seen.has_key(marker)
           # but in new ones:
           if marker in seen: continue
           seen[marker] = 1
           result.append(item)
           result2.append(seq2[i])
       return result,result2

    filepath = os.path.join( os.getcwd(), "PyTAAA_status.params" )

    date = []
    value = []
    #try:
    with open( filepath, "r" ) as f:
        # get number of lines in file
        lines = f.read().split("\n")
        numlines = len (lines)
        for i in range(numlines):
            #try:
            statusline = lines[i]
            statusline_list = (statusline.split("\r")[0]).split(" ")
            if len( statusline_list ) >= 4:
                date.append( datetime.datetime.strptime( statusline_list[1], '%Y-%m-%d') )
                value.append( float(statusline_list[3]) )
            #except:
            #   break

            #print "rankingMessage -----"
            #print rankingMessage
    '''
    except:
        print " Error: unable to read updates from PyTAAA_status.params"
        print ""
    '''

    value = np.array( value ).astype('float')

    #print "\n\n\ndate = ", date
    #print "\n\n\nvalue = ", value

    '''
    # calculate mid-channel and compare to MA
    dailyValue = [ value[-1] ]
    dailyDate = [ date[-1] ]
    for ii in range( len(value)-2, 0, -1 ):
        if date[ii] != date[ii+1]:
            dailyValue.append( value[ii] )
            dailyDate.append( date[ii] )
    '''
    '''
    sortindices = (np.array( dailyDate )).argsort()
    sortedDailyValue = (np.array( dailyValue ))[ sortindices ]
    sortedDailyDate = (np.array( dailyDate ))[ sortindices ]
    print 'sortindices = ', sortindices
    '''
    # reverse date and value to keep most recent values
    sortedDailyDate, sortedDailyValue = uniqueify2lists(date[::-1],value[::-1])
    # reverse results so most recent are last
    sortedDailyDate, sortedDailyValue = sortedDailyDate[::-1], sortedDailyValue[::-1]

    #print "\n\n\nsortedDailyValue = ", sortedDailyValue

    minchannel, maxchannel = dpgchannel( sortedDailyValue, 5, 18, 4 )
    midchannel = ( minchannel + maxchannel )/2.
    MA_midchannel = SMA( midchannel, 5 )

    # create signal 11,000 for 'long' and 10,001 for 'cash'
    signal = np.ones_like( sortedDailyValue ) * 11000.
    signal[ MA_midchannel > midchannel ] = 10001
    signal[0] = 11000.

    # apply trading signal to portfolio values
    _gainloss = np.array(sortedDailyValue)[1:] / np.array(sortedDailyValue)[:-1]
    _gainloss = np.hstack(( (1.), _gainloss ))
    _gainloss -= 1.
    last_signal = (signal/11000.).astype('int')
    _gainloss *= last_signal.astype('float')
    _gainloss += 1
    _gainloss[0]=sortedDailyValue[0]
    traded_values = np.cumprod(_gainloss)

    #return dailyDate[-1], format(int(signal[-1]/11000.),'-2d')
    return sortedDailyDate, traded_values, sortedDailyValue, (signal/11000.).astype('int')

def GetIP( ):
    ######################
    ### Input current cumulative value
    ######################

    import urllib.request, urllib.parse, urllib.error
    import re
    f = urllib.request.urlopen("http://www.canyouseeme.org/")
    html_doc = f.read().decode('utf-8')
    f.close()
    m = re.search(r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
                  r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
                  r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
                  r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',html_doc)
    return m.group(0)

def GetEdition( ):
    ######################
    ### Input current cumulative value
    ######################

    import platform

    # get edition from where software is running
    if 'armv6l' in platform.uname()[4] :
        edition = 'pi'
    elif 'x86' in platform.uname()[4] :
        edition = 'Windows32'
    elif 'AMD64' in platform.uname()[4] :
        edition = 'Windows64'
    elif 'arm64' in platform.uname()[4] :
        edition = 'MacOS'
    else:
        edition = 'none'

    return edition


def GetSymbolsFile(json_fn=None):
    """
    Get filename where list of symbols is stored (legacy interface).
    
    This function provides backward compatibility. New code should use
    get_symbols_file(json_fn) instead.
    
    Args:
        json_fn: Path to JSON configuration file. If None, tries to find
                a configuration file in common locations.
                
    Returns:
        str: Path to symbols file
    """
    if json_fn is None:
        # Try to find JSON config in common locations
        possible_configs = [
            'pytaaa_generic.json',
            os.path.join(os.path.expanduser('~'), 'pyTAAA_data', 'pytaaa_generic.json')
        ]
        for config in possible_configs:
            if os.path.exists(config):
                json_fn = config
                break
        if json_fn is None:
            raise FileNotFoundError("No JSON config file found. Please specify json_fn parameter.")
    
    return get_symbols_file(json_fn)

def parse_pytaaa_status(file_path: str) -> Tuple[list, list]:
    """
    Parse the PyTAAA_status.params file to extract date and portfolio value columns.

    Args:
        file_path (str): Path to the PyTAAA_status.params file.

    Returns:
        Tuple[list, list]: A tuple containing two lists: dates and portfolio values.
    """
    dates = []
    portfolio_values = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) < 4:
                continue  # Skip lines that do not have enough columns
            try:
                dates.append(parts[0])
                portfolio_values.append(float(parts[3]))
            except ValueError as e:
                print(f"Skipping line due to parsing error: {line.strip()}\nError: {e}")

    return dates, portfolio_values

def validate_model_choices(model_choices: dict) -> dict:
    """
    Validate the paths in the `model_choices` dictionary to ensure all required files are present.

    Args:
        model_choices (dict): A dictionary where keys are model names and values are file paths.

    Returns:
        dict: A dictionary with validation results for each model.
    """
    validation_results = {}
    for model, path in model_choices.items():
        if path:
            validation_results[model] = os.path.exists(path)
        else:
            validation_results[model] = True  # Cash model has no file path
    return validation_results
