import os
import numpy as np
import configparser
import json

def from_config_file(config_filename):
    with open(config_filename, "r") as fid:
        config = configparser.ConfigParser(strict=False)
        params = config.read_file(fid)
    return params


def get_symbols_file(json_fn):
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


def get_performance_store(json_fn):
    ######################
    ### get folder where performance history files (*.params) are stored
    ######################
    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)

    # Access and print different sections
    valuation_section = config.get('Valuation')
    p_store = valuation_section["performance_store"]

    return p_store


def get_webpage_store(json_fn):
    ######################
    ### get folder where updated webpage is created
    ######################

    with open(json_fn, 'r') as json_file:
        config = json.load(json_file)

    # Access and print different sections
    valuation_section = config.get('Valuation')
    w_store = valuation_section["webpage"]

    return w_store


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
        factor = 6060*60*24*365.25
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
        factor = 6060*60*24*365.25
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

    params['minperiod'] = int( config.get("Valuation")["minperiod"])
    params['maxperiod'] = int( config.get("Valuation")["maxperiod"])
    params['incperiod'] = int( config.get("Valuation")["incperiod"])
    params['numdaysinfit'] = int( config.get("Valuation")["numdaysinfit"])
    params['numdaysinfit2'] = int( config.get("Valuation")["numdaysinfit2"])
    params['offset'] = int( config.get("Valuation")["offset"])

    params['stockList'] = config.get("Valuation")["stockList"]
    params['symbols_file'] = config.get("Valuation")["symbols_file"]

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


def GetParams():
    ######################
    ### Input parameters
    ######################

    # set default values
    defaults = { "Runtime": ["2 days"], "Pausetime": ["1 hours"] }
    params = {}

    # read the parameters form the configuration file
    config_filename = "PyTAAA.params"

    #params = from_config_file(config_filename)
    config = configparser.ConfigParser(strict=False, defaults=defaults)
    configfile = open(config_filename, "r")
    config.read_file(configfile)

    toaddrs = config.get("Email", "To").split()
    fromaddr = config.get("Email", "From").split()
    toSMS = config.get("Text_from_email", "phoneEmail").split()
    send_texts = config.get("Text_from_email", "send_texts").split()
    pw = config.get("Email", "PW")
    runtime = config.get("Setup", "Runtime").split()
    pausetime = config.get("Setup", "Pausetime").split()

    quote_server = config.get("stock_server", "quote_download_server")

    if len(runtime) == 1:
        runtime.join('days')
    if len(pausetime) == 1:
        pausetime.join('hours')

    if runtime[1] == 'seconds':
        factor = 1
    elif runtime[1] == 'minutes':
        factor = 60
    elif runtime[1] == 'hours':
        factor = 60*60
    elif runtime[1] == 'days':
        factor = 60*60*24
    elif runtime[1] == 'months':
        factor = 60*60*24*30.4
    elif runtime[1] == 'years':
        factor = 6060*60*24*365.25
    else:
        # assume days
        factor = 60*60*24

    max_uptime = int(runtime[0]) * factor

    if pausetime[1] == 'seconds':
        factor = 1
    elif pausetime[1] == 'minutes':
        factor = 60
    elif pausetime[1] == 'hours':
        factor = 60*60
    elif pausetime[1] == 'days':
        factor = 60*60*24
    elif pausetime[1] == 'months':
        factor = 60*60*24*30.4
    elif pausetime[1] == 'years':
        factor = 6060*60*24*365.25
    else:
        # assume hour
        factor = 60*60

    seconds_between_runs = int(pausetime[0]) * factor

    # put params in a dictionary
    params['fromaddr'] = str(fromaddr[0])
    params['toaddrs'] = str(toaddrs[0])
    params['toSMS'] = toSMS[0]
    if send_texts[0].lower() == 'true':
        params['send_texts'] = True
    elif send_texts[0].lower() == 'false':
        params['send_texts'] = False
    params['PW'] = str(pw)
    params['runtime'] = max_uptime
    params['pausetime'] = seconds_between_runs
    params['quote_server'] = quote_server
    params['numberStocksTraded'] = int( config.get("Valuation", "numberStocksTraded") )
    params['trade_cost'] = float( config.get("Valuation", "trade_cost") )
    params['monthsToHold'] = int( config.get("Valuation", "monthsToHold") )
    params['LongPeriod'] = int( config.get("Valuation", "LongPeriod") )
    params['stddevThreshold'] = float( config.get("Valuation", "stddevThreshold") )
    params['MA1'] = int( config.get("Valuation", "MA1") )
    params['MA2'] = int( config.get("Valuation", "MA2") )
    params['MA3'] = int( config.get("Valuation", "MA3") )
    params['MA2offset'] = params['MA3'] - params['MA2']
    params['MA2factor'] = float( config.get("Valuation", "sma2factor") )
    params['rankThresholdPct'] = float( config.get("Valuation", "rankThresholdPct") )
    params['riskDownside_min'] = float( config.get("Valuation", "riskDownside_min") )
    params['riskDownside_max'] = float( config.get("Valuation", "riskDownside_max") )
    params['narrowDays'] = [ float(config.get("Valuation", "narrowDays_min")), float(config.get("Valuation", "narrowDays_max")) ]
    params['mediumDays'] = [ float(config.get("Valuation", "mediumDays_min")), float(config.get("Valuation", "mediumDays_max")) ]
    params['wideDays'] = [ float(config.get("Valuation", "wideDays_min")), float(config.get("Valuation", "wideDays_max")) ]
    params['uptrendSignalMethod'] = config.get("Valuation", "uptrendSignalMethod")
    params['lowPct'] = config.get("Valuation", "lowPct")
    params['hiPct'] = config.get("Valuation", "hiPct")

    params['minperiod'] = int( config.get("Valuation", "minperiod") )
    params['maxperiod'] = int( config.get("Valuation", "maxperiod") )
    params['incperiod'] = int( config.get("Valuation", "incperiod") )
    params['numdaysinfit'] = int( config.get("Valuation", "numdaysinfit") )
    params['numdaysinfit2'] = int( config.get("Valuation", "numdaysinfit2") )
    params['offset'] = int( config.get("Valuation", "offset") )

    params['stockList'] = config.get("Valuation", "stockList")

    return params


def GetFTPParams():
    ######################
    ### Input FTP parameters
    ######################

    # set default values
    ftpparams = {}

    # read the parameters form the configuration file
    config_filename = "PyTAAA.params"

    config = configparser.ConfigParser(strict=False)
    configfile = open(config_filename, "r")
    config.read_file(configfile)

    ftpHostname = config.get("FTP", "hostname")
    ftpUsername = config.get("FTP", "username")
    ftpPassword = config.get("FTP", "password")
    ftpRemotePath = config.get("FTP", "remotepath")
    ftpRemoteIP   = config.get("FTP", "remoteIP")

    # put params in a dictionary
    ftpparams['ftpHostname'] = str( ftpHostname )
    ftpparams['ftpUsername'] = str( ftpUsername )
    ftpparams['ftpPassword'] = str( ftpPassword )
    ftpparams['remotepath'] = str( ftpRemotePath )
    ftpparams['remoteIP'] = str( ftpRemoteIP )

    return ftpparams


def GetHoldings():
    ######################
    ### Input current holdings and cash
    ######################

    # set default values
    holdings = {}

    # read the parameters form the configuration file
    config_filename = "PyTAAA_holdings.params"

    config = configparser.ConfigParser(strict=False)
    configfile = open(config_filename, "r")
    config.read_filefp(configfile)

    # put params in a dictionary
    holdings['stocks'] = config.get("Holdings", "stocks").split()
    holdings['shares'] = config.get("Holdings", "shares").split()
    holdings['buyprice'] = config.get("Holdings", "buyprice").split()
    holdings['cumulativecashin'] = config.get("Holdings", "cumulativecashin").split()

    # get rankings for latest dates for all stocks in index
    # read the parameters form the configuration file
    print(" ...inside GetHoldings...  pwd = ", os.getcwd())
    config_filename = "PyTAAA_ranks.params"
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


def GetStatus():
    ######################
    ### Input current cumulative value
    ######################

    # read the parameters form the configuration file
    status_filename = "PyTAAA_status.params"

    config = configparser.ConfigParser(strict=False)
    configfile = open(status_filename, "r")
    config.read_file(configfile)

    # put params in a dictionary
    status = config.get("Status", "cumu_value").split()[-3]

    return status


def PutStatus( cumu_status ):
    ######################
    ### Input current cumulative value
    ######################

    import datetime

    # read the parameters form the configuration file
    status_filename = "PyTAAA_status.params"

    # check last value written to file for comparison with current cumu_status. Update if different.
    with open(status_filename, 'r') as f:
        lines = f.read()
    old_cumu_status = lines.split("\n")[-2]
    #old_cumu_status = old_cumu_status.split(" ")[-1]
    old_cumu_status = old_cumu_status.split(" ")[-3]

    old_cumu_signal = lines.split("\n")[-2]
    old_cumu_signal = old_cumu_signal.split(" ")[-2]

    # check current signal based on system protfolio value trend
    _, traded_values, _, last_signal = computeLongHoldSignal()

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
    m = re.search('(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',html_doc)
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


def GetSymbolsFile( ):
    ######################
    ### get filename where list of symbols is stored
    ######################
    ##
    ##  Import list of symbols to process.
    ##
    params = GetParams()
    stockList = params['stockList']

    # read list of symbols from disk.
    symbol_directory = os.path.join( os.getcwd(), "symbols" )
    if stockList == 'Naz100':
        symbol_file = "Naz100_Symbols.txt"
    elif stockList == 'SP500':
        symbol_file = "SP500_Symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )

    return symbols_file
