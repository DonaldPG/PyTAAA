#import sys
#import string
#import re
import ConfigParser

def from_config_file(config_filename):
    with open(config_filename, "r") as fid:
        config = ConfigParser.ConfigParser()
        params = config.readfp(fid)
    return params

def GetParams():
    ######################
    ### Input parameters
    ######################

    # set default values
    defaults = { "Runtime": ["2 days"], "Pausetime": ["1 hours"] }
    params = {}

    # read the parameters form the configuration file
    #config = ConfigParser.ConfigParser()
    config_filename = "C:\\Users\\Don\\PyTAAA\\PyTAAA.params"
    #print "config_filename = ", config_filename

    #params = from_config_file(config_filename)
    config = ConfigParser.ConfigParser(defaults=defaults)
    #config = ConfigParser.ConfigParser()
    configfile = open(config_filename, "r")
    config.readfp(configfile)

    toaddrs = config.get("Email", "To").split()
    fromaddr = config.get("Email", "From").split()
    runtime = config.get("Setup", "Runtime").split()
    pausetime = config.get("Setup", "Pausetime").split()
    
    #print "debug runtime = ", runtime
    #print "debug pausetime = ", pausetime

    if len(runtime) == 1:
        runtime.join('days')
    if len(pausetime) == 1:
        paustime.join('housrs')

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
    params['runtime'] = max_uptime
    params['pausetime'] = seconds_between_runs
    params['numberStocksTraded'] = int( config.get("Valuation", "numberStocksTraded") )
    params['monthsToHold'] = int( config.get("Valuation", "monthsToHold") )
    params['LongPeriod'] = int( config.get("Valuation", "LongPeriod") )
    params['MA1'] = int( config.get("Valuation", "MA1") )
    params['MA2'] = int( config.get("Valuation", "MA2") )
    params['MA3'] = int( config.get("Valuation", "MA3") )
    params['MA2factor'] = float( config.get("Valuation", "sma2factor") )
    params['rankThresholdPct'] = float( config.get("Valuation", "rankThresholdPct") )
    params['riskDownside_min'] = float( config.get("Valuation", "riskDownside_min") )
    params['riskDownside_max'] = float( config.get("Valuation", "riskDownside_max") )
    
    return params

def GetHoldings():
    ######################
    ### Input current holdings and cash
    ######################

    # set default values
    holdings = {}

    # read the parameters form the configuration file
    config_filename = "C:\\Users\\Don\\PyTAAA\\PyTAAA_holdings.params"

    config = ConfigParser.ConfigParser()
    configfile = open(config_filename, "r")
    config.readfp(configfile)

    # put params in a dictionary
    holdings['stocks'] = config.get("Holdings", "stocks").split()
    holdings['shares'] = config.get("Holdings", "shares").split()
    holdings['buyprice'] = config.get("Holdings", "buyprice").split()
   
    return holdings


