
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
    config_filename = "PyTAAA.params"

    #params = from_config_file(config_filename)
    config = ConfigParser.ConfigParser(defaults=defaults)
    configfile = open(config_filename, "r")
    config.readfp(configfile)

    toaddrs = config.get("Email", "To").split()
    fromaddr = config.get("Email", "From").split()
    pw = config.get("Email", "PW")
    runtime = config.get("Setup", "Runtime").split()
    pausetime = config.get("Setup", "Pausetime").split()


    if len(runtime) == 1:
        runtime.join('days')
    if len(pausetime) == 1:
        paustime.join('hours')

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
    params['PW'] = str(pw)
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


def GetFTPParams():
    ######################
    ### Input FTP parameters
    ######################

    # set default values
    ftpparams = {}

    # read the parameters form the configuration file
    config_filename = "PyTAAA.params"

    config = ConfigParser.ConfigParser()
    configfile = open(config_filename, "r")
    config.readfp(configfile)

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

    config = ConfigParser.ConfigParser()
    configfile = open(config_filename, "r")
    config.readfp(configfile)

    # put params in a dictionary
    holdings['stocks'] = config.get("Holdings", "stocks").split()
    holdings['shares'] = config.get("Holdings", "shares").split()
    holdings['buyprice'] = config.get("Holdings", "buyprice").split()
    holdings['cumulativecashin'] = config.get("Holdings", "cumulativecashin").split()

    return holdings


def GetStatus():
    ######################
    ### Input current cumulative value
    ######################

    # read the parameters form the configuration file
    status_filename = "PyTAAA_status.params"

    config = ConfigParser.ConfigParser()
    configfile = open(status_filename, "r")
    config.readfp(configfile)

    # put params in a dictionary
    status = config.get("Status", "cumu_value").split()[-1]

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
        old_cumu_status = f.read()
    old_cumu_status = old_cumu_status.split("\n")[-2]
    old_cumu_status = old_cumu_status.split(" ")[-1]

    if str(cumu_status) != old_cumu_status:
        with open(status_filename, 'a') as f:
            f.write( "cumu_value: "+str(datetime.datetime.now())+" "+str(cumu_status)+"\n" )

    return


def GetIP( ):
    ######################
    ### Input current cumulative value
    ######################

    import urllib
    import re
    f = urllib.urlopen("http://www.canyouseeme.org/")
    html_doc = f.read()
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
    elif 'XPS' in platform.uname()[1] :
        edition = 'XPS'
    elif 'AMD64' in platform.uname()[4] :
        edition = 'Windows64'
    else:
        edition = 'none'

    return edition
