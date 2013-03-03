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

    return str(fromaddr[0]), str(toaddrs[0]), max_uptime, seconds_between_runs
