'''
Created on May 12, 202

@author: donaldpg
'''

def readSymbolList(filename,verbose=False):
    # Get the Data
    infile = file(filename,"r")
    symbols = []
    while infile:
        line = infile.readline()
        s = line.split()
        n = len(s)
        if n <> 0:
            for i in range(len(s)):
                symbols.append(s[i])
        else:
            break

    # print list of symbols
    if verbose:
        print "number of symbols is ",len(symbols)
        print symbols

    return symbols
