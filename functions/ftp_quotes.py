# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 19:06:54 2018

@author: dp
"""
from typing import Optional


def ftp_copy_quotes_hdf(computerName: str, local_path: str, json_fn: Optional[str] = None) -> None:

    # based on a demo in ptyhon package paramiko.
    #
    import os
    import getpass
    import sys
    import traceback
    import paramiko
    from functions.GetParams import get_json_ftp_params

    # get hostname and credentials
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
            print("*** Error: No JSON config file found. Please specify json_fn parameter.")
            sys.exit(1)
    
    ftpparams = get_json_ftp_params(json_fn)
    print("\n\n\n ... ftpparams = ", ftpparams, "\n\n\n")
    hostname = ftpparams['ftpHostname']
    hostIP   = ftpparams['remoteIP']
    username = ftpparams['ftpUsername']
    password = ftpparams['ftpPassword']

    if hostname == '':
        hostname = input('Hostname: ')
    if len(hostname) == 0:
        print('*** Hostname required.')
        sys.exit(1)
    port = 22
    if hostname.find(':') >= 0:
        hostname, portstr = hostname.split(':')
        port = int(portstr)

    # get username
    if username == '':
        default_username = getpass.getuser()
        username = input('Username [%s]: ' % default_username)
        if len(username) == 0:
            username = default_username

    # now, connect and use paramiko Transport to negotiate SSH2
    # across the connection
    try:
        print(' connecting to remote server')
        t = paramiko.Transport((hostIP, port))
        t.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(t)

        transfer_list = ['/home/pi/pyTAAApi/symbols/Naz100_Symbols_.hdf5',
                         '/home/pi/pyTAAApi/symbols/Naz100_Symbols.txt',
                         '/home/pi/pyTAAApi/symbols/companyNames.txt']

        for i, remote_file in enumerate(transfer_list):
            _, remote_file_noPath = os.path.split(remote_file)
            local_file = os.path.join(local_path, remote_file_noPath)
            print("\nstarting file ", i)
            print("...remote_file = ", remote_file)
            print("...local_file = ", local_file, os.path.isfile(local_file))
            sftp.get(remote_file, local_file)
            print('  ...created '+local_file+' on '+computerName)

        t.close()

    except Exception as e:
        print('*** Caught exception: %s: %s' % (e.__class__, e))
        traceback.print_exc()
        try:
            t.close()
        except:
            pass
        sys.exit(1)

    return


#if __name__ == "__main__":
def copy_updated_quotes(json_fn: Optional[str] = None) -> None:

    import os
    import platform

    ###
    ### test file copy from raspberry pi that has updated stock quotes
    ###

    # cd to target path on local machine
    # check the machine name to set local target path

    operatingSystem = platform.system()
    architecture = platform.uname()[4]
    computerName = platform.uname()[1]

    print("  ...platform: ", operatingSystem, architecture, computerName)

    if operatingSystem == 'Linux' and architecture == 'armv6l':
        # raspberry pi
        print("  ...on raspberry pi")

    elif operatingSystem == 'Windows' and computerName == 'DonEnvy' \
            and os.chdir() == 'C:\\Users\\dp\\raspberrypi\\PyTAAADL_tracker':

        print("  ...on dpg_envy, PyTAAADL_tracker")
        local_folder = 'C:\\Users\\dp\\raspberrypi\\PyTAAADL_tracker\\symbols'
        ftp_copy_quotes_hdf(computerName, local_folder, json_fn)

        '''
        try:
            print "  ...on dpg_envy, PyTAAADL_tracker"
            local_folder = u'C:\\Users\\dp\\raspberrypi\\PyTAAADL_tracker\\symbols'
            ftp_copy_quotes_hdf(computerName, local_folder)
        except:
            print "Could not update quotes hdf from pi pyTAAApi..."
        '''

    elif operatingSystem == 'Windows' and computerName == 'Spectre' \
            and os.getcwd() == 'C:\\Users\\Don\\Desktop\\pine_backup\\PyTAAA-analyzestocksPy3\\py3':

        print("  ...on dpg_envy, pine64 methods, PyTAAA_web")
        local_folder = 'C:\\Users\\Don\\Desktop\\pine_backup\\PyTAAA-analyzestocksPy3\\py3\\symbols'
        ftp_copy_quotes_hdf(computerName, local_folder, json_fn)


    elif operatingSystem == 'Linux' and computerName == 'pine64' :
        try:
            print("  ...on pine, PyTAAA-analyzestocks")
            local_folder = 'C:\\Users\\dp\\raspberrypi\\PyTAAADL_tracker\\symbols'
            ftp_copy_quotes_hdf(computerName, local_folder, json_fn)
        except:
            print("Could not update quotes hdf from pi pyTAAApi...")

    else:
        print("Could not place web files on server...")
