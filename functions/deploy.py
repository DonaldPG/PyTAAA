"""Web asset deployment utilities for PyTAAA.

This module provides two file-transfer strategies for pushing the
generated HTML dashboard and associated assets to their serving
location:

- :func:`ftpMoveDirectory` — SFTP via ``paramiko`` for remote Linux
  servers (Windows hosts or Linux ``pine64`` machines).
- :func:`piMoveDirectory` — local ``shutil`` copy for Raspberry Pi
  deployments where the web server root is a locally mounted path.

Example::

    from functions.deploy import piMoveDirectory
    piMoveDirectory("pytaaa_model_switching_params.json")
"""

import shutil

from functions.GetParams import get_json_ftp_params


def ftpMoveDirectory(json_fn: str) -> None:
    """Transfer web output files to a remote host via SFTP.

    Reads SFTP connection details from the JSON configuration file and
    uses the ``paramiko`` library to upload all files in the local
    ``./pyTAAA_web`` directory (plus holdings and status parameter
    files) to the configured remote path.

    During market hours (08:00–15:00 local time) PNG files that are not
    named ``PyTAAA*`` and all ``.db`` files are excluded from the
    transfer to reduce bandwidth.

    Args:
        json_fn: Path to the JSON configuration file containing FTP
            credentials and remote path under keys ``ftpHostname``,
            ``remoteIP``, ``ftpUsername``, ``ftpPassword``, and
            ``remotepath``.

    Returns:
        None

    Raises:
        SystemExit: If the hostname is empty after prompting or if an
            unrecoverable SFTP error occurs.

    Notes:
        This function is only invoked on Windows hosts or Linux
        ``pine64`` machines.  On ARM Linux (Raspberry Pi) use
        :func:`piMoveDirectory` instead.
    """
    import datetime
    import getpass
    import os
    import sys
    import traceback

    import paramiko

    # Get hostname and credentials.
    ftpparams = get_json_ftp_params(json_fn)
    print("\n\n\n ... ftpparams = ", ftpparams, "\n\n\n")
    hostname = ftpparams['ftpHostname']
    hostIP   = ftpparams['remoteIP']
    username = ftpparams['ftpUsername']
    password = ftpparams['ftpPassword']
    remote_path = ftpparams['remotepath']

    if hostname == '' :
        hostname = input('Hostname: ')
    if len(hostname) == 0:
        print('*** Hostname required.')
        sys.exit(1)
    port = 22
    if hostname.find(':') >= 0:
        hostname, portstr = hostname.split(':')
        port = int(portstr)

    # Get username.
    if username == '':
        default_username = getpass.getuser()
        username = input('Username [%s]: ' % default_username)
        if len(username) == 0:
            username = default_username

    # Connect and use paramiko Transport to negotiate SSH2.
    try:
        print(' connecting to remote server')
        t = paramiko.Transport((hostIP, port))
        t.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(t)

        try:
            sftp.mkdir( remote_path )
        except IOError:
            print('  ...'+remote_path+' already exists)')

        sftp.open(
            os.path.join(remote_path, 'README'), 'w'
        ).write(
            'This was created by pyTAAA/WriteWebPage.py on DonXPS\n'
        )

        transfer_list = os.listdir("./pyTAAA_web")
        transfer_list.append( '../PyTAAA_holdings.params' )
        transfer_list.append( '../PyTAAA_status.params' )

        # Remove PNG files during market hours (08:00–15:00) to reduce
        # bandwidth; also skip all .db files.
        today = datetime.datetime.now()
        hourOfDay = today.hour
        if 8 < hourOfDay < 15 :
            for i in range( len(transfer_list)-1,-1,-1 ):
                name, extension = os.path.splitext( transfer_list[i] )
                if (
                    extension == ".png" and "PyTAAA" not in name
                ) or extension == '.db':
                    transfer_list.pop(i)

        for i, local_file in enumerate(transfer_list):
            _, local_file_noPath = os.path.split( local_file )
            remote_file = os.path.join( remote_path, local_file_noPath )
            sftp.put(
                os.path.join("./pyTAAA_web/", local_file), remote_file
            )
            print('  ...created '+remote_file+' on piDonaldPG')

        t.close()

    except Exception as e:
        print('*** Caught exception: %s: %s' % (e.__class__, e))
        traceback.print_exc()
        try:
            t.close()
        except Exception:
            pass
        sys.exit(1)


def piMoveDirectory(json_fn: str) -> None:
    """Copy web output files to a locally accessible deployment path.

    Reads the remote (target) path from the JSON configuration file and
    uses ``shutil.copyfile`` to copy all files in ``./pyTAAA_web`` plus
    holdings and status parameter files to that path.  During market
    hours (08:00–15:00 local time) PNG files not named
    ``PyTAAA_value`` are excluded to avoid overwriting live charts.

    All errors are caught and printed rather than raised so that a
    deployment failure does not abort the main pipeline.

    Args:
        json_fn: Path to the JSON configuration file containing the
            deployment target under the key ``remotepath``.

    Returns:
        None

    Notes:
        This function is intended for Raspberry Pi deployments where
        the web server root is a locally mounted path.  For remote SFTP
        transfers use :func:`ftpMoveDirectory` instead.
    """
    import datetime
    import os

    try:
        # Get remote path location.
        ftpparams = get_json_ftp_params(json_fn)
        remote_path = ftpparams['remotepath']

        print("\n\n ... diagnostic:  ftpparams = ", ftpparams)

        # Create target directory if it does not exist.
        try:
            os.mkdirs( remote_path )
        except Exception:
            print('  ...'+remote_path+' already exists)')

        print("\n\n ... diagnostic:  remote_path = ", remote_path)

        # Create README in target directory.
        try:
            with open( os.path.join(remote_path, 'README'), 'w') as f:
                f.write(
                    'This was created by pyTAAA/WriteWebPage_pi.py'
                    ' on piDonaldPG\n'
                )
                print(
                    '  ...'
                    + os.path.join(remote_path, 'README')
                    + ' created'
                )
        except Exception:
            print(
                '  ...'
                + os.path.join(remote_path, 'README')
                + ' could not be created. Maybe already exists?'
            )

        # Build list of files to copy.
        source_directory = "./pyTAAA_web"
        transfer_list = os.listdir( source_directory )
        transfer_list.append(
            os.path.join( '..', 'PyTAAA_holdings.params' )
        )
        transfer_list.append(
            os.path.join( '..', 'PyTAAA_status.params' )
        )

        print("\n\n ... diagnostic:  transfer_list = ", transfer_list)

        # Skip non-PyTAAA_value PNG files during market hours.
        today = datetime.datetime.now()
        hourOfDay = today.hour
        if 8 < hourOfDay < 15 :
            for i in range( len(transfer_list)-1,-1,-1 ):
                name, extension = os.path.splitext( transfer_list[i] )
                if extension == ".png" and name != "PyTAAA_value" :
                    transfer_list.pop(i)

        print(
            "\n\n ... updated diagnostic:  transfer_list = ",
            transfer_list,
        )

        for f in transfer_list:
            local_file = os.path.join( source_directory, f )
            _, local_file_noPath = os.path.split( local_file )
            remote_file = os.path.join( remote_path, local_file_noPath )
            print(
                "\n ... diagnostic:  local_file, remote_file = ",
                local_file,
                remote_file,
            )
            shutil.copyfile( local_file, remote_file )
            print(
                '  ...created ' + remote_file + ' on piDonaldPG web server'
            )

    except Exception:
        print(" Unable to create updated web page...")
