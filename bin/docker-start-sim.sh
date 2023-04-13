#!/bin/bash

# This script merely wraps the simulation.py module by assuming that the data folder is in /data and the output in /home/genn

# First, some security measures to avoid running stuff as root

#USER_ID=${LOCAL_USER_ID:-9001}
#GROUP_ID=${LOCAL_USER_ID:-$USER_ID}

#groupadd -g $GROUP_ID genn
#useradd --shell /bin/bash -u $USER_ID -g genn -o -c "" -m genn
export HOME=/home/genn
#chown genn:genn $HOME # make sure it is writeable by genn

# Launch the script
cd $HOME
exec python3 -m beegenn.simulation /data $1
