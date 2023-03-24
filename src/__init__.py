import sys

if sys.version_info < (3, 10, 0):
    sys.stderr.write("You need Python 3.10.0 or later to run this script\n")
    exit(1)