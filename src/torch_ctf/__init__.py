"""CTF calculation for cryoEM in torch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-ctf")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"
