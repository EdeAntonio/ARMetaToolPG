import os
import toml

# Conveniences to other module directories via relative paths
ARMT_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

ARMT_ASSETS_DATA_DIR = os.path.join(ARMT_ASSETS_DIR, "data")
"""Path to the extension data directory."""

from .robots import *