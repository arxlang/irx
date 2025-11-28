"""Top-level package for IRx."""

__author__ = """Ivan Ogasawara"""
__email__ = "ivan.ogasawara@gmail.com"

# Dynamically retrieve version from package metadata
from importlib.metadata import version as _version

__version__ = _version("pyirx")  # semantic-release
