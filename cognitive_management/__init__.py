"""Compatibility exports for the renamed public cognitive types module."""
import sys
from . import cognitive_types as types

sys.modules.setdefault(__name__ + ".types", types)
