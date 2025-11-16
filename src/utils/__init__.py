"""
Utilities Package
================

Common utilities and helpers:
- Configuration management
- Caching with parquet storage
- Spatial operations (reprojection, density calculations, joins)
"""

from . import cache
from . import spatial

__all__ = ['cache', 'spatial']
