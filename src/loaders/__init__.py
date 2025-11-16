"""
Data Loaders Package
===================

Unified data loading from various sources:
- IRIS (geographical boundaries and names)
- Education establishments and student data
- OSM data (transport stops, commercial establishments)
"""

from .iris_loader import load_iris_rennes, standardize_iris_columns
from .enseignement_loader import load_education_rennes, load_education_france_aggregated
from .osm_loader import load_transport_stops, load_commercial_establishments

__all__ = [
    'load_iris_rennes',
    'standardize_iris_columns',
    'load_education_rennes',
    'load_education_france_aggregated',
    'load_transport_stops',
    'load_commercial_establishments'
]
