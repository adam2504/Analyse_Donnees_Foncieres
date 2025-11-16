"""
Data Loaders Package
===================

Unified data loading from various sources:
- IRIS (geographical boundaries and names)
- Education establishments and student data
- OSM data (transport stops, commercial establishments)
- Population data for French cities
- Geocoding services
"""

from .iris_loader import load_iris_rennes
from .enseignement_loader import load_education_rennes, load_education_france_aggregated
from .osm_loader import load_transport_stops, load_commercial_establishments
from .population_loader import load_french_cities_100k, aggregate_population_by_city
from .geocoding_loader import geocode_french_cities, ajouter_coordonnees

__all__ = [
    'load_iris_rennes',
    'load_education_rennes',
    'load_education_france_aggregated',
    'load_transport_stops',
    'load_commercial_establishments',
    'load_french_cities_100k',
    'aggregate_population_by_city',
    'geocode_french_cities',
    'ajouter_coordonnees'
]
