"""
Configuration Module
===================

Centralized configuration for URLs, paths, timeouts, and default values used across the project.

Author: Adam Jouini
"""

import os

# Base directory for project (makes paths relative to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data URLs
IRIS_GEOMETRIES_URL = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/contours-iris-pe.gpkg"
IRIS_NAMES_URL = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"
EDUCATION_DATA_URL = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"

# Default paths (relative to project root)
IRIS_GEOMETRIES_LOCAL_PATH = os.path.join(BASE_DIR, "data", "contours-iris-pe.gpkg")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache", "")

# Coordinate Reference Systems
CRS_WGS84 = "EPSG:4326"  # WGS84 lat/lon
CRS_LAMBERT93 = "EPSG:2154"  # Lambert 93 for French mainland distances/areas

# OSM Query timeouts and parameters
OSM_TIMEOUT = 120  # seconds
OSM_ADMIN_LEVEL = "8"  # for communes

# Cache settings
CACHE_EXPIRY_HOURS = 24  # Default cache expiry if not specified
CACHE_FORMAT = "parquet"  # parquet, pickle, etc.

# Plotting defaults
DEFAULT_FIG_SIZE = (14, 8)
DEFAULT_DPI = 100

# Logging level (for future use)
LOG_LEVEL = "INFO"
