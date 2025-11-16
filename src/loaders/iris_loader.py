"""
IRIS Data Loader Module
======================

Loads and processes IRIS (Ilots Regroupés pour l'Information Statistique)
geographical data for Rennes, including geometry contours and reference names.

Author: Adam Jouini
"""

import os
import pandas as pd
import geopandas as gpd
import requests
from fiona import listlayers

from ..config import IRIS_GEOMETRIES_URL, IRIS_NAMES_URL, IRIS_GEOMETRIES_LOCAL_PATH
from ..utils.cache import cached_download_dataframe


def load_iris_geometries(url=None, local_path=None):
    """
    Downloads and loads IRIS contour geometries from GPKG file.

    Args:
        url (str, optional): URL to download the GPKG file from.
        local_path (str, optional): Local path to save/retrieve the file.

    Returns:
        GeoDataFrame: IRIS geometries with original attributes.
    """
    if url is None:
        url = IRIS_GEOMETRIES_URL
    if local_path is None:
        local_path = IRIS_GEOMETRIES_LOCAL_PATH

    # Use cached download for metadata, but still handle local file
    # For now, keep file-based caching since GPKG is large
    if not os.path.exists(local_path):
        print(f"Downloading IRIS geometries from {url}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

    layers = listlayers(local_path)
    iris_gdf = gpd.read_file(local_path, layer=layers[0])
    print(f"Loaded {len(iris_gdf)} IRIS geometries")
    return iris_gdf


def load_iris_names(url=None):
    """
    Loads IRIS reference names and codes from Excel file.

    Args:
        url (str, optional): URL to download the Excel file from.

    Returns:
        DataFrame: IRIS codes and names mapping.
    """
    if url is None:
        url = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"

    iris_names = pd.read_excel(url).rename(columns={'CODE_IRIS': 'code_iris'})
    print(f"Loaded reference names for {len(iris_names)} IRIS")
    return iris_names


def load_iris_rennes(url_geometries=None, url_names=None):
    """
    Loads and merges complete IRIS data for Rennes, including geometries and names.

    Args:
        url_geometries (str, optional): URL for geometries
        url_names (str, optional): URL for names

    Returns:
        GeoDataFrame: IRIS data for Rennes with merged names
    """
    print("Loading IRIS data for Rennes...")

    # Load geometries and filter for Rennes
    iris_all = load_iris_geometries(url_geometries)
    iris_rennes = iris_all[iris_all['nom_commune'].str.contains("Rennes", case=False, na=False)].copy()

    # Load names and merge
    iris_names = load_iris_names(url_names)
    iris_rennes = iris_rennes.merge(iris_names[['code_iris', 'LIB_IRIS', 'LIBCOM']],
                                   on='code_iris', how='left')

    # Standardize column names for consistency
    iris_rennes = standardize_iris_columns(iris_rennes)

    print(f"Prepared {len(iris_rennes)} IRIS polygons for Rennes")
    return iris_rennes


def standardize_iris_columns(df):
    """
    Standardize column names for easier use.

    Args:
        df (DataFrame): IRIS data

    Returns:
        DataFrame: Data with standardized column names
    """
    column_mapping = {
        'nom_commune': 'commune_name',
        'nom_iris': 'iris_name',
        'code_insee': 'insee_code',
        'type_iris': 'iris_type',
        'cleabs': 'iris_key'
    }

    df_standardized = df.rename(columns=column_mapping)
    return df_standardized


# Convenience aliases
load_iris_DATA = load_iris_rennes  # Backwards compatibility
