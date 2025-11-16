"""
Education Data Loader Module
===========================

Loads and processes higher education establishments data,
including student numbers and geographical coordinates.

Author: Adam Jouini
"""

import pandas as pd
import geopandas as gpd


def load_enseignement_data(url=None):
    """
    Downloads and loads raw higher education establishments data.

    Args:
        url (str, optional): URL to download the CSV file from.

    Returns:
        DataFrame: Raw education establishments data.
    """
    if url is None:
        url = (
            "https://huggingface.co/datasets/analysedonneesfoncieresdata/"
            "analyse_fonciere_data/resolve/main/"
            "fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"
        )

    df = pd.read_csv(url, delimiter=';')
    print(f"Loaded education data: {len(df)} establishments")
    return df


def filter_education_rennes(df, require_gps=True):
    """
    Filters education data for Rennes establishments.

    Args:
        df (DataFrame): Raw education data.
        require_gps (bool): Whether to drop rows without GPS coordinates.

    Returns:
        GeoDataFrame: Filtered and cleaned data for Rennes.
    """
    df_rennes = df[df['Commune'] == 'Rennes'].copy()

    if require_gps:
        df_rennes = df_rennes.dropna(subset=['gps'])
        # Split GPS into lat/lon
        df_rennes[['lat', 'lon']] = df_rennes['gps'].str.split(',', expand=True)
        df_rennes = df_rennes.dropna(subset=['lat', 'lon'])
        df_rennes['lat'] = df_rennes['lat'].astype(float)
        df_rennes['lon'] = df_rennes['lon'].astype(float)

        # Convert to GeoDataFrame
        df_rennes = gpd.GeoDataFrame(
            df_rennes,
            geometry=gpd.points_from_xy(df_rennes.lon, df_rennes.lat),
            crs="EPSG:4326"
        )

    print(f"Filtered for Rennes: {len(df_rennes)} establishments")
    return df_rennes


def aggregate_education_france(df, exclude_overseas=True):
    """
    Aggregates education data by department for France-wide statistics.

    Args:
        df (DataFrame): Raw education data.
        exclude_overseas (bool): Whether to exclude overseas territories.

    Returns:
        DataFrame: Aggregated student numbers by department.
    """
    df_filtered = df if not exclude_overseas else df[~df['département'].str.contains("Étranger", na=False)]

    df_aggregated = df_filtered.groupby('département', as_index=False).agg({
        'nombre total d\'étudiants inscrits hors doubles inscriptions université/CPGE': 'sum',
        'dont femmes': 'sum',
        'dont hommes': 'sum',
        'objectid': 'first',  # Keep first for merging
        'reg': 'first'
    })

    # Add dep code (2 digits)
    df_aggregated['dep'] = df_aggregated['département'].str[:2].str.strip()

    print(f"Aggregated education data for {len(df_aggregated)} departments")
    return df_aggregated


# Convenience functions for common use cases
def load_education_rennes(url=None):
    """
    Load and filter education data for Rennes with GPS coordinates.
    """
    raw_data = load_enseignement_data(url)
    return filter_education_rennes(raw_data)


def load_education_france_aggregated(url=None):
    """
    Load and aggregate education data for France by department.
    """
    raw_data = load_enseignement_data(url)
    return aggregate_education_france(raw_data)


# Backwards compatibility
load_education_DATA_rennes = load_education_rennes
