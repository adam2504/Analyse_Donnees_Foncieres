"""
Education Data Loader Module
===========================

Loads and processes higher education establishments data,
including student numbers and geographical coordinates.

Author: Adam Jouini
"""

import pandas as pd
import geopandas as gpd
from typing import Optional, Union


def load_enseignement_data(url: Optional[str] = None) -> pd.DataFrame:
    """
    Downloads and loads raw higher education establishments data.

    Args:
        url (str, optional): URL to download the CSV file from.

    Returns:
        DataFrame: Raw education establishments data with standardized column names.
    """
    if url is None:
        url = (
            "https://huggingface.co/datasets/analysedonneesfoncieresdata/"
            "analyse_fonciere_data/resolve/main/"
            "fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"
        )

    df = pd.read_csv(url, delimiter=';')

    # Standardize column names immediately for consistency
    df = standardize_education_columns(df)

    print(f"Loaded education data: {len(df)} establishments")
    return df


def filter_education_rennes(df: pd.DataFrame, require_gps: bool = True) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
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


def standardize_education_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for easier use.

    Args:
        df (DataFrame): Education data

    Returns:
        DataFrame: Data with standardized column names
    """
    # Find the student column dynamically (it contains 'étudiants')
    student_cols = [col for col in df.columns if 'étudiants' in col.lower()]
    if student_cols:
        column_mapping = {student_cols[0]: "nb_etudiants"}
    else:
        column_mapping = {}

    df_standardized = df.rename(columns=column_mapping)
    return df_standardized


def aggregate_education_france(df: pd.DataFrame, exclude_overseas: bool = True) -> pd.DataFrame:
    """
    Aggregates education data by department for France-wide statistics.

    Args:
        df (DataFrame): Raw education data (already standardized).
        exclude_overseas (bool): Whether to exclude overseas territories.

    Returns:
        DataFrame: Aggregated student numbers by department.
    """
    df_filtered = df if not exclude_overseas else df[~df['département'].str.contains("Étranger", na=False)]

    df_aggregated = df_filtered.groupby('département', as_index=False).agg({
        "nb_etudiants": 'sum',  # Now using standardized column name
        'dont femmes': 'sum',
        'dont hommes': 'sum'
    })

    # Add dep code (2 digits)
    df_aggregated['dep'] = df_aggregated['département'].str[:2].str.strip()

    # Column names are already standardized from load_enseignement_data()
    print(f"Aggregated education data for {len(df_aggregated)} departments")
    return df_aggregated


# Convenience functions for common use cases
def load_education_rennes(url: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Load and filter education data for Rennes with GPS coordinates.
    """
    raw_data = load_enseignement_data(url)
    filtered_data = filter_education_rennes(raw_data)
    return standardize_education_columns(filtered_data)


def load_education_france_aggregated(url: Optional[str] = None) -> pd.DataFrame:
    """
    Load and aggregate education data for France by department.
    """
    raw_data = load_enseignement_data(url)
    return aggregate_education_france(raw_data)


# Backwards compatibility
load_education_DATA_rennes = load_education_rennes
