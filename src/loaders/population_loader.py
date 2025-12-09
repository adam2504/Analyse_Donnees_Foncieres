"""
French Population Data Loader Module
===================================

Loads and processes population data for French communes and cities.
Handles filtering, aggregation, and geographical processing.

Author: Adam Jouini
"""

import pandas as pd
from typing import Optional

from ..config import EDUCATION_DATA_URL
from .enseignement_loader import load_education_france_aggregated


def load_population_data(url: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw population data for French communes.

    Args:
        url (str, optional): URL to download population data

    Returns:
        DataFrame: Raw population data
    """
    if url is None:
        url = (
            "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/"
            "resolve/main/POPULATION_MUNICIPALE_COMMUNES_FRANCE.xlsx"
        )

    print("📥 Loading French population data...")
    df = pd.read_excel(url)

    # Drop unnecessary historical columns
    cols_to_drop = [col for col in df.columns if col.startswith(('p13_', 'p14_', 'p15_', 'p16_', 'p17_', 'p18_', 'p19_', 'p20_')) and col.endswith('_pop')]
    df.drop(cols_to_drop, axis=1, inplace=True)

    print(f"✅ Loaded population data: {len(df)} communes")
    return df


def clean_city_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize city names (handle Paris, Lyon, Marseille arrondissements).

    Args:
        df (DataFrame): Population data with city names

    Returns:
        DataFrame: Data with cleaned city names
    """
    df_clean = df.copy()

    # Replace arrondissement names with main city names
    df_clean['libgeo'] = df_clean['libgeo'].replace(
        to_replace=[r'Paris.*', r'Lyon.*', r'Marseille.*'],
        value=['Paris', 'Lyon', 'Marseille'],
        regex=True,
    )

    return df_clean


def aggregate_population_by_city(df: pd.DataFrame, min_population: Optional[int] = None) -> pd.DataFrame:
    """
    Aggregate population data by city (libgeo).

    Args:
        df (DataFrame): Population data
        min_population (int, optional): Minimum population threshold

    Returns:
        DataFrame: Aggregated population by city
    """
    # Clean city names first
    df_clean = clean_city_names(df)

    # Group by city
    df_grouped = df_clean.groupby('libgeo', as_index=False).agg({
        'objectid': 'first',
        'reg': 'first',
        'dep': 'first',
        'cv': 'first',
        'codgeo': 'first',
        'p21_pop': 'sum'  # Sum populations for cities with multiple communes
    })

    # Apply population filter if specified
    if min_population is not None:
        df_grouped = df_grouped[df_grouped['p21_pop'] > min_population]

    # Sort by population descending
    df_grouped = df_grouped.sort_values(by='p21_pop', ascending=False)

    print(f"✅ Aggregated to {len(df_grouped)} cities" +
          (f" with population > {min_population:,}" if min_population else ""))

    return df_grouped

def merge_population_education(pop_df: pd.DataFrame, edu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge population and education data by department.

    Args:
        pop_df (DataFrame): Population data
        edu_df (DataFrame): Education data

    Returns:
        DataFrame: Merged data with student density
    """
    print("🔗 Merging population and education data...")

    # Merge on department code
    df_joined = pd.merge(pop_df, edu_df, on='dep', how='inner')

    # Calculate student density
    df_joined['student_density'] = (
        df_joined['nb_etudiants']
        / df_joined['p21_pop']
    )

    print(f"✅ Merge successful: {len(df_joined)} cities with student data")
    return df_joined


# Convenience functions for common use cases
def load_french_cities_100k(url_pop: Optional[str] = None, url_edu: Optional[str] = None) -> pd.DataFrame:
    """
    Load and merge data for French cities with > 100k population.
    """
    # Load and aggregate data
    pop_raw = load_population_data(url_pop)
    edu_agg = load_education_france_aggregated(url_edu)

    # Aggregate population by city
    cities_df = aggregate_population_by_city(pop_raw, min_population=100000)

    # Merge with education data
    result_df = merge_population_education(cities_df, edu_agg)

    return result_df


def load_french_cities_all(url_pop: Optional[str] = None, url_edu: Optional[str] = None) -> pd.DataFrame:
    """
    Load and merge data for all French cities (no population filter).
    """
    pop_raw = load_population_data(url_pop)
    edu_agg = load_education_france_aggregated(url_edu)

    cities_df = aggregate_population_by_city(pop_raw)

    result_df = merge_population_education(cities_df, edu_agg)

    return result_df


# Backwards compatibility
charger_donnees_population = load_french_cities_100k  # Old function name
