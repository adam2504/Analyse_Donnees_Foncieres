"""
National Student Density Analysis Module
========================================

Analyzes student concentration across French cities at national scale.
Handles city-level aggregations and national statistics.

Author: Adam Jouini
"""

import pandas as pd
from typing import Dict, Any, Optional, List

from ..loaders import load_french_cities_100k
from ..loaders.geocoding_loader import geocode_french_cities


def calculate_national_student_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate various student density metrics for national analysis.

    Args:
        df (DataFrame): DataFrame with student and population data

    Returns:
        dict: Dictionary of calculated metrics
    """
    metrics = {}

    # Basic counts
    metrics['total_cities'] = len(df)
    metrics['total_population'] = df['p21_pop'].sum()
    metrics['total_students'] = df['nb_etudiants'].sum()

    # Average densities
    metrics['avg_student_density'] = df['student_density'].mean()
    metrics['median_student_density'] = df['student_density'].median()
    metrics['max_student_density'] = df['student_density'].max()
    metrics['min_student_density'] = df['student_density'].min()

    # Top cities
    top_by_students = df.nlargest(5, 'nb_etudiants')
    metrics['top_cities_students'] = top_by_students[['libgeo', 'nb_etudiants']].to_dict('records')

    top_by_density = df.nlargest(5, 'student_density')
    metrics['top_cities_density'] = top_by_density[['libgeo', 'student_density']].to_dict('records')

    return metrics


def analyze_student_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the distribution of student concentrations.

    Args:
        df (DataFrame): DataFrame with student density data

    Returns:
        dict: Distribution analysis results
    """
    distribution = {}

    # Quartiles and percentiles
    distribution['quartiles'] = df['student_density'].quantile([0.25, 0.5, 0.75]).to_dict()
    distribution['percentile_90'] = df['student_density'].quantile(0.9)
    distribution['percentile_95'] = df['student_density'].quantile(0.95)

    # Cities above certain thresholds
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]  # 1%, 2%, etc.
    threshold_counts = {}
    for threshold in thresholds:
        count = (df['student_density'] >= threshold).sum()
        threshold_counts[f'cities_above_{threshold:.3f}'] = count

    distribution['threshold_analysis'] = threshold_counts

    return distribution


def classify_cities_by_density(df: pd.DataFrame, thresholds: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Classify cities by their student density levels.

    Args:
        df (DataFrame): DataFrame with student density data
        thresholds (list, optional): Density thresholds for classification

    Returns:
        DataFrame: Original data with added classification column
    """
    if thresholds is None:
        thresholds = [0.005, 0.01, 0.02, 0.05, 0.08]  # 0.5%, 1%, etc.

    df_classified = df.copy()

    # Create labels based on thresholds
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    if len(labels) != len(thresholds) + 1:
        # Adjust labels if needed
        labels = [f'Level {i}' for i in range(len(thresholds) + 1)]

    # Add buffer for infinity
    thresholds_extended = thresholds + [float('inf')]

    df_classified['density_category'] = pd.cut(
        df_classified['student_density'],
        bins=[0] + thresholds_extended,
        labels=labels,
        include_lowest=True
    )

    return df_classified


def generate_city_rankings(df: pd.DataFrame, column: str = 'student_density', top_n: int = 20, ascending: bool = False) -> pd.DataFrame:
    """
    Generate rankings of cities by a specific metric.

    Args:
        df (DataFrame): DataFrame with city data
        column (str): Column to rank by
        top_n (int): Number of top cities to return
        ascending (bool): False for descending (highest first)

    Returns:
        DataFrame: Ranked cities
    """
    ranked = df.sort_values(column, ascending=ascending).head(top_n).copy()
    ranked['rank'] = range(1, len(ranked) + 1)

    # Keep only relevant columns
    main_cols = ['rank', 'libgeo', 'p21_pop', column]
    if 'dep' in ranked.columns:
        main_cols.insert(2, 'dep')

    return ranked[main_cols]


# Main analysis function (national pipeline)
def analyze_national_student_density(data_df: Optional[pd.DataFrame] = None, add_coordinates: bool = True) -> Dict[str, Any]:
    """
    Complete national student density analysis pipeline.

    Args:
        data_df (DataFrame, optional): Pre-loaded data. If None, loads automatically.
        add_coordinates (bool): Whether to add geographical coordinates

    Returns:
        dict: Complete analysis results
    """
    print("🇫🇷 Running national student density analysis...")

    # Load data if not provided
    if data_df is None:
        data_df = load_french_cities_100k()

    # Add coordinates if requested (and not already present)
    if add_coordinates and 'latitude' not in data_df.columns:
        data_df = geocode_french_cities(data_df)

    # Calculate basic metrics
    metrics = calculate_national_student_metrics(data_df)

    # Analyze distribution
    distribution = analyze_student_distribution(data_df)

    # Classify cities
    data_classified = classify_cities_by_density(data_df)

    # Generate rankings
    rank_by_density = generate_city_rankings(data_classified, 'student_density')
    rank_by_students = generate_city_rankings(data_classified, 'nb_etudiants')
    rank_by_population = generate_city_rankings(data_classified, 'p21_pop')

    # Create summary statistics
    summary_stats = {
        'total_cities': len(data_df),
        'total_population': data_df['p21_pop'].sum(),
        'total_students': data_df['nb_etudiants'].sum(),
        'avg_density': data_df['student_density'].mean(),
        'urbanization_rate': (data_df['p21_pop'] > 50000).sum() / len(data_df),
    }
    summary_stats['student_rate'] = summary_stats['total_students'] / summary_stats['total_population']

    results = {
        'data': data_classified,
        'metrics': metrics,
        'distribution': distribution,
        'rankings': {
            'by_density': rank_by_density,
            'by_students': rank_by_students,
            'by_population': rank_by_population
        },
        'summary': summary_stats,
        'geocoded': add_coordinates
    }

    print("✅ National student density analysis completed")
    return results
