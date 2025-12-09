"""
Student Analysis Module
======================

Business logic for analyzing student concentration and education density in IRIS areas.
Calculates statistics, densities, and aggregations for educational establishments.

Author: Adam Jouini
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for relative imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import geopandas as gpd
import numpy as np

from ..config import CRS_LAMBERT93
from ..utils.spatial import reproject_to_french_mainland, add_area_column, spatial_join_within


def calculate_student_density_stats(education_gdf: gpd.GeoDataFrame,
                                 iris_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate student density statistics per IRIS.

    Args:
        education_gdf: GeoDataFrame with education establishments (with student counts)
        iris_gdf: GeoDataFrame with IRIS polygons

    Returns:
        GeoDataFrame: Statistics per IRIS including establishment counts and student numbers
    """
    print("Calculating student density statistics...")

    # Spatial join: assign establishments to IRIS
    establishments_with_iris = spatial_join_within(
        education_gdf,
        iris_gdf[['code_iris', 'LIB_IRIS', 'geometry']],
        keep_right_cols=['code_iris', 'LIB_IRIS']
    )

    # Count establishments per IRIS
    establishments_count = establishments_with_iris.groupby('code_iris').size().reset_index(name='nb_etabs')

    # Sum students per IRIS
    students_sum = establishments_with_iris.groupby('code_iris')[
        "nb_etudiants"
    ].sum().reset_index(name='nb_etudiants')

    # Merge with IRIS base data
    stats_gdf = iris_gdf[['code_iris', 'LIB_IRIS', 'geometry']].merge(
        establishments_count, on='code_iris', how='left').fillna({'nb_etabs': 0})

    stats_gdf = stats_gdf.merge(students_sum, on='code_iris', how='left').fillna({'nb_etudiants': 0})

    # Ensure proper data types
    stats_gdf['nb_etabs'] = stats_gdf['nb_etabs'].astype(int)
    stats_gdf['nb_etudiants'] = stats_gdf['nb_etudiants'].astype(int)

    # Add area column if not present
    if 'area_km2' not in stats_gdf.columns:
        stats_gdf = add_area_column(stats_gdf, 'area_km2', 'km2')

    # Calculate density metrics
    stats_gdf['etabs_per_km2'] = stats_gdf['nb_etabs'] / stats_gdf['area_km2']
    stats_gdf['students_per_km2'] = stats_gdf['nb_etudiants'] / stats_gdf['area_km2']

    # Handle division by zero and round appropriately
    stats_gdf['etabs_per_km2'] = stats_gdf['etabs_per_km2'].replace([np.inf, -np.inf], 0).round(1)
    stats_gdf['students_per_km2'] = stats_gdf['students_per_km2'].replace([np.inf, -np.inf], 0).round(1)

    print(f"Calculated student stats for {len(stats_gdf)} IRIS")
    return stats_gdf


def get_top_student_areas(stats_gdf: gpd.GeoDataFrame,
                         column: str = 'students_per_km2',
                         top_n: int = 10,
                         ascending: bool = False) -> gpd.GeoDataFrame:
    """
    Get top N IRIS by student-related metric.

    Args:
        stats_gdf: Statistics GeoDataFrame
        column: Column to sort by ('students_per_km2', 'nb_etudiants', 'etabs_per_km2')
        top_n: Number of top items to return
        ascending: Sort order

    Returns:
        DataFrame: Top N rows
    """
    return stats_gdf.nlargest(top_n, column) if not ascending else stats_gdf.nsmallest(top_n, column)


def aggregate_student_statistics(stats_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregate student statistics across all IRIS.

    Args:
        stats_gdf: Statistics GeoDataFrame

    Returns:
        DataFrame: Aggregated statistics
    """
    summary = {
        'total_iris': len(stats_gdf),
        'total_establishments': stats_gdf['nb_etabs'].sum(),
        'total_students': stats_gdf['nb_etudiants'].sum(),
        'iris_with_students': (stats_gdf['nb_etudiants'] > 0).sum(),
        'avg_students_per_iris': round(stats_gdf['nb_etudiants'].mean(), 1),
        'max_students_per_iris': stats_gdf['nb_etudiants'].max(),
        'avg_density_students_per_km2': round(stats_gdf['students_per_km2'].mean(), 1),
        'max_density_students_per_km2': round(stats_gdf['students_per_km2'].max(), 1)
    }

    return pd.DataFrame([summary])


def analyze_concentration_levels(stats_gdf: gpd.GeoDataFrame,
                                density_threshold: float = 100.0) -> pd.DataFrame:
    """
    Analyze student concentration levels based on density thresholds.

    Args:
        stats_gdf: Statistics GeoDataFrame
        density_threshold: Minimum students/km² for "high" concentration

    Returns:
        DataFrame: Concentration analysis
    """
    concentration = stats_gdf.copy()

    # Classify concentration levels
    concentration['concentration_level'] = pd.cut(
        concentration['students_per_km2'],
        bins=[0, density_threshold/4, density_threshold/2, density_threshold, density_threshold*2, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

    return concentration[['code_iris', 'LIB_IRIS', 'students_per_km2', 'nb_etudiants', 'concentration_level']]


def calculate_student_to_establishment_ratio(stats_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculate and analyze student-to-establishment ratios.

    Args:
        stats_gdf: Statistics GeoDataFrame

    Returns:
        DataFrame: Ratio analysis
    """
    ratio_df = stats_gdf.copy()

    # Calculate ratio (avoid division by zero)
    ratio_df['students_per_establishment'] = np.where(
        ratio_df['nb_etabs'] > 0,
        ratio_df['nb_etudiants'] / ratio_df['nb_etabs'],
        0
    ).round(1)

    # Classify establishment size
    ratio_df['establishment_size'] = pd.cut(
        ratio_df['students_per_establishment'],
        bins=[0, 100, 500, 1000, 2000, float('inf')],
        labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
    )

    return ratio_df[['code_iris', 'LIB_IRIS', 'nb_etabs', 'nb_etudiants', 'students_per_establishment', 'establishment_size']]


def identify_student_hubs(stats_gdf: gpd.GeoDataFrame,
                         student_threshold: int = 1000,
                         density_threshold: float = 200.0) -> pd.DataFrame:
    """
    Identify student concentration hubs based on multiple criteria.

    Args:
        stats_gdf: Statistics GeoDataFrame
        student_threshold: Minimum total students
        density_threshold: Minimum density (students/km²)

    Returns:
        DataFrame: Identified hubs
    """
    hubs = stats_gdf[
        (stats_gdf['nb_etudiants'] >= student_threshold) &
        (stats_gdf['students_per_km2'] >= density_threshold)
    ].copy()

    hubs = hubs.sort_values('nb_etudiants', ascending=False)
    return hubs[['code_iris', 'LIB_IRIS', 'nb_etudiants', 'students_per_km2', 'area_km2']]


# Main analysis function (pipeline)
def analyze_student_concentration(iris_gdf: gpd.GeoDataFrame,
                                education_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Complete student concentration analysis pipeline.

    Args:
        iris_gdf: IRIS polygons
        education_gdf: Education establishments with student data

    Returns:
        dict: Analysis results including stats, top areas, and summary
    """
    print("Running complete student concentration analysis...")

    # Calculate stats
    stats = calculate_student_density_stats(education_gdf, iris_gdf)

    # Get top dense areas
    top_dense = get_top_student_areas(stats, 'students_per_km2', 10)

    # Get summary
    summary = aggregate_student_statistics(stats)

    # Concentration analysis
    concentration = analyze_concentration_levels(stats)

    # Ratio analysis
    ratios = calculate_student_to_establishment_ratio(stats)

    # Identify hubs
    hubs = identify_student_hubs(stats)

    results = {
        'stats': stats,
        'top_dense': top_dense,
        'summary': summary,
        'concentration': concentration,
        'ratios': ratios,
        'hubs': hubs
    }

    print("Student analysis completed")
    return results
