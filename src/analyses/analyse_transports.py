"""
Transport Analysis Module
========================

Business logic for analyzing public transport density and coverage in IRIS areas.
Calculates statistics, densities, and aggregations for transport stops.

Author: Adam Jouini
"""

import sys
from pathlib import Path

# Add parent directory to path for relative imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import geopandas as gpd
import numpy as np

from ..config import CRS_LAMBERT93
from ..utils.spatial import reproject_to_french_mainland, add_area_column, spatial_join_within


def calculate_transport_density_stats(transports_gdf: gpd.GeoDataFrame,
                                   iris_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculate transport density statistics per IRIS.

    Args:
        transports_gdf: GeoDataFrame with transport stops (categorized)
        iris_gdf: GeoDataFrame with IRIS polygons

    Returns:
        DataFrame: Statistics per IRIS including counts, areas, and densities
    """
    print("Calculating transport density statistics...")

    # Spatial join: assign transports to IRIS
    transports_with_iris = spatial_join_within(transports_gdf, iris_gdf[['code_iris', 'LIB_IRIS', 'geometry']],
                                               keep_right_cols=['code_iris', 'LIB_IRIS'])

    # Count total stops per IRIS
    total_stops = transports_with_iris.groupby('code_iris').size().reset_index(name='total_arrets')

    # Count stops by category per IRIS
    category_pivot = transports_with_iris.pivot_table(
        index='code_iris',
        columns='categorie',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    # Merge with IRIS base data
    stats_df = iris_gdf[['code_iris', 'LIB_IRIS', 'geometry']].merge(
        total_stops, on='code_iris', how='left').fillna({'total_arrets': 0})

    stats_df = stats_df.merge(category_pivot, on='code_iris', how='left').fillna(0)

    # Fill missing categories with 0
    transport_categories = ['Bus', 'Métro', 'Train', 'Autre']
    for cat in transport_categories:
        if cat not in stats_df.columns:
            stats_df[cat] = 0

    # Calculate areas and densities
    stats_gdf = gpd.GeoDataFrame(stats_df, geometry='geometry', crs=iris_gdf.crs)
    stats_gdf = add_area_column(stats_gdf, 'surface_km2', 'km2')

    # Calculate density per km², handle division by zero
    stats_gdf['densite_arrets'] = (stats_gdf['total_arrets'] / stats_gdf['surface_km2']).round(1)

    # Clean up infinite values
    stats_gdf['densite_arrets'] = stats_gdf['densite_arrets'].replace([np.inf, -np.inf], 0)

    # Ensure integer types for counts
    count_columns = ['total_arrets'] + transport_categories
    stats_gdf[count_columns] = stats_gdf[count_columns].astype(int)

    print(f"Calculated transport stats for {len(stats_gdf)} IRIS")
    return stats_gdf


def get_top_dense_iris(stats_gdf: gpd.GeoDataFrame,
                      column: str = 'densite_arrets',
                      top_n: int = 10,
                      ascending: bool = False) -> pd.DataFrame:
    """
    Get top N IRIS by specified column.

    Args:
        stats_gdf: Statistics DataFrame
        column: Column to sort by
        top_n: Number of top items to return
        ascending: Sort order

    Returns:
        DataFrame: Top N rows
    """
    return stats_gdf.nlargest(top_n, column) if not ascending else stats_gdf.nsmallest(top_n, column)


def aggregate_transport_coverage(stats_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregate transport coverage statistics.

    Args:
        stats_gdf: Statistics DataFrame

    Returns:
        DataFrame: Aggregated statistics
    """
    summary = {
        'total_iris': len(stats_gdf),
        'iris_with_transports': (stats_gdf['total_arrets'] > 0).sum(),
        'coverage_percentage': round((stats_gdf['total_arrets'] > 0).mean() * 100, 1),
        'total_stops': stats_gdf['total_arrets'].sum(),
        'avg_density': round(stats_gdf['densite_arrets'].mean(), 1),
        'max_density': round(stats_gdf['densite_arrets'].max(), 1)
    }

    return pd.DataFrame([summary])


def analyze_transport_accessibility(stats_gdf: gpd.GeoDataFrame,
                                  min_threshold: float = 1.0) -> pd.DataFrame:
    """
    Analyze accessibility based on density thresholds.

    Args:
        stats_gdf: Statistics DataFrame
        min_threshold: Minimum density threshold for "good" accessibility

    Returns:
        DataFrame: Accessibility analysis
    """
    accessibility = stats_gdf.copy()
    accessibility['accessibility_score'] = pd.cut(
        accessibility['densite_arrets'],
        bins=[0, min_threshold/2, min_threshold, min_threshold*2, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High']
    )

    return accessibility[['code_iris', 'LIB_IRIS', 'densite_arrets', 'accessibility_score']]


# Main analysis function (pipeline)
def analyze_transport_coverage(iris_gdf: gpd.GeoDataFrame,
                             transport_stops_gdf: gpd.GeoDataFrame) -> dict:
    """
    Complete transport coverage analysis pipeline.

    Args:
        iris_gdf: IRIS polygons
        transport_stops_gdf: Categorized transport stops

    Returns:
        dict: Analysis results including stats, top areas, and summary
    """
    print("Running complete transport coverage analysis...")

    # Calculate stats
    stats = calculate_transport_density_stats(transport_stops_gdf, iris_gdf)

    # Get top dense areas
    top_dense = get_top_dense_iris(stats, 'densite_arrets', 10)

    # Get summary
    summary = aggregate_transport_coverage(stats)

    # Accessibility analysis
    accessibility = analyze_transport_accessibility(stats)

    results = {
        'stats': stats,
        'top_dense': top_dense,
        'summary': summary,
        'accessibility': accessibility
    }

    print("Transport analysis completed")
    return results
