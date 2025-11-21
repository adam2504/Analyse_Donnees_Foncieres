"""
Commerce Analysis Module
=======================

Business logic for analyzing commercial establishments density and coverage in IRIS areas.
Calculates statistics, densities, and aggregations for restaurants, bars, and supermarkets.

Author: Valentine (modularized by Adam Jouini)
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


def calculate_commerce_density_stats(commerces_gdf: gpd.GeoDataFrame,
                                   iris_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculate commerce density statistics per IRIS.

    Args:
        commerces_gdf: GeoDataFrame with commercial establishments (categorized)
        iris_gdf: GeoDataFrame with IRIS polygons

    Returns:
        DataFrame: Statistics per IRIS including counts, areas, and densities
    """
    print("Calculating commerce density statistics...")

    # Spatial join: assign commerces to IRIS
    commerces_with_iris = spatial_join_within(commerces_gdf, iris_gdf[['code_iris', 'LIB_IRIS', 'geometry']],
                                              keep_right_cols=['code_iris', 'LIB_IRIS'])

    # Count total commerces per IRIS
    total_commerces = commerces_with_iris.groupby('code_iris').size().reset_index(name='total_commerces')

    # Count commerces by category per IRIS
    category_pivot = commerces_with_iris.pivot_table(
        index='code_iris',
        columns='category',  # Note: using 'category' from osm_loader
        aggfunc='size',
        fill_value=0
    ).reset_index()

    # Merge with IRIS base data
    stats_df = iris_gdf[['code_iris', 'LIB_IRIS', 'geometry']].merge(
        total_commerces, on='code_iris', how='left').fillna({'total_commerces': 0})

    stats_df = stats_df.merge(category_pivot, on='code_iris', how='left').fillna(0)

    # Fill missing categories with 0
    commerce_categories = ['Restaurant', 'Bar/Café', 'Supermarché']
    for cat in commerce_categories:
        if cat not in stats_df.columns:
            stats_df[cat] = 0

    # Calculate areas and densities
    stats_gdf = gpd.GeoDataFrame(stats_df, geometry='geometry', crs=iris_gdf.crs)
    stats_gdf = reproject_to_french_mainland(stats_gdf)
    stats_gdf = add_area_column(stats_gdf, 'surface_km2', 'km2')

    # Calculate commerce density score (weighted by type)
    # Restaurants and bars/cafes: 2 points, Supermarkets: 1.5 points
    stats_gdf['score_densite'] = (
        stats_gdf['Restaurant'] * 2 +
        stats_gdf['Bar/Café'] * 2 +
        stats_gdf['Supermarché'] * 1.5
    ) / stats_gdf['surface_km2']

    # Clean up infinite values
    stats_gdf['score_densite'] = stats_gdf['score_densite'].replace([np.inf, -np.inf], 0).round(1)

    # Ensure integer types for counts
    count_columns = ['total_commerces'] + commerce_categories
    stats_gdf[count_columns] = stats_gdf[count_columns].astype(int)

    print(f"Calculated commerce stats for {len(stats_gdf)} IRIS")
    return stats_gdf


def get_top_dense_iris_commerce(stats_gdf: gpd.GeoDataFrame,
                               column: str = 'score_densite',
                               top_n: int = 10,
                               ascending: bool = False) -> pd.DataFrame:
    """
    Get top N IRIS by specified commerce density column.

    Args:
        stats_gdf: Statistics DataFrame
        column: Column to sort by ('score_densite', 'total_commerces', etc.)
        top_n: Number of top items to return
        ascending: Sort order

    Returns:
        DataFrame: Top N rows
    """
    # Filter out IRIS with no area or no commerces for density calculations
    valid_stats = stats_gdf[(stats_gdf['total_commerces'] > 0) & (stats_gdf['surface_km2'] > 0)].copy()

    if column == 'score_densite':
        return valid_stats.nlargest(top_n, column) if not ascending else valid_stats.nsmallest(top_n, column)
    else:
        return stats_gdf.nlargest(top_n, column) if not ascending else stats_gdf.nsmallest(top_n, column)


def aggregate_commerce_coverage(stats_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregate commerce coverage statistics.

    Args:
        stats_gdf: Statistics DataFrame

    Returns:
        DataFrame: Aggregated statistics
    """
    # Valid stats (excluding IRIS with no area)
    valid_stats = stats_gdf[stats_gdf['surface_km2'] > 0]

    summary = {
        'total_iris': len(stats_gdf),
        'valid_iris': len(valid_stats),
        'iris_with_commerces': (stats_gdf['total_commerces'] > 0).sum(),
        'coverage_percentage': round((stats_gdf['total_commerces'] > 0).mean() * 100, 1),
        'total_establishments': stats_gdf['total_commerces'].sum(),
        'avg_density_score': round(valid_stats['score_densite'].mean(), 1),
        'max_density_score': round(valid_stats['score_densite'].max(), 1),
        'restaurants_total': stats_gdf['Restaurant'].sum(),
        'bars_cafes_total': stats_gdf['Bar/Café'].sum(),
        'supermarkets_total': stats_gdf['Supermarché'].sum()
    }

    return pd.DataFrame([summary])


def analyze_commerce_accessibility(stats_gdf: gpd.GeoDataFrame,
                                 score_threshold: float = 2.0) -> pd.DataFrame:
    """
    Analyze commerce accessibility based on density score thresholds.

    Args:
        stats_gdf: Statistics DataFrame
        score_threshold: Minimum density score threshold for "good" accessibility

    Returns:
        DataFrame: Accessibility analysis
    """
    accessibility = stats_gdf.copy()
    accessibility['accessibility_score'] = pd.cut(
        accessibility['score_densite'],
        bins=[0, score_threshold/4, score_threshold/2, score_threshold, score_threshold*2, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

    return accessibility[['code_iris', 'LIB_IRIS', 'score_densite', 'accessibility_score']]


# Main analysis function (pipeline)
def analyze_commerce_density(iris_gdf: gpd.GeoDataFrame,
                           commerces_gdf: gpd.GeoDataFrame,
                           score_weights: dict = None) -> dict:
    """
    Complete commerce density analysis pipeline.

    Args:
        iris_gdf: IRIS polygons
        commerces_gdf: Categorized commercial establishments
        score_weights: Custom weights for density score calculation (optional)

    Returns:
        dict: Analysis results including stats, top areas, and summary
    """
    print("Running complete commerce density analysis...")

    # Ensure commerces have the right column name
    if 'categorie' in commerces_gdf.columns and 'category' not in commerces_gdf.columns:
        commerces_gdf = commerces_gdf.rename(columns={'categorie': 'category'})

    # Calculate stats
    stats = calculate_commerce_density_stats(commerces_gdf, iris_gdf)

    # Get top dense areas by score
    top_dense_score = get_top_dense_iris_commerce(stats, 'score_densite', 10)
    top_dense_count = get_top_dense_iris_commerce(stats, 'total_commerces', 10)

    # Get summary
    summary = aggregate_commerce_coverage(stats)

    # Accessibility analysis
    accessibility = analyze_commerce_accessibility(stats)

    results = {
        'stats': stats,
        'top_by_score': top_dense_score,
        'top_by_count': top_dense_count,
        'summary': summary,
        'accessibility': accessibility
    }

    print("Commerce analysis completed")
    return results
