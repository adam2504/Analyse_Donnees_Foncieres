"""
Spatial Utilities Module
=======================

Helper functions for geospatial operations including coordinate transformations,
density calculations, and spatial joins.

Author: Adam Jouini
"""

import geopandas as gpd
from typing import Optional
import numpy as np

from ..config import CRS_WGS84, CRS_LAMBERT93, OSM_ADMIN_LEVEL


def reproject_to_french_mainland(gdf: gpd.GeoDataFrame,
                               target_crs: str = CRS_LAMBERT93) -> gpd.GeoDataFrame:
    """
    Reproject GeoDataFrame to Lambert 93 (mainland France) or specified CRS.

    Args:
        gdf: Input GeoDataFrame
        target_crs: Target coordinate reference system

    Returns:
        Reprojected GeoDataFrame
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined")
    return gdf.to_crs(target_crs)


def calculate_density_per_km2(gdf: gpd.GeoDataFrame,
                             count_column: str,
                             area_column: Optional[str] = None,
                             area_unit: str = 'km2') -> np.ndarray:
    """
    Calculate density per km² for a given count column.

    Args:
        gdf: GeoDataFrame with polygons
        count_column: Column name containing counts
        area_column: If provided, use this area column instead of calculating
        area_unit: Unit for area calculation ('km2' or 'm2')

    Returns:
        Array of density values
    """
    if area_column:
        area = gdf[area_column]
    else:
        # Assume Lambert 93 for area calculation
        if gdf.crs != CRS_LAMBERT93:
            temp_gdf = reproject_to_french_mainland(gdf)
        else:
            temp_gdf = gdf

        area = temp_gdf.geometry.area
        if area_unit == 'km2':
            area = area / 1_000_000

    density = gdf[count_column] / area
    return density.replace([np.inf, -np.inf], 0).fillna(0)


def add_area_column(gdf: gpd.GeoDataFrame,
                   column_name: str = 'area_km2',
                   unit: str = 'km2') -> gpd.GeoDataFrame:
    """
    Add an area column to GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame
        column_name: Name for the new area column
        unit: Unit for area ('km2' or 'm2')

    Returns:
        GeoDataFrame with area column added
    """
    temp_gdf = reproject_to_french_mainland(gdf) if gdf.crs != CRS_LAMBERT93 else gdf
    area = temp_gdf.geometry.area

    if unit == 'km2':
        area = area / 1_000_000

    gdf = gdf.copy()
    gdf[column_name] = area
    return gdf


def spatial_join_within(left_gdf: gpd.GeoDataFrame,
                       right_gdf: gpd.GeoDataFrame,
                       keep_right_cols: Optional[list] = None) -> gpd.GeoDataFrame:
    """
    Perform spatial join with 'within' predicate, keeping only specified right columns.

    Args:
        left_gdf: Left GeoDataFrame (points)
        right_gdf: Right GeoDataFrame (polygons)
        keep_right_cols: Columns to keep from right GeoDataFrame

    Returns:
        Joined GeoDataFrame
    """
    # Ensure same CRS
    if left_gdf.crs != right_gdf.crs:
        left_gdf = left_gdf.to_crs(right_gdf.crs)

    columns_to_keep = ['geometry'] + (keep_right_cols or right_gdf.columns.tolist())
    right_subset = right_gdf[columns_to_keep]

    joined = gpd.sjoin(left_gdf, right_subset, how='left', predicate='within')
    return joined


def get_commune_bbox(commune_name: str, country: str = "France") -> tuple:
    """
    Get approximate bounding box for a commune (for OSM queries).

    This is a placeholder - in production, use geocoding service or pre-computed bounds.
    """
    # Placeholder implementation - would need actual geocoding
    # For Rennes specifically
    if commune_name.lower() == "rennes":
        return (-1.75, 48.08, -1.60, 48.15)  # min_lon, min_lat, max_lon, max_lat

    raise NotImplementedError("Dynamic bbox calculation not implemented. Use known bounds.")


def validate_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate and fix geometry if needed.

    Args:
        gdf: Input GeoDataFrame

    Returns:
        GeoDataFrame with valid geometries
    """
    gdf = gdf.copy()
    if not gdf.geometry.is_valid.all():
        gdf.geometry = gdf.geometry.make_valid()
    return gdf


def buffer_geometry(gdf: gpd.GeoDataFrame,
                   distance_meters: float,
                   target_crs: str = CRS_LAMBERT93) -> gpd.GeoDataFrame:
    """
    Create buffer around geometries.

    Args:
        gdf: Input GeoDataFrame
        distance_meters: Buffer distance in meters
        target_crs: CRS to use for buffering (should be metric)

    Returns:
        GeoDataFrame with buffered geometries
    """
    temp_gdf = gdf.to_crs(target_crs) if gdf.crs != target_crs else gdf
    temp_gdf = temp_gdf.copy()
    temp_gdf.geometry = temp_gdf.geometry.buffer(distance_meters)
    return temp_gdf.to_crs(gdf.crs) if gdf.crs != target_crs else temp_gdf
