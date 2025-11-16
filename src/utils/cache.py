"""
Caching Utilities Module
=======================

Provides functions for caching dataframes and GeoDataFrames to disk,
with support for expiry times and format selection.

Author: Adam Jouini
"""

import os
import hashlib
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union, Optional

from config import CACHE_DIR, CACHE_FORMAT, CACHE_EXPIRY_HOURS


def ensure_cache_dir():
    """Ensure the cache directory exists."""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def get_cache_path(key: str, format: str = CACHE_FORMAT) -> str:
    """Get the full path for a cached file."""
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{key}.{format}")


def get_cache_metadata_path(key: str) -> str:
    """Get the path for cache metadata (timestamp)."""
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{key}.meta")


def generate_cache_key(content: str) -> str:
    """Generate a cache key from string content using hash."""
    return hashlib.md5(content.encode()).hexdigest()[:16]


def save_to_cache(data: Union[pd.DataFrame, gpd.GeoDataFrame],
                  key: str,
                  format: str = CACHE_FORMAT) -> None:
    """
    Save dataframe to cache with timestamp.

    Args:
        data: DataFrame or GeoDataFrame to cache
        key: Cache key identifier
        format: Save format ('parquet', 'pickle', etc.)
    """
    cache_path = get_cache_path(key, format)
    meta_path = get_cache_metadata_path(key)

    # Save data
    if format == 'parquet':
        if hasattr(data, 'crs'):  # GeoDataFrame
            data.to_parquet(cache_path, index=True)
        else:
            data.to_parquet(cache_path, index=False)
    elif format == 'pickle':
        data.to_pickle(cache_path)

    # Save metadata (timestamp)
    with open(meta_path, 'w') as f:
        f.write(datetime.now().isoformat())


def load_from_cache(key: str,
                   format: str = CACHE_FORMAT,
                   max_age_hours: int = CACHE_EXPIRY_HOURS) -> Optional[Union[pd.DataFrame, gpd.GeoDataFrame]]:
    """
    Load dataframe from cache if it exists and is not expired.

    Args:
        key: Cache key identifier
        format: Cache format
        max_age_hours: Maximum cache age in hours

    Returns:
        DataFrame/GeoDataFrame if cache is valid, None otherwise
    """
    cache_path = get_cache_path(key, format)
    meta_path = get_cache_metadata_path(key)

    if not os.path.exists(cache_path) or not os.path.exists(meta_path):
        return None

    # Check timestamp
    try:
        with open(meta_path, 'r') as f:
            timestamp_str = f.read().strip()
        timestamp = datetime.fromisoformat(timestamp_str)
        if datetime.now() - timestamp > timedelta(hours=max_age_hours):
            return None  # Expired
    except:
        return None  # Invalid metadata

    # Load data
    try:
        if format == 'parquet':
            if cache_path.endswith('.parquet'):
                # Try GeoDataFrame first, fallback to DataFrame
                try:
                    gdf = gpd.read_parquet(cache_path)
                    return gdf
                except:
                    return pd.read_parquet(cache_path)
        elif format == 'pickle':
            return pd.read_pickle(cache_path)
    except:
        return None

    return None


def cached_download_dataframe(url: str,
                             loader_func,
                             cache_key: Optional[str] = None,
                             **kwargs) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Download dataframe with caching.

    Args:
        url: URL to download from
        loader_func: Function to load data (should return DataFrame)
        cache_key: Optional custom cache key

    Returns:
        DataFrame/GeoDataFrame from cache or fresh download
    """
    key = cache_key or generate_cache_key(url)

    # Try cache first
    cached_data = load_from_cache(key)
    if cached_data is not None:
        print(f"Loaded from cache: {key}")
        return cached_data

    # Download fresh
    print(f"Downloading fresh data from {url}")
    data = loader_func(**kwargs)

    # Cache it
    save_to_cache(data, key)

    return data


# Convenience functions for common use cases
def cache_iris_geometries(gdf: gpd.GeoDataFrame):
    """Cache IRIS geometries."""
    save_to_cache(gdf, 'iris_geometries')


def cache_iris_names(df: pd.DataFrame):
    """Cache IRIS names."""
    save_to_cache(df, 'iris_names')


def cache_education_data(df: pd.DataFrame):
    """Cache education establishments data."""
    save_to_cache(df, 'education_data')


def load_cached_iris_geometries() -> Optional[gpd.GeoDataFrame]:
    """Load cached IRIS geometries."""
    return load_from_cache('iris_geometries')


def load_cached_iris_names() -> Optional[pd.DataFrame]:
    """Load cached IRIS names."""
    return load_from_cache('iris_names')


def load_cached_education_data() -> Optional[pd.DataFrame]:
    """Load cached education data."""
    return load_from_cache('education_data')
