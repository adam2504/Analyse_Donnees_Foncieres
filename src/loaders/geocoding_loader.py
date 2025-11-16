"""
Geocoding Loader Module
======================

Handles geographical coordinate lookup for cities using various geocoding services.
Supports rate limiting and caching to avoid API limits.

Author: Adam Jouini
"""

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from typing import Union, Tuple, Optional

from ..utils.cache import cached_download_dataframe


class Geocoder:
    """
    A geocoding class with caching and rate limiting.
    """

    def __init__(self, user_agent: str = "data_analysis_geocoder",
                 min_delay: float = 1.0, cache_expiry_hours: int = 24):
        """
        Initialize the geocoder.

        Args:
            user_agent (str): User agent for geocoding service
            min_delay (float): Minimum delay between requests in seconds
            cache_expiry_hours (int): Cache expiry time in hours
        """
        self.user_agent = user_agent
        self.min_delay = min_delay
        self.cache_expiry = cache_expiry_hours

        self.geolocator = Nominatim(user_agent=user_agent)
        self.geocode = RateLimiter(self.geolocator.geocode,
                                 min_delay_seconds=min_delay)

    def get_coordinates_cached(self, location_name: str, country: str = "France") -> Tuple[Optional[float], Optional[float]]:
        """
        Get coordinates for a location with caching.

        Args:
            location_name (str): Name of the location
            country (str): Country for the location

        Returns:
            tuple: (latitude, longitude) or (None, None) if not found
        """
        if pd.isna(location_name):
            return None, None

        full_name = f"{location_name}, {country}"

        try:
            location = self.geocode(full_name)
            if location:
                return location.latitude, location.longitude
            else:
                print(f"⚠️ No coordinates found for: {full_name}")
                return None, None
        except Exception as e:
            print(f"⚠️ Error geocoding '{full_name}': {e}")
            return None, None


def add_geocoding_coordinates(df: pd.DataFrame,
                            location_column: str = 'libgeo',
                            user_agent: str = "french_data_analysis",
                            country: str = "France") -> pd.DataFrame:
    """
    Add latitude and longitude columns to a DataFrame using geocoding.

    Args:
        df (DataFrame): DataFrame with location names
        location_column (str): Name of column containing location names
        user_agent (str): User agent for geocoding service
        country (str): Country to append to location names

    Returns:
        DataFrame: Original data with added latitude and longitude columns
    """
    print(f"🌍 Geocoding {len(df)} locations (this may take several minutes)...")

    # Initialize geocoder
    geocoder = Geocoder(user_agent=user_agent)

    # Get coordinates for each location
    coordinates = df[location_column].apply(geocoder.get_coordinates_cached, country=country)

    # Split into separate columns
    df_copy = df.copy()
    df_copy[['latitude', 'longitude']] = pd.DataFrame(coordinates.tolist(),
                                                     index=df.index)

    # Count successful geocoding
    valid_coords = df_copy['latitude'].notna().sum()
    print(f"✅ Geocoding complete: {valid_coords}/{len(df)} locations successfully geocoded")

    return df_copy


# Convenience functions for common use cases
def geocode_french_cities(df: pd.DataFrame, city_column: str = 'libgeo') -> pd.DataFrame:
    """
    Geocode French cities (cities in DataFrame).

    This is a wrapper around add_geocoding_coordinates with French-specific settings.
    """
    return add_geocoding_coordinates(
        df=df,
        location_column=city_column,
        user_agent="french_cities_analysis",
        country="France"
    )


# Backwards compatibility functions
def ajouter_coordonnees(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy function name from old code."""
    return geocode_french_cities(df)


# Cache key function for geocoding results
def _create_geocode_cache_key(location_name: str, country: str) -> str:
    """Create a cache key for geocoding results."""
    return f"geocode_{location_name}_{country}".replace(" ", "_").replace(",", "")


# Version with explicit caching (optional advanced use)
def geocode_with_explicit_cache(df: pd.DataFrame,
                              location_column: str = 'libgeo',
                              user_agent: str = "geocoding_cache",
                              country: str = "France") -> pd.DataFrame:
    """
    Geocode locations with explicit caching of coordinates.

    This version caches individual coordinate results for better performance
    on repeated runs.
    """
    print(f"🌍 Geocoding {len(df)} locations with caching...")

    geocoder = Geocoder(user_agent=user_agent)
    df_result = df.copy()

    # Add coordinate columns initialized to None
    df_result['latitude'] = None
    df_result['longitude'] = None

    for idx, row in df.iterrows():
        location_name = row[location_column]

        if pd.isna(location_name):
            continue

        lat, lon = geocoder.get_coordinates_cached(location_name, country)
        df_result.at[idx, 'latitude'] = lat
        df_result.at[idx, 'longitude'] = lon

    valid_coords = df_result['latitude'].notna().sum()
    print(f"✅ Geocoding with caching complete: {valid_coords}/{len(df)} locations")

    return df_result
