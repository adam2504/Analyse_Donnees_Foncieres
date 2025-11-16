"""
OSM Data Loader Module
=====================

Loads geographic data from OpenStreetMap Overpass API,
specifically for transport stops and commercial establishments.

Author: Adam Jouini
"""

import requests
import geopandas as gpd


def query_overpass_data(query, timeout=120):
    """
    Executes an Overpass API query and returns the JSON response.

    Args:
        query (str): Complete Overpass query string.
        timeout (int): Request timeout in seconds.

    Returns:
        dict: JSON response from Overpass API.

    Raises:
        Exception: If the query fails.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"

    try:
        response = requests.post(overpass_url, data={"data": query}, timeout=timeout)
        response.raise_for_status()
        osm_data = response.json()
        print(f"Retrieved {len(osm_data['elements'])} OSM elements")
        return osm_data
    except Exception as e:
        print(f"Error querying Overpass API: {e}")
        raise


def parse_osm_elements(osm_data, category_logic=None):
    """
    Parses OSM elements into a list of dictionaries with lat/lon and category.

    Args:
        osm_data (dict): JSON response from Overpass API.
        category_logic (callable, optional): Function to determine category from tags.

    Returns:
        list: List of dicts with 'lat', 'lon', and optionally 'category'.
    """
    elements_list = []

    for element in osm_data.get('elements', []):
        tags = element.get('tags', {})

        # Extract coordinates
        if element['type'] == 'node':
            lat, lon = element['lat'], element['lon']
        elif 'center' in element:
            lat, lon = element['center']['lat'], element['center']['lon']
        else:
            continue

        item = {'lat': lat, 'lon': lon, 'tags': tags}

        if category_logic:
            item['category'] = category_logic(tags)

        elements_list.append(item)

    return elements_list


def load_transport_stops(commune="Rennes", admin_level="8", timeout=120):
    """
    Loads public transport stops from OSM for a given commune.

    Args:
        commune (str): Name of the commune to query.
        admin_level (str): Administrative level for the area.
        timeout (int): Query timeout.

    Returns:
        GeoDataFrame: Transport stops with categories.
    """
    print(f"Loading transport stops for {commune}...")

    query = f"""
    [out:json][timeout:90];
    area["name"="{commune}"]["admin_level"="{admin_level}"]->.searchArea;
    (
      node["public_transport"="stop_position"](area.searchArea);
      node["highway"="bus_stop"](area.searchArea);
      node["railway"="station"](area.searchArea);
      node["railway"="halt"](area.searchArea);
      node["railway"="subway_entrance"](area.searchArea);
    );
    out center;
    """

    osm_data = query_overpass_data(query, timeout)

    def categorize_transport(tags):
        if 'bus' in tags.get('highway', '') or tags.get('public_transport') == 'stop_position':
            return 'Bus'
        elif 'subway' in tags.get('railway', '') or 'subway' in tags.get('public_transport', ''):
            return 'Métro'
        elif 'station' in tags.get('railway', '') or 'halt' in tags.get('railway', ''):
            return 'Train'
        else:
            return 'Autre'

    elements = parse_osm_elements(osm_data, category_logic=categorize_transport)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        elements,
        geometry=gpd.points_from_xy([e['lon'] for e in elements], [e['lat'] for e in elements]),
        crs="EPSG:4326"
    )

    print(f"Loaded {len(gdf)} transport stops")
    return gdf


def load_commercial_establishments(commune="Rennes", admin_level="8", timeout=120):
    """
    Loads commercial establishments (restaurants, bars, supermarkets) from OSM.

    Args:
        commune (str): Name of the commune to query.
        admin_level (str): Administrative level for the area.
        timeout (int): Query timeout.

    Returns:
        GeoDataFrame: Commercial establishments with categories.
    """
    print(f"Loading commercial establishments for {commune}...")

    query = f"""
    [out:json][timeout:90];
    area["name"="{commune}"]["admin_level"="{admin_level}"]->.searchArea;
    (
      node["amenity"="restaurant"](area.searchArea);
      node["amenity"="fast_food"](area.searchArea);
      node["amenity"="bar"](area.searchArea);
      node["amenity"="pub"](area.searchArea);
      node["amenity"="cafe"](area.searchArea);
      node["shop"="supermarket"](area.searchArea);
      node["shop"="convenience"](area.searchArea);
      node["shop"="grocery"](area.searchArea);
    );
    out center;
    """

    osm_data = query_overpass_data(query, timeout)

    def categorize_commerce(tags):
        amenity = tags.get('amenity', '')
        shop = tags.get('shop', '')

        if amenity in ['restaurant', 'fast_food']:
            return 'Restaurant'
        elif amenity in ['bar', 'pub', 'cafe']:
            return 'Bar/Café'
        elif shop in ['supermarket', 'convenience', 'grocery']:
            return 'Supermarché'
        else:
            return 'Autre'

    elements = parse_osm_elements(osm_data, category_logic=categorize_commerce)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        elements,
        geometry=gpd.points_from_xy([e['lon'] for e in elements], [e['lat'] for e in elements]),
        crs="EPSG:4326"
    )

    print(f"Loaded {len(gdf)} commercial establishments")
    return gdf


# Backwards compatibility
load_transports_rennes = load_transport_stops
