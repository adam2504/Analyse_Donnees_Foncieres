"""
Analyse des IRIS de Rennes les plus denses en commerces vivants
================================================================
Version modulaire : affichage direct des graphiques sans sauvegarde
"""

import os
import pandas as pd
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from fiona import listlayers

warnings.filterwarnings('ignore')


def analyse_commerces_rennes():
    print("=" * 80)
    print("ANALYSE DES IRIS - COMMERCES VIVANTS A RENNES")
    print("=" * 80)

    # ============================================================================
    # 1. CHARGEMENT DES IRIS
    # ============================================================================
    print("\nETAPE 1 : Chargement des donnees IRIS")
    print("-" * 80)

    url_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/contours-iris-pe.gpkg"
    local_path = "contours-iris-pe.gpkg"

    if not os.path.exists(local_path):
        print("Téléchargement du fichier...")
        r = requests.get(url_iris)
        with open(local_path, "wb") as f:
            f.write(r.content)

    layers = listlayers(local_path)
    iris_start = gpd.read_file(local_path, layer=layers[0])
    print(f"OK - {len(iris_start)} IRIS géographiques téléchargés")

    url_ref_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"
    iris_noms = pd.read_excel(url_ref_iris).rename(columns={'CODE_IRIS': 'code_iris'})
    print(f"OK - {len(iris_noms)} IRIS chargés dans la base nationale")

    iris_rennes = iris_start[iris_start['nom_commune'].str.contains("Rennes", case=False, na=False)].copy()
    print(f"OK - {len(iris_rennes)} IRIS conservés pour Rennes")

    iris = iris_rennes.merge(iris_noms[['code_iris', 'LIB_IRIS', 'LIBCOM']], 
                             on='code_iris', how='left')
    print(f"OK - Fusion terminée : {len(iris)} IRIS avec noms et géométries")

    # ============================================================================
    # 2. RÉCUPÉRATION DES COMMERCES OSM
    # ============================================================================
    print("\nETAPE 2 : Recuperation des commerces depuis OpenStreetMap")
    print("-" * 80)

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:90];
    area["name"="Rennes"]["admin_level"="8"]->.searchArea;
    (
      node["amenity"="restaurant"](area.searchArea);
      way["amenity"="restaurant"](area.searchArea);
      node["amenity"="fast_food"](area.searchArea);
      way["amenity"="fast_food"](area.searchArea);
      node["amenity"="bar"](area.searchArea);
      way["amenity"="bar"](area.searchArea);
      node["amenity"="pub"](area.searchArea);
      way["amenity"="pub"](area.searchArea);
      node["amenity"="cafe"](area.searchArea);
      way["amenity"="cafe"](area.searchArea);
      node["shop"="supermarket"](area.searchArea);
      way["shop"="supermarket"](area.searchArea);
      node["shop"="convenience"](area.searchArea);
      way["shop"="convenience"](area.searchArea);
      node["shop"="grocery"](area.searchArea);
      way["shop"="grocery"](area.searchArea);
    );
    out center;
    """

    try:
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=120)
        response.raise_for_status()
        osm_data = response.json()
        print(f"OK - {len(osm_data['elements'])} commerces récupérés")
    except Exception as e:
        print(f"ERREUR - Impossible de récupérer les données OSM : {e}")
        return None, None, None

    # ============================================================================
    # 3. CONVERSION ET CATÉGORISATION
    # ============================================================================
    print("\nETAPE 3 : Conversion et catégorisation")
    commerces_list = []

    for element in osm_data['elements']:
        tags = element.get('tags', {})
        if element['type'] == 'node':
            lat, lon = element['lat'], element['lon']
        elif 'center' in element:
            lat, lon = element['center']['lat'], element['center']['lon']
        else:
            continue

        amenity = tags.get('amenity', '')
        shop = tags.get('shop', '')

        if amenity in ['restaurant', 'fast_food']:
            categorie = 'Restaurant'
        elif amenity in ['bar', 'pub', 'cafe']:
            categorie = 'Bar/Café'
        elif shop in ['supermarket', 'convenience', 'grocery']:
            categorie = 'Supermarché'
        else:
            categorie = 'Autre'

        commerces_list.append({'categorie': categorie, 'latitude': lat, 'longitude': lon})

    commerces_df = pd.DataFrame(commerces_list)
    commerces_gdf = gpd.GeoDataFrame(
        commerces_df,
        geometry=gpd.points_from_xy(commerces_df.longitude, commerces_df.latitude),
        crs="EPSG:4326"
    ).to_crs(iris.crs)
    print(f"OK - {len(commerces_gdf)} commerces géolocalisés")
    print("Répartition par catégorie :\n", commerces_gdf['categorie'].value_counts())

    # ============================================================================
    # 4. JOINTURE SPATIALE
    # ============================================================================
    print("\nETAPE 4 : Jointure spatiale (commerces x IRIS)")
    commerces_iris = gpd.sjoin(commerces_gdf, iris[['code_iris', 'LIB_IRIS', 'geometry']], 
                                how='left', predicate='within')
    nb_associes = commerces_iris['code_iris'].notna().sum()
    print(f"OK - {nb_associes} commerces associés à un IRIS")

    stats_iris = commerces_iris.groupby('code_iris').size().reset_index(name='total_commerces')
    pivot = commerces_iris.pivot_table(index='code_iris', columns='categorie', aggfunc='size', fill_value=0).reset_index()
    stats_iris = iris[['code_iris', 'LIB_IRIS', 'geometry']].merge(stats_iris, on='code_iris', how='left')
    stats_iris = stats_iris.merge(pivot, on='code_iris', how='left')

    for col in ['total_commerces', 'Restaurant', 'Bar/Café', 'Supermarché']:
        if col in stats_iris.columns:
            stats_iris[col] = stats_iris[col].fillna(0).astype(int)
        else:
            stats_iris[col] = 0

    stats_iris = stats_iris.to_crs(epsg=2154)
    stats_iris['surface_km2'] = stats_iris.geometry.area / 1_000_000
    stats_iris['score_densite'] = (
        stats_iris['Restaurant']*2 + stats_iris['Bar/Café']*2 + stats_iris['Supermarché']*1.5
    ) / stats_iris['surface_km2']
    stats_iris['score_densite'] = stats_iris['score_densite'].replace([float('inf')], 0).round(1)

    iris_valides = stats_iris[(stats_iris['total_commerces'] > 0) & (stats_iris['surface_km2'] > 0)]
    top10 = iris_valides.nlargest(10, 'score_densite')

    # ============================================================================
    # 5. GRAPHIQUES
    # ============================================================================
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Graphique 1
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    top10_sorted = top10.sort_values('score_densite', ascending=True)
    ax1.barh(range(len(top10_sorted)), top10_sorted['total_commerces'], color=sns.color_palette("RdYlGn_r", len(top10_sorted)))
    ax1.set_yticks(range(len(top10_sorted)))
    ax1.set_yticklabels(top10_sorted['LIB_IRIS'])
    ax1.set_xlabel('Nombre de commerces')
    ax1.set_title('Top 10 IRIS Rennes - commerces vivants')
    plt.tight_layout()

    # Graphique 2
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    x = range(len(top10))
    width = 0.25
    ax2.bar([i - width for i in x], top10['Restaurant'], width, label='Restaurants', color='#FF6B6B')
    ax2.bar(x, top10['Bar/Café'], width, label='Bars/Cafés', color='#4ECDC4')
    ax2.bar([i + width for i in x], top10['Supermarché'], width, label='Supermarchés', color='#45B7D1')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top10['LIB_IRIS'], rotation=45, ha='right')
    ax2.set_ylabel('Nombre de commerces')
    ax2.set_title('Composition du Top 10 IRIS par type de commerce')
    ax2.legend()
    plt.tight_layout()

    print("OK - Graphiques générés")

    return fig1, fig2, stats_iris


if __name__ == "__main__":
    analyse_commerces_rennes()
