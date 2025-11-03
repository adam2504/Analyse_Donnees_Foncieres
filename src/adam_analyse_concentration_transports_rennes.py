"""
Analyse des IRIS de Rennes les plus denses en transports en commun
==================================================================
Version modulaire robuste (gestion des catégories manquantes et affichage des "Autre")
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import requests
import warnings
from fiona import listlayers

warnings.filterwarnings('ignore')


def analyse_transports_rennes():
    print("=" * 80)
    print("ANALYSE DES IRIS - TRANSPORTS EN COMMUN A RENNES")
    print("=" * 80)

    # ============================================================================
    # 1. CHARGEMENT DES IRIS
    # ============================================================================
    print("\nETAPE 1 : Chargement des données IRIS")
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

    iris_rennes = iris_start[iris_start['nom_commune'].str.contains("Rennes", case=False, na=False)].copy()
    print(f"OK - {len(iris_rennes)} IRIS conservés pour Rennes")

    url_ref_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"
    iris_noms = pd.read_excel(url_ref_iris).rename(columns={'CODE_IRIS': 'code_iris'})

    iris = iris_rennes.merge(iris_noms[['code_iris', 'LIB_IRIS', 'LIBCOM']], 
                             on='code_iris', how='left')
    print(f"OK - Fusion terminée : {len(iris)} IRIS avec noms et géométries")

    # ============================================================================
    # 2. RÉCUPÉRATION DES TRANSPORTS EN COMMUN OSM
    # ============================================================================
    print("\n" + "=" * 80)
    print("ETAPE 2 : Récupération des arrêts de transports depuis OpenStreetMap")
    print("-" * 80)

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json][timeout:90];
    area["name"="Rennes"]["admin_level"="8"]->.searchArea;
    (
      node["public_transport"="stop_position"](area.searchArea);
      node["highway"="bus_stop"](area.searchArea);
      node["railway"="station"](area.searchArea);
      node["railway"="halt"](area.searchArea);
      node["railway"="subway_entrance"](area.searchArea);
    );
    out center;
    """

    print("Envoi de la requête à l’API Overpass (peut durer 30-60s)...")
    try:
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=120)
        response.raise_for_status()
        osm_data = response.json()
        print(f"OK - {len(osm_data['elements'])} arrêts récupérés")
    except Exception as e:
        print(f"ERREUR - Impossible de récupérer les données OSM : {e}")
        return None, None, None

    # ============================================================================
    # 3. CATÉGORISATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("ETAPE 3 : Catégorisation des arrêts")
    print("-" * 80)

    transports_list = []
    for element in osm_data['elements']:
        tags = element.get('tags', {})
        if element['type'] == 'node':
            lat, lon = element['lat'], element['lon']
        elif 'center' in element:
            lat, lon = element['center']['lat'], element['center']['lon']
        else:
            continue

        if 'bus' in tags.get('highway', '') or tags.get('public_transport') == 'stop_position':
            categorie = 'Bus'
        elif 'subway' in tags.get('railway', '') or 'subway' in tags.get('public_transport', ''):
            categorie = 'Métro'
        elif 'station' in tags.get('railway', '') or 'halt' in tags.get('railway', ''):
            categorie = 'Train'
        else:
            categorie = 'Autre'

        transports_list.append({'categorie': categorie, 'latitude': lat, 'longitude': lon, 'tags': tags})

    transports_df = pd.DataFrame(transports_list)
    print("\nRépartition par catégorie :")
    print(transports_df['categorie'].value_counts().to_string())

    # ============================================================================
    # 4. JOINTURE SPATIALE
    # ============================================================================
    print("\n" + "=" * 80)
    print("ETAPE 4 : Jointure spatiale (transports x IRIS)")
    print("-" * 80)

    transports_gdf = gpd.GeoDataFrame(
        transports_df,
        geometry=gpd.points_from_xy(transports_df.longitude, transports_df.latitude),
        crs="EPSG:4326"
    ).to_crs(iris.crs)

    transports_iris = gpd.sjoin(transports_gdf, iris[['code_iris', 'LIB_IRIS', 'geometry']], how='left', predicate='within')
    nb_associes = transports_iris['code_iris'].notna().sum()
    print(f"OK - {nb_associes} arrêts associés à un IRIS")

    # ============================================================================
    # 5. STATISTIQUES PAR IRIS
    # ============================================================================
    print("\n" + "=" * 80)
    print("ETAPE 5 : Agrégation et calculs")
    print("-" * 80)

    stats_iris = transports_iris.groupby('code_iris').size().reset_index(name='total_arrets')
    pivot = transports_iris.pivot_table(index='code_iris', columns='categorie', aggfunc='size', fill_value=0).reset_index()

    stats_iris = iris[['code_iris', 'LIB_IRIS', 'geometry']].merge(stats_iris, on='code_iris', how='left')
    stats_iris = stats_iris.merge(pivot, on='code_iris', how='left')

    for c in ['Bus', 'Métro', 'Train']:
        if c not in stats_iris.columns:
            stats_iris[c] = 0

    stats_iris = stats_iris.to_crs(epsg=2154)
    stats_iris['surface_km2'] = stats_iris.geometry.area / 1_000_000
    stats_iris['densite_arrets'] = (stats_iris['total_arrets'] / stats_iris['surface_km2']).replace([float('inf')], 0).round(1)
    stats_iris['total_arrets'] = stats_iris['total_arrets'].fillna(0).astype(int)

    top10 = stats_iris.nlargest(10, 'densite_arrets')
    print("\nTop 10 IRIS les plus denses en transports en commun :")
    print(top10[['LIB_IRIS', 'total_arrets', 'densite_arrets']])

    # ============================================================================
    # 6. GRAPHIQUES
    # ============================================================================
    print("\n" + "=" * 80)
    print("ETAPE 6 : Génération des graphiques")
    print("-" * 80)

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # GRAPHIQUE 1
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    top10_sorted = top10.sort_values('densite_arrets', ascending=True)
    bars = ax1.barh(range(len(top10_sorted)), top10_sorted['densite_arrets'],
                    color=sns.color_palette("RdYlGn_r", len(top10_sorted)))
    ax1.set_yticks(range(len(top10_sorted)))
    ax1.set_yticklabels(top10_sorted['LIB_IRIS'])
    ax1.set_xlabel('Densité d’arrêts (par km²)')
    ax1.set_title('Top 10 des IRIS de Rennes - Densité transports en commun')
    plt.tight_layout()

    # GRAPHIQUE 2
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    x = range(len(top10))
    width = 0.25
    ax2.bar([i - width for i in x], top10['Bus'], width, label='Bus', color='#FF6B6B')
    ax2.bar(x, top10['Métro'], width, label='Métro', color='#4ECDC4')
    ax2.bar([i + width for i in x], top10['Train'], width, label='Train', color='#45B7D1')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top10['LIB_IRIS'], rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylabel('Nombre d’arrêts')
    ax2.set_title('Répartition par type de transport - Top 10 IRIS')
    plt.tight_layout()

    print("OK - Graphiques générés")

    return fig1, fig2, stats_iris


# Permet exécution directe
if __name__ == "__main__":
    analyse_transports_rennes()
