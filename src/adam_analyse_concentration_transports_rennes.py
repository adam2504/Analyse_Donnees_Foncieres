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
import warnings
from loaders.iris_loader import load_iris_rennes
from loaders.osm_loader import load_transport_stops

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

    iris = load_iris_rennes()

    # ============================================================================
    # 2. RÉCUPÉRATION DES TRANSPORTS EN COMMUN OSM
    # ============================================================================
    print("\n" + "=" * 80)
    print("ETAPE 2 : Récupération des arrêts de transports depuis OpenStreetMap")
    print("-" * 80)

    transports_gdf = load_transport_stops()
    transports_df = transports_gdf[['categorie', 'lat', 'lon', 'tags']].rename(columns={'lat': 'latitude', 'lon': 'longitude'})

    print("\nRépartition par catégorie :")
    print(transports_df['categorie'].value_counts().to_string())

    # ============================================================================
    # 3. JOINTURE SPATIALE
    # ============================================================================
    print("\n" + "=" * 80)
    print("ETAPE 3 : Jointure spatiale (transports x IRIS)")
    print("-" * 80)

    transports_gdf = transports_gdf.to_crs(iris.crs)

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
