"""
Analyse des IRIS de Rennes les plus denses en transports en commun
==================================================================
Version robuste (gestion des catégories manquantes et affichage des "Autre")
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
import requests
import warnings
warnings.filterwarnings('ignore')

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

from fiona import listlayers
layers = listlayers(local_path)
iris_start = gpd.read_file(local_path, layer=layers[0])
print(f"OK - {len(iris_start)} IRIS géographiques téléchargés")

# Filtrage sur Rennes
iris_rennes = iris_start[iris_start['nom_commune'].str.contains("Rennes", case=False, na=False)].copy()
print(f"OK - {len(iris_rennes)} IRIS conservés pour Rennes")

# Ajout des noms IRIS
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
    exit(1)

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

    # Catégorisation basique
    if 'bus' in tags.get('highway', '') or tags.get('public_transport') == 'stop_position':
        categorie = 'Bus'
    elif 'subway' in tags.get('railway', '') or 'subway' in tags.get('public_transport', ''):
        categorie = 'Métro'
    elif 'station' in tags.get('railway', '') or 'halt' in tags.get('railway', ''):
        categorie = 'Train'
    else:
        categorie = 'Autre'

    transports_list.append({
        'categorie': categorie,
        'latitude': lat,
        'longitude': lon,
        'tags': tags
    })

transports_df = pd.DataFrame(transports_list)
print("\nRépartition par catégorie :")
print(transports_df['categorie'].value_counts().to_string())

# >>> Ajout : affichage de quelques exemples de la catégorie "Autre"
print("\nExemples d'entrées classées en 'Autre' :")
print(transports_df[transports_df['categorie'] == 'Autre'].head(10)['tags'].to_list())


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

transports_iris = gpd.sjoin(transports_gdf, iris[['code_iris', 'LIB_IRIS', 'geometry']], 
                            how='left', predicate='within')

nb_associes = transports_iris['code_iris'].notna().sum()
print(f"OK - {nb_associes} arrêts associés à un IRIS")
print(f"     {len(transports_iris) - nb_associes} arrêts hors IRIS")

# ============================================================================
# 5. STATISTIQUES PAR IRIS
# ============================================================================
print("\n" + "=" * 80)
print("ETAPE 5 : Agrégation et calculs")
print("-" * 80)

stats_iris = transports_iris.groupby('code_iris').size().reset_index(name='total_arrets')

pivot = transports_iris.pivot_table(index='code_iris', columns='categorie', 
                                   aggfunc='size', fill_value=0).reset_index()

stats_iris = iris[['code_iris', 'LIB_IRIS', 'geometry']].merge(stats_iris, on='code_iris', how='left')
stats_iris = stats_iris.merge(pivot, on='code_iris', how='left')

# >>> Correction dynamique des colonnes
for c in ['Bus', 'Métro', 'Train']:
    if c not in stats_iris.columns:
        stats_iris[c] = 0

stats_iris['total_arrets'] = stats_iris['total_arrets'].fillna(0).astype(int)
for c in ['Bus', 'Métro', 'Train']:
    stats_iris[c] = stats_iris[c].fillna(0).astype(int)

print("\nColonnes finales disponibles :", [c for c in ['Bus', 'Métro', 'Train'] if c in stats_iris.columns])

# Calcul des surfaces (Lambert 93)
stats_iris = stats_iris.to_crs(epsg=2154)
stats_iris['surface_km2'] = stats_iris.geometry.area / 1_000_000

# Calcul densité
stats_iris['densite_arrets'] = stats_iris['total_arrets'] / stats_iris['surface_km2']
stats_iris['densite_arrets'] = stats_iris['densite_arrets'].replace([float('inf')], 0).round(1)

print(f"OK - Densités calculées pour {len(stats_iris)} IRIS")

# TOP 10
top10 = stats_iris.nlargest(10, 'densite_arrets')
print("\nTop 10 IRIS les plus denses en transports en commun :")
print(top10[['LIB_IRIS', 'total_arrets', 'densite_arrets']])

# ============================================================================
# 6. GRAPHIQUES
# ============================================================================
print("\n" + "=" * 80)
print("ETAPE 6 : Generation des graphiques")
print("-" * 80)
print("Nombre de graphiques a generer : 2")

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================ #
# GRAPHIQUE 1 : TOP 10 CLASSEMENT
# ============================================================================ #
print("\nCréation du graphique 1 : Classement du Top 10...")

fig, ax = plt.subplots(figsize=(14, 8))
top10_sorted = top10.sort_values('densite_arrets', ascending=True)

bars = ax.barh(
    range(len(top10_sorted)),
    top10_sorted['densite_arrets'],
    color=sns.color_palette("RdYlGn_r", len(top10_sorted))
)

labels = [row['LIB_IRIS'][:40] if pd.notna(row['LIB_IRIS']) else 'Sans nom'
          for _, row in top10_sorted.iterrows()]

ax.set_yticks(range(len(top10_sorted)))
ax.set_yticklabels(labels)

for i, (_, row) in enumerate(top10_sorted.iterrows()):
    ax.text(row['densite_arrets'] + 1, i, f"{row['densite_arrets']:.1f}", 
            va='center', fontweight='bold')

ax.set_xlabel('Densité d’arrêts (par km²)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 des IRIS de Rennes les plus denses en transports en commun',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
print("OK - Graphique 1 généré")

# ============================================================================ #
# GRAPHIQUE 2 : COMPOSITION PAR TYPE DE TRANSPORT
# ============================================================================ #
print("\nCréation du graphique 2 : Composition par type de transport...")

fig, ax = plt.subplots(figsize=(14, 8))
x = range(len(top10))
width = 0.25

ax.bar([i - width for i in x], top10['Bus'].values, width, label='Bus', color='#FF6B6B')
ax.bar(x, top10['Métro'].values, width, label='Métro', color='#4ECDC4')
ax.bar([i + width for i in x], top10['Train'].values, width, label='Train', color='#45B7D1')

labels = [row['LIB_IRIS'][:25] if pd.notna(row['LIB_IRIS']) else 'Sans nom'
          for _, row in top10.iterrows()]

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Nombre d’arrêts', fontsize=12, fontweight='bold')
ax.set_title('Composition du Top 10 des IRIS par type de transport',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
print("OK - Graphique 2 généré")