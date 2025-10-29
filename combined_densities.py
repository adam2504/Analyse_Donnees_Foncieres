"""
Analyse des IRIS de Rennes les plus denses en commerces vivants
================================================================
Version legere : affichage direct des graphiques sans sauvegarde
"""

import os
import pandas as pd
import geopandas as gpd
import plotly.express as px
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
# Télécharger une seule fois
if not os.path.exists(local_path):
    print("Téléchargement du fichier...")
    r = requests.get(url_iris)
    with open(local_path, "wb") as f:
        f.write(r.content)
# Lister les couches avec fiona
from fiona import listlayers
layers = listlayers(local_path)
print("Couches disponibles:", layers)
# Charger la couche principale
iris_start = gpd.read_file(local_path, layer=layers[0])
iris_start.head()
print(f"OK - {len(iris_start)} IRIS geographiques telecharges")

print("Telechargement de la base de reference IRIS depuis HuggingFace...")
url_ref_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"
iris_noms = pd.read_excel(url_ref_iris)
iris_noms = iris_noms.rename(columns={'CODE_IRIS': 'code_iris'})
print(f"OK - {len(iris_noms)} IRIS charges dans la base nationale")

print("\nFiltrage sur la commune de Rennes (code INSEE 35238)...")
iris_rennes = iris_start[iris_start['nom_commune'].str.contains("Rennes", case=False, na=False)].copy()
print(f"OK - {len(iris_rennes)} IRIS conserves pour Rennes")

print("\nFusion des noms avec les geometries...")
iris = iris_rennes.merge(iris_noms[['code_iris', 'LIB_IRIS', 'LIBCOM']], 
                         on='code_iris', how='left')
print(f"OK - Fusion terminee : {len(iris)} IRIS avec noms et geometries")

# ============================================================================
# 2. RÉCUPÉRATION DES COMMERCES OSM
# ============================================================================
print("\n" + "=" * 80)
print("ETAPE 2 : Recuperation des commerces depuis OpenStreetMap")
print("-" * 80)

overpass_url = "http://overpass-api.de/api/interpreter"

print("Construction de la requete Overpass API...")
print("Types de commerces recherches :")
print("  - Restaurants et fast-foods")
print("  - Bars, pubs et cafes")
print("  - Supermarches et epiceries")

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

print("\nEnvoi de la requete a l'API Overpass...")
print("(Cette etape peut prendre 30-60 secondes)")

try:
    response = requests.post(overpass_url, data={"data": overpass_query}, timeout=120)
    response.raise_for_status()
    osm_data = response.json()
    print(f"OK - {len(osm_data['elements'])} commerces recuperes")
except Exception as e:
    print(f"ERREUR - Impossible de recuperer les donnees OSM : {e}")
    exit(1)

# ============================================================================
# 3. CONVERSION ET CATÉGORISATION
# ============================================================================
print("\n" + "=" * 80)
print("ETAPE 3 : Conversion et categorisation des commerces")
print("-" * 80)

commerces_list = []

print("Traitement des elements OpenStreetMap...")
for element in osm_data['elements']:
    tags = element.get('tags', {})
    
    # Extraction coordonnees
    if element['type'] == 'node':
        lat, lon = element['lat'], element['lon']
    elif 'center' in element:
        lat, lon = element['center']['lat'], element['center']['lon']
    else:
        continue
    
    # Categorisation
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
    
    commerces_list.append({
        'categorie': categorie,
        'latitude': lat,
        'longitude': lon
    })

print(f"OK - {len(commerces_list)} commerces traites")

print("\nCreation du GeoDataFrame avec les coordonnees...")
commerces_df = pd.DataFrame(commerces_list)
commerces_gdf = gpd.GeoDataFrame(
    commerces_df,
    geometry=gpd.points_from_xy(commerces_df.longitude, commerces_df.latitude),
    crs="EPSG:4326"
)

print("Reprojection dans le systeme de coordonnees des IRIS...")
commerces_gdf = commerces_gdf.to_crs(iris.crs)
print(f"OK - {len(commerces_gdf)} commerces geolocalises")

print("\nRepartition par categorie :")
print(commerces_gdf['categorie'].value_counts().to_string())

# ============================================================================
# 4. JOINTURE SPATIALE
# ============================================================================
print("\n" + "=" * 80)
print("ETAPE 4 : Jointure spatiale (commerces x IRIS)")
print("-" * 80)

print("Association de chaque commerce a son IRIS...")
commerces_iris = gpd.sjoin(commerces_gdf, iris[['code_iris', 'LIB_IRIS', 'geometry']], 
                            how='left', predicate='within')

nb_associes = commerces_iris['code_iris'].notna().sum()
print(f"OK - {nb_associes} commerces associes a un IRIS")
print(f"     {len(commerces_iris) - nb_associes} commerces hors IRIS")

print("\nComptage du nombre de commerces par IRIS...")
stats_iris = commerces_iris.groupby('code_iris').size().reset_index(name='total_commerces')
print(f"OK - {len(stats_iris)} IRIS contiennent au moins 1 commerce")

print("\nComptage par categorie de commerce...")
pivot = commerces_iris.pivot_table(index='code_iris', columns='categorie', 
                                   aggfunc='size', fill_value=0).reset_index()
print(f"OK - Tableau croise cree")

print("\nFusion avec les informations des IRIS...")
stats_iris = stats_iris.merge(pivot, on='code_iris', how='left')
stats_iris = iris[['code_iris', 'LIB_IRIS', 'geometry']].merge(stats_iris, on='code_iris', how='left')

print("Remplissage des valeurs manquantes par 0...")
for col in ['total_commerces', 'Restaurant', 'Bar/Café', 'Supermarché']:
    if col in stats_iris.columns:
        stats_iris[col] = stats_iris[col].fillna(0).astype(int)
    else:
        stats_iris[col] = 0

print(f"OK - Table finale : {len(stats_iris)} IRIS avec statistiques")

print("\nCalcul de la surface de chaque IRIS...")
print("Reprojection en Lambert 93 (systeme metrique francais)...")
stats_iris = gpd.GeoDataFrame(stats_iris, geometry='geometry', crs=iris.crs)

# Reprojeter en Lambert 93 (EPSG:2154) pour avoir des surfaces en mètres
stats_iris_projected = stats_iris.to_crs(epsg=2154)
stats_iris['surface_km2'] = stats_iris_projected.geometry.area / 1_000_000

print(f"OK - Surfaces calculees (min: {stats_iris['surface_km2'].min():.3f} km², max: {stats_iris['surface_km2'].max():.3f} km²)")

# Vérification
if stats_iris['surface_km2'].max() == 0:
    print("ATTENTION - Probleme de calcul de surface, utilisation de la geometrie originale...")
    stats_iris['surface_km2'] = stats_iris.geometry.area / 1_000_000
    print(f"OK - Surfaces recalculees (min: {stats_iris['surface_km2'].min():.6f} km², max: {stats_iris['surface_km2'].max():.3f} km²)")


print("\nCalcul du score de densite pondere...")

print("\nCalcul du score de densite pondere...")
print("Formule : (Restaurants x 2 + Bars/Cafes x 2 + Supermarches x 1.5) / surface_km2")

stats_iris['score_densite'] = (
    stats_iris['Restaurant'] * 2.0 +
    stats_iris['Bar/Café'] * 2.0 +
    stats_iris['Supermarché'] * 1.5
) / stats_iris['surface_km2']

stats_iris['score_densite'] = stats_iris['score_densite'].replace([float('inf')], 0).round(1)
print(f"OK - Scores calcules pour tous les IRIS")

# ============================================================================
# 5. TOP 10
# ============================================================================
print("\n" + "=" * 80)
print("TOP 10 DES IRIS LES PLUS DENSES EN COMMERCES VIVANTS")
print("=" * 80)
print("\nCalcul de la densite pour chaque IRIS...")
print("Formule : (Restaurants x2 + Bars x2 + Supermarches x1.5) / surface")

iris_valides = stats_iris[(stats_iris['total_commerces'] > 0) & (stats_iris['surface_km2'] > 0)]
top10 = iris_valides.nlargest(10, 'score_densite')

print(f"\nRESULTAT : {len(iris_valides)} IRIS contiennent au moins 1 commerce")
print("\n" + "=" * 80)

print(f"\n{'Rang':<6} {'IRIS':<45} {'Restos':<8} {'Bars':<8} {'Supers':<8} {'Total':<8}")
print("-" * 85)

for idx, (_, row) in enumerate(top10.iterrows(), 1):
    nom = row['LIB_IRIS'] if pd.notna(row['LIB_IRIS']) else 'Sans nom'
    nom = nom[:43]
    print(f"{idx:<6} {nom:<45} {int(row['Restaurant']):<8} {int(row['Bar/Café']):<8} "
          f"{int(row['Supermarché']):<8} {int(row['total_commerces']):<8}")

print("\n" + "=" * 80)
print(f"STATISTIQUES GLOBALES")
print("=" * 80)
print(f"IRIS total a Rennes : {len(iris)}")
print(f"IRIS avec commerces : {len(iris_valides)}")
print(f"IRIS sans commerce : {len(iris) - len(iris_valides)}")
print(f"\nTotal restaurants : {int(stats_iris['Restaurant'].sum())}")
print(f"Total bars/cafes : {int(stats_iris['Bar/Café'].sum())}")
print(f"Total supermarches : {int(stats_iris['Supermarché'].sum())}")
print("=" * 80)

# ============================================================================
# 6. GRAPHIQUES
# ============================================================================
print("\n" + "=" * 80)
print("ETAPE 5 : Generation des graphiques")
print("-" * 80)
print("Nombre de graphiques a generer : 2")

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# GRAPHIQUE 1 : TOP 10 CLASSEMENT
# ============================================================================
print("\nCreation du graphique 1 : Classement du Top 10...")

fig, ax = plt.subplots(figsize=(14, 8))

top10_sorted = top10.sort_values('score_densite', ascending=True)

bars = ax.barh(
    range(len(top10_sorted)),
    top10_sorted['total_commerces'],
    color=sns.color_palette("RdYlGn_r", len(top10_sorted))
)

labels = [row['LIB_IRIS'][:40] if pd.notna(row['LIB_IRIS']) else 'Sans nom' 
          for _, row in top10_sorted.iterrows()]

ax.set_yticks(range(len(top10_sorted)))
ax.set_yticklabels(labels)

for i, (_, row) in enumerate(top10_sorted.iterrows()):
    ax.text(row['total_commerces'] + 1, i, f"{int(row['total_commerces'])}", 
            va='center', fontweight='bold')

ax.set_xlabel('Nombre de commerces', fontsize=12, fontweight='bold')
ax.set_title('Top 10 des IRIS de Rennes les plus denses en commerces vivants',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
print("OK - Graphique 1 genere")
plt.show()

# ============================================================================
# GRAPHIQUE 2 : COMPOSITION
# ============================================================================
print("\nCreation du graphique 2 : Composition par type de commerce...")

fig, ax = plt.subplots(figsize=(14, 8))

x = range(len(top10))
width = 0.25

ax.bar([i - width for i in x], top10['Restaurant'].values, width, 
       label='Restaurants', color='#FF6B6B')
ax.bar(x, top10['Bar/Café'].values, width, 
       label='Bars/Cafes', color='#4ECDC4')
ax.bar([i + width for i in x], top10['Supermarché'].values, width, 
       label='Supermarches', color='#45B7D1')

labels = [row['LIB_IRIS'][:25] if pd.notna(row['LIB_IRIS']) else 'Sans nom' 
          for _, row in top10.iterrows()]

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Nombre de commerces', fontsize=12, fontweight='bold')
ax.set_title('Composition du Top 10 des IRIS par type de commerce',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
print("OK - Graphique 2 genere")
plt.show()


url_df_enseignement_sup = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"

df_enseignement_sup = pd.read_csv(url_df_enseignement_sup, delimiter=';')

df_rennes = df_enseignement_sup[df_enseignement_sup['Commune'] == 'Rennes']

print(f"Nombre de lignes à Rennes :", df_rennes.shape[0])
print(f"Nombres de lignes avec gps à Rennes :", (df_rennes['gps'].isnull() == False).sum())
print(f"Nombres de lignes sans gps à Rennes avant drop:", df_rennes['gps'].isnull().sum())

df_rennes = df_rennes.dropna(subset=['gps'])

print(f"Nombres de lignes sans gps à Rennes après drop:", df_rennes['gps'].isnull().sum())

# Nettoyage des coordonnées
df_rennes[['lat', 'lon']] = df_rennes['gps'].str.split(',', expand=True)
df_rennes['lat'] = df_rennes['lat'].astype(float)
df_rennes['lon'] = df_rennes['lon'].astype(float)

print(f"Nombre de lignes à Rennes :", df_rennes[['lat', 'lon']].shape[0])
print(f"Nombres de lignes avec lat et lon à Rennes :", ((df_rennes['lat'].isnull() == False) & (df_rennes['lon'].isnull() == False)).sum())
print(f"Nombres de lignes sans lat et lon à Rennes avant drop:", ((df_rennes['lat'].isnull()) & (df_rennes['lon'].isnull())).sum())

# On enlève les lignes sans coordonnées
df_rennes = df_rennes.dropna(subset=['lat', 'lon'])

print(f"Nombres de lignes sans lat et lon à Rennes après drop:", ((df_rennes['lat'].isnull()) & (df_rennes['lon'].isnull())).sum())

# Transformation en GeoDataFrame 
df_rennes = gpd.GeoDataFrame(
    df_rennes,
    geometry=gpd.points_from_xy(df_rennes.lon, df_rennes.lat),
    crs="EPSG:4326"  # WGS84
)

url_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/contours-iris-pe.gpkg"
local_path = "contours-iris-pe.gpkg"

# Télécharger une seule fois
if not os.path.exists(local_path):
    print("Téléchargement du fichier...")
    r = requests.get(url_iris)
    with open(local_path, "wb") as f:
        f.write(r.content)

# Lister les couches avec fiona
from fiona import listlayers
layers = listlayers(local_path)
print("Couches disponibles:", layers)

# Charger la couche principale
iris = gpd.read_file(local_path, layer=layers[0])
url_ref_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"

iris_noms = pd.read_excel(url_ref_iris)

iris_noms = iris_noms.rename(columns={'CODE_IRIS': 'code_iris'})

# fusion avec ton GeoDataFrame (qui contient les codes IRIS)
iris = iris.merge(iris_noms[['code_iris', 'LIB_IRIS', 'LIBCOM']], on='code_iris', how='left')

# Filtrer
iris_rennes = iris[iris['nom_commune'].str.contains("Rennes", case=False, na=False)].copy()

print(f"Rennes contient", len(iris_rennes), "IRIS")

# On met tout dans la même CRS projetée (ex: EPSG:2154 ou utiliser EPSG:4326 par la suite)
iris_rennes = iris_rennes.to_crs(epsg=4326)
df_rennes = df_rennes.to_crs(epsg=4326)

# spatial join : chaque établissement rattaché à un IRIS
etabs_par_iris_rennes = gpd.sjoin(df_rennes, iris_rennes, how="inner", predicate="within")

iris_rennes_stats = iris_rennes.copy()

# Nombre d'établissements
iris_rennes_stats = iris_rennes_stats.merge(
    etabs_par_iris_rennes.groupby('LIB_IRIS').size().reset_index(name='nb_etabs'),
    on='LIB_IRIS', how='left'
).fillna({'nb_etabs':0})
iris_rennes_stats['nb_etabs'] = iris_rennes_stats['nb_etabs'].astype(int)

# Nombre total d'étudiants
iris_rennes_stats = iris_rennes_stats.merge(
    etabs_par_iris_rennes.groupby('LIB_IRIS')['nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE'].sum().reset_index(name='nb_etudiants'),
    on='LIB_IRIS', how='left'
).fillna({'nb_etudiants':0})
iris_rennes_stats['nb_etudiants'] = iris_rennes_stats['nb_etudiants'].astype(int)

iris_rennes_stats = iris_rennes_stats.to_crs(epsg=2154)  # crs projetée pour calculer surface en m²
iris_rennes_stats['area_m2'] = iris_rennes_stats.geometry.area
iris_rennes_stats['area_km2'] = iris_rennes_stats['area_m2'] / 1000000
iris_rennes_stats['etabs_per_km2'] = iris_rennes_stats['nb_etabs'] / iris_rennes_stats['area_km2']
iris_rennes_stats['students_per_km2'] = iris_rennes_stats['nb_etudiants'] / iris_rennes_stats['area_km2']

colorscale = "YlOrRd"  # jaune = faible, rouge = élevé

# Top 10 IRIS par densité (par défaut)
top10_density = iris_rennes_stats.sort_values(by='students_per_km2', ascending=False).head(10)

# Top 10 IRIS par nombre d'étudiants
top10_students = iris_rennes_stats.sort_values(by='nb_etudiants', ascending=False).head(10)

# Carte initiale avec coloraxis défini
fig = px.bar(
    top10_density,
    x='LIB_IRIS',
    y='students_per_km2',
    color='students_per_km2',
    hover_data=['nb_etudiants','area_km2','nb_etabs'],
    color_continuous_scale=colorscale,
    title="Top 10 IRIS les plus denses en étudiants - Rennes"
)

# On associe explicitement la trace à coloraxis et fixe les limites
fig.update_traces(marker_coloraxis="coloraxis")

# Définition globale de coloraxis
fig.update_layout(
    coloraxis=dict(
        colorscale=colorscale,
        reversescale=False,  # rouge = élevé, jaune = faible
        cmin=top10_density['students_per_km2'].min(),
        cmax=top10_density['students_per_km2'].max(),
        colorbar=dict(title="Étudiants/km²")
    )
)

# Boutons interactifs : on ne touche qu'à x, y, marker.color et coloraxis
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            x=0.0, y=1.05, showactive=True,
            buttons=[
                dict(
                    label="Densité",
                    method="update",
                    args=[
                        {"x": [top10_density['LIB_IRIS']],
                         "y": [top10_density['students_per_km2']],
                         "marker.color": [top10_density['students_per_km2']]},
                        {"coloraxis.cmin": top10_density['students_per_km2'].min(),
                         "coloraxis.cmax": top10_density['students_per_km2'].max(),
                         "coloraxis.colorbar.title": "Étudiants/km²",
                         "yaxis.title": "Étudiants / km²"}
                    ]
                ),
                dict(
                    label="Nombre d'étudiants",
                    method="update",
                    args=[
                        {"x": [top10_students['LIB_IRIS']],
                         "y": [top10_students['nb_etudiants']],
                         "marker.color": [top10_students['nb_etudiants']]},
                        {"coloraxis.cmin": top10_students['nb_etudiants'].min(),
                         "coloraxis.cmax": top10_students['nb_etudiants'].max(),
                         "coloraxis.colorbar.title": "Nombre d'étudiants",
                         "yaxis.title": "Nombre d'étudiants"}
                    ]
                ),
            ]
        )
    ]
)

fig.update_layout(xaxis_title="Quartier (IRIS)", yaxis_title="Étudiants / km²")
fig.show()

# ============================================================================
# GRAPHIQUE SCATTER : CORRELATION COMMERCES VS ETUDIANTS
# ============================================================================
print("\n" + "=" * 80)
print("GRAPHIQUE SCATTER : Correlation top10 commerces vs top10 etudiants")
print("-" * 80)

# ============================================================================
# AJOUT TRANSPORT DENSITY
# ============================================================================
print("\n" + "=" * 80)
print("AJOUT DONNEES TRANSPORT RENNES")
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

print("Envoi de la requete a l'API Overpass pour transports...")
try:
    response = requests.post(overpass_url, data={"data": overpass_query}, timeout=120)
    response.raise_for_status()
    osm_data = response.json()
    print(f"OK - {len(osm_data['elements'])} arpets recuperes")
except Exception as e:
    print(f"ERREUR - Impossible de recuperer les donnees OSM transports : {e}")
    exit(1)

transports_list = []
for element in osm_data['elements']:
    tags = element.get('tags', {})
    if element['type'] == 'node':
        lat, lon = element['lat'], element['lon']
    elif 'center' in element:
        lat, lon = element['center']['lat'], element['center']['lon']
    else:
        continue
    categorie = 'Autre'
    if 'bus' in tags.get('highway', '') or tags.get('public_transport') == 'stop_position':
        categorie = 'Bus'
    elif 'subway' in tags.get('railway', '') or 'subway' in tags.get('public_transport', ''):
        categorie = 'Metro'
    elif 'station' in tags.get('railway', '') or 'halt' in tags.get('railway', ''):
        categorie = 'Train'
    transports_list.append({'categorie': categorie, 'latitude': lat, 'longitude': lon})

transports_df = pd.DataFrame(transports_list)
transports_gdf = gpd.GeoDataFrame(transports_df, geometry=gpd.points_from_xy(transports_df.longitude, transports_df.latitude), crs="EPSG:4326").to_crs(iris.crs)
transports_iris = gpd.sjoin(transports_gdf, iris_rennes[['code_iris', 'LIB_IRIS', 'geometry']], how='left', predicate='intersects')
stats_transports = transports_iris.groupby('code_iris').size().reset_index(name='total_arrets')
stats_transports = iris_rennes[['code_iris', 'LIB_IRIS']].merge(stats_transports, on='code_iris', how='left')
stats_transports['total_arrets'] = stats_transports['total_arrets'].fillna(0).astype(int)
stats_transports = stats_transports.merge(iris_rennes[['code_iris', 'geometry']], on='code_iris', how='left')
stats_transports = gpd.GeoDataFrame(stats_transports, geometry='geometry', crs=iris_rennes.crs)
stats_transports = stats_transports.to_crs(epsg=2154)
stats_transports['area_m2'] = stats_transports.geometry.area
stats_transports['area_km2'] = stats_transports['area_m2'] / 1000000
stats_transports['densite_arrets_km2'] = (stats_transports['total_arrets'] / stats_transports['area_km2']).replace([float('inf')], 0).round(1)

print(f"OK - Transport density calculated for {len(stats_transports)} IRIS")

# ============================================================================
# TOP 146 (ALL IRIS) CALCULATION
# ============================================================================
print("\n" + "=" * 80)
print("TOP 146 IRIS DENSITIES")
print("-" * 80)

# Top 146 densité commerciale (score_densite)
top146_commerce = stats_iris.nlargest(146, 'score_densite')[['LIB_IRIS', 'score_densite', 'total_commerces', 'surface_km2']].copy()
print("TOP 146 commerce density calculated")

# Top 146 densité étudiante
top146_etudiants = iris_rennes_stats.nlargest(146, 'students_per_km2')[['LIB_IRIS', 'students_per_km2', 'nb_etudiants', 'area_km2']].copy()
print("TOP 146 student density calculated")

# Top 146 densité transport
top146_transport = stats_transports.nlargest(146, 'densite_arrets_km2')[['LIB_IRIS', 'densite_arrets_km2', 'total_arrets', 'area_km2']].copy()
print("TOP 146 transport density calculated")

# Merge all top 146
merged_146 = top146_commerce.merge(
    top146_etudiants[['LIB_IRIS', 'students_per_km2', 'nb_etudiants']].rename(columns={'area_km2': 'etudiants_area_km2'}),
    on='LIB_IRIS',
    how='outer'
).merge(
    top146_transport[['LIB_IRIS', 'densite_arrets_km2', 'total_arrets']].rename(columns={'area_km2': 'transport_area_km2'}),
    on='LIB_IRIS',
    how='outer'
)

# Fill NaN with 0
merged_146 = merged_146.fillna(0)

print(f"OK - {len(merged_146)} IRIS combines avec les 3 densites")

# ============================================================================
# 3D SCATTER PLOT
# ============================================================================
print("\n" + "=" * 80)
print("GRAPHIQUE 3D SCATTER : Correlation des trois densites")
print("-" * 80)

import plotly.graph_objects as go

fig_3d = go.Figure(data=[go.Scatter3d(
    x=merged_146['students_per_km2'],
    y=merged_146['score_densite'],
    z=merged_146['densite_arrets_km2'],
    mode='markers',
    marker=dict(
        size=5,
        color=merged_146['students_per_km2'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Densité étudiante (km²)")
    ),
    text=merged_146['LIB_IRIS'],
    hovertemplate=(
        '<b>%{text}</b><br>' +
        'Étudiants/km²: %{x}<br>' +
        'Commerces/km²: %{y}<br>' +
        'Transports/km²: %{z}'
    )
)])

fig_3d.update_layout(
    title="Correlation 3D - Densité étudiante, commerciale et transport - Top 146 IRIS Rennes",
    scene=dict(
        xaxis=dict(title='Densité étudiante (étudiants/km²)', gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)'),
        yaxis=dict(title='Densité commerciale (score/km²)', gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)'),
        zaxis=dict(title='Densité transports (arrêts/km²)', gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)')
    ),
    width=800,
    height=600
)

print("OK - Graphique 3D scatter genere pour top 146")
fig_3d.show()
