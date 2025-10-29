# Script pour analyser la concentration étudiante à Rennes
# Ce script charge les données d'enseignement supérieur, filtre sur Rennes,
# et visualise la répartition des étudiants par IRIS avec une carte interactive.

import pandas as pd
import plotly.express as px
import geopandas as gpd
import requests
import os

# ============================================================================
# 1. CHARGEMENT DU DATASET ENSEIGNEMENT SUPÉRIEUR
# ============================================================================
# Chargement des données d'effectifs étudiants en enseignement supérieur pour la France
# Source: Atlas régional des effectifs étudiants inscrits par établissement

url_df_enseignement_sup = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"
df_enseignement_sup = pd.read_csv(url_df_enseignement_sup, delimiter=';')

# ============================================================================
# 2. FILTRE SUR RENNES
# ============================================================================
# Filtrage des données pour ne garder que les établissements situés à Rennes
# Nettoyage des coordonnées GPS manquantes

df_rennes = df_enseignement_sup[df_enseignement_sup['Commune'] == 'Rennes']
print(f"Nombre de lignes à Rennes :", df_rennes.shape[0])
print(f"Nombres de lignes avec gps à Rennes :", (df_rennes['gps'].isnull() == False).sum())
print(f"Nombres de lignes sans gps à Rennes avant drop:", df_rennes['gps'].isnull().sum())
df_rennes = df_rennes.dropna(subset=['gps'])
print(f"Nombres de lignes sans gps à Rennes après drop:", df_rennes['gps'].isnull().sum())

# Nettoyage des coordonnées
# Séparation des coordonnées GPS (format "lat,lon") en colonnes séparées et conversion en float
df_rennes[['lat', 'lon']] = df_rennes['gps'].str.split(',', expand=True)
df_rennes['lat'] = df_rennes['lat'].astype(float)
df_rennes['lon'] = df_rennes['lon'].astype(float)
print(f"Nombre de lignes à Rennes :", df_rennes[['lat', 'lon']].shape[0])
print(f"Nombres de lignes avec lat et lon à Rennes :", ((df_rennes['lat'].isnull() == False) & (df_rennes['lon'].isnull() == False)).sum())
print(f"Nombres de lignes sans lat et lon à Rennes avant drop:", ((df_rennes['lat'].isnull()) & (df_rennes['lon'].isnull())).sum())
# On enlève les lignes sans coordonnées
df_rennes = df_rennes.dropna(subset=['lat', 'lon'])
print(f"Nombres de lignes sans lat et lon à Rennes après drop:", ((df_rennes['lat'].isnull()) & (df_rennes['lon'].isnull())).sum())

# Transformation en GeoDataFrame pour faciliter les analyses géospatiales
df_rennes = gpd.GeoDataFrame(
    df_rennes,
    geometry=gpd.points_from_xy(df_rennes.lon, df_rennes.lat),
    crs="EPSG:4326"  # WGS84 (coordonnées géographiques)
)

# ============================================================================
# 3. CHARGEMENT DES IRIS
# ============================================================================
# Chargement des contours IRIS (Ilots Regroupés pour l'Information Statistique)
# Les IRIS sont des subdivisions statistiques fines des communes

url_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/contours-iris-pe.gpkg"
local_path = "contours-iris-pe.gpkg"

# Téléchargement du fichier de contours IRIS en local (une seule fois)
if not os.path.exists(local_path):
    print("Téléchargement du fichier...")
    r = requests.get(url_iris)
    with open(local_path, "wb") as f:
        f.write(r.content)

# Liste des couches disponibles dans le fichier GeoPackage
from fiona import listlayers
layers = listlayers(local_path)
print("Couches disponibles:", layers)
# Chargement de la couche principale des contours IRIS
iris = gpd.read_file(local_path, layer=layers[0])

# Chargement des noms et libellés des IRIS pour faciliter la lecture
url_ref_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"
iris_noms = pd.read_excel(url_ref_iris)
iris_noms = iris_noms.rename(columns={'CODE_IRIS': 'code_iris'})
# Fusion avec le GeoDataFrame pour ajouter les noms des IRIS et communes
iris = iris.merge(iris_noms[['code_iris', 'LIB_IRIS', 'LIBCOM']], on='code_iris', how='left')


# Filtrage pour ne garder que les IRIS de Rennes
iris_rennes = iris[iris['nom_commune'].str.contains("Rennes", case=False, na=False)].copy()
print(f"Rennes contient", len(iris_rennes), "IRIS")

# Conversion des systèmes de coordonnées en EPSG:4326 pour la compatibilité
iris_rennes = iris_rennes.to_crs(epsg=4326)
df_rennes = df_rennes.to_crs(epsg=4326)

# Jointure spatiale : rattachement de chaque établissement à l'IRIS qui le contient
etabs_par_iris_rennes = gpd.sjoin(df_rennes, iris_rennes, how="inner", predicate="within")
etabs_par_iris_rennes.head()

# Calcul des statistiques par IRIS
iris_rennes_stats = iris_rennes.copy()

# Nombre d'établissements par IRIS
iris_rennes_stats = iris_rennes_stats.merge(
    etabs_par_iris_rennes.groupby('LIB_IRIS').size().reset_index(name='nb_etabs'),
    on='LIB_IRIS', how='left'
).fillna({'nb_etabs':0})
iris_rennes_stats['nb_etabs'] = iris_rennes_stats['nb_etabs'].astype(int)

# Nombre total d'étudiants par IRIS (excluant les doubles inscriptions)
iris_rennes_stats = iris_rennes_stats.merge(
    etabs_par_iris_rennes.groupby('LIB_IRIS')['nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE'].sum().reset_index(name='nb_etudiants'),
    on='LIB_IRIS', how='left'
).fillna({'nb_etudiants':0})
iris_rennes_stats['nb_etudiants'] = iris_rennes_stats['nb_etudiants'].astype(int)

# Calcul des surfaces en mètres carrés (nécessite un CRS projeté)
iris_rennes_stats = iris_rennes_stats.to_crs(epsg=2154)  # RGF93 / Lambert-93 (métric)
iris_rennes_stats['area_m2'] = iris_rennes_stats.geometry.area
iris_rennes_stats['area_km2'] = iris_rennes_stats['area_m2'] / 1000000
iris_rennes_stats['etabs_per_km2'] = iris_rennes_stats['nb_etabs'] / iris_rennes_stats['area_km2']
iris_rennes_stats['students_per_km2'] = iris_rennes_stats['nb_etudiants'] / iris_rennes_stats['area_km2']


# Re-conversion en EPSG:4326 pour la visualisation cartographique
iris_plot = iris_rennes_stats.to_crs(epsg=4326)

# Visualisation interactive avec Plotly
colorscale = "YlOrRd"  # jaune = faible concentration, rouge = élevé

# Création d'une carte choroplèthe initiale montrant la densité étudiante par IRIS
fig = px.choropleth_mapbox(
    iris_plot,
    geojson=iris_plot.__geo_interface__,
    locations=iris_plot.index,
    color='students_per_km2',  # couleur basée sur la densité étudiante initiale
    hover_name='LIB_IRIS',
    hover_data=['nb_etabs','area_km2','nb_etudiants'],  # données affichées au survol
    mapbox_style="carto-positron",
    center={"lat":48.117, "lon":-1.677},  # centre sur Rennes
    zoom=12,
    opacity=0.6,
)

# Configuration de l'axe de couleurs partagé
fig.update_traces(
    coloraxis="coloraxis"
)

# Définition de l'échelle de couleurs globale avec les limites appropriées
fig.update_layout(
    coloraxis=dict(
        colorscale=colorscale,
        cmin=iris_plot['students_per_km2'].min(),
        cmax=iris_plot['students_per_km2'].max(),
        colorbar=dict(title="Étudiants/km²")
    )
)

# Ajout des points représentant les établissements d'enseignement supérieur
fig.add_scattermapbox(
    lat=df_rennes['lat'],
    lon=df_rennes['lon'],
    mode='markers',
    marker=dict(size=6, color='blue'),
    text=etabs_par_iris_rennes["libellé de l'établissement"],  #tooltip avec nom de l'établissement
    name='Établissements OSM'
)

# Ajout de boutons interactifs pour basculer entre différentes visualisations
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
                        {"z": [iris_plot['students_per_km2']]},  # variable de couleur
                        {"coloraxis.cmin": iris_plot['students_per_km2'].min(),
                         "coloraxis.cmax": iris_plot['students_per_km2'].max(),
                         "coloraxis.colorbar.title": "Étudiants/km²"}
                    ]
                ),
                dict(
                    label="Nombre d'étudiants",
                    method="update",
                    args=[
                        {"z": [iris_plot['nb_etudiants']]},  # variable de couleur
                        {"coloraxis.cmin": iris_plot['nb_etudiants'].min(),
                         "coloraxis.cmax": iris_plot['nb_etudiants'].max(),
                         "coloraxis.colorbar.title": "Nombre d'étudiants"}
                    ]
                ),
            ]
        )
    ]
)

# Configuration finale du layout et affichage
fig.update_layout(margin={"r":0,"t":75,"l":0,"b":0}, title="Étudiants sup. par IRIS - Rennes")
fig.show()
