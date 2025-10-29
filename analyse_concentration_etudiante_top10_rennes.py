# ============================================================================
# ANALYSE DE LA CONCENTRATION ÉTUDIANTE À RENNES
# AVEC WIDGETS INTERACTIFS (ipywidgets + plotly)
# ============================================================================

import pandas as pd
import plotly.express as px
import geopandas as gpd
import requests
import os
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display

# ============================================================================
# 1. CHARGEMENT DU DATASET ENSEIGNEMENT SUPÉRIEUR
# ============================================================================
url_df_enseignement_sup = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"
df_enseignement_sup = pd.read_csv(url_df_enseignement_sup, delimiter=';')

# ============================================================================
# 2. FILTRE SUR RENNES
# ============================================================================
df_rennes = df_enseignement_sup[df_enseignement_sup['Commune'] == 'Rennes'].dropna(subset=['gps'])
df_rennes[['lat', 'lon']] = df_rennes['gps'].str.split(',', expand=True)
df_rennes['lat'] = df_rennes['lat'].astype(float)
df_rennes['lon'] = df_rennes['lon'].astype(float)
df_rennes = gpd.GeoDataFrame(df_rennes, geometry=gpd.points_from_xy(df_rennes.lon, df_rennes.lat), crs="EPSG:4326")

# ============================================================================
# 3. CHARGEMENT DES CONTOURS IRIS
# ============================================================================
url_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/contours-iris-pe.gpkg"
local_path = "contours-iris-pe.gpkg"

if not os.path.exists(local_path):
    print("Téléchargement du fichier IRIS...")
    r = requests.get(url_iris)
    with open(local_path, "wb") as f:
        f.write(r.content)

from fiona import listlayers
layers = listlayers(local_path)
iris = gpd.read_file(local_path, layer=layers[0])

# Chargement des noms IRIS
url_ref_iris = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"
iris_noms = pd.read_excel(url_ref_iris)
iris_noms = iris_noms.rename(columns={'CODE_IRIS': 'code_iris'})
iris = iris.merge(iris_noms[['code_iris', 'LIB_IRIS', 'LIBCOM']], on='code_iris', how='left')

# Filtrage Rennes uniquement
iris_rennes = iris[iris['nom_commune'].str.contains("Rennes", case=False, na=False)].copy()
iris_rennes = iris_rennes.to_crs(epsg=4326)
df_rennes = df_rennes.to_crs(epsg=4326)

# ============================================================================
# 4. JOINTURE SPATIALE & STATISTIQUES
# ============================================================================
etabs_par_iris_rennes = gpd.sjoin(df_rennes, iris_rennes, how="inner", predicate="within")

iris_rennes_stats = iris_rennes.copy()

# Nombre d’établissements par IRIS
iris_rennes_stats = iris_rennes_stats.merge(
    etabs_par_iris_rennes.groupby('LIB_IRIS').size().reset_index(name='nb_etabs'),
    on='LIB_IRIS', how='left'
).fillna({'nb_etabs': 0}).astype({'nb_etabs': int})

# Nombre total d’étudiants par IRIS
iris_rennes_stats = iris_rennes_stats.merge(
    etabs_par_iris_rennes.groupby('LIB_IRIS')['nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE'].sum().reset_index(name='nb_etudiants'),
    on='LIB_IRIS', how='left'
).fillna({'nb_etudiants': 0}).astype({'nb_etudiants': int})

# Surface et densité
iris_rennes_stats = iris_rennes_stats.to_crs(epsg=2154)
iris_rennes_stats['area_m2'] = iris_rennes_stats.geometry.area
iris_rennes_stats['area_km2'] = iris_rennes_stats['area_m2'] / 1_000_000
iris_rennes_stats['students_per_km2'] = iris_rennes_stats['nb_etudiants'] / iris_rennes_stats['area_km2']

# ============================================================================
# 5. WIDGET INTERACTIF
# ============================================================================

colorscale = "YlOrRd"

def make_fig(mode="Densité", top_n=10):
    """
    Génère un graphique en fonction :
    - du mode sélectionné ('Densité' ou 'Nombre d’étudiants')
    - du nombre de top IRIS à afficher
    """
    if mode == "Densité":
        data = iris_rennes_stats.sort_values(by='students_per_km2', ascending=False).head(top_n)
        y_col = "students_per_km2"
        title = f"Top {top_n} IRIS les plus denses en étudiants - Rennes"
        color_title = "Étudiants/km²"
    else:
        data = iris_rennes_stats.sort_values(by='nb_etudiants', ascending=False).head(top_n)
        y_col = "nb_etudiants"
        title = f"Top {top_n} IRIS par nombre total d'étudiants - Rennes"
        color_title = "Nombre d'étudiants"

    fig = px.bar(
        data,
        x='LIB_IRIS',
        y=y_col,
        color=y_col,
        hover_data=['nb_etudiants', 'area_m2', 'nb_etabs'],
        color_continuous_scale=colorscale,
        title=title
    )

    fig.update_layout(
        xaxis_title="Quartier (IRIS)",
        yaxis_title=color_title,
        coloraxis_colorbar=dict(title=color_title),
        margin=dict(t=60, l=50, r=50, b=50)
    )
    fig.show()


# Widgets
mode_selector = widgets.ToggleButtons(
    options=['Densité', 'Nombre d\'étudiants'],
    description='Afficher :',
    button_style='info',
    style={'description_width': 'initial'}
)

top_slider = widgets.IntSlider(
    value=10,
    min=5,
    max=30,
    step=1,
    description='Top N IRIS :',
    continuous_update=False,
    style={'description_width': 'initial'}
)

# Liaison interactive
interactive_plot = interactive(make_fig, mode=mode_selector, top_n=top_slider)
display(interactive_plot)
