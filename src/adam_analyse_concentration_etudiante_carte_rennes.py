# ======================================================================
# MODULE : analyse_concentration_etudiante_carte_rennes.py
# ======================================================================
# Objectif : analyser et visualiser la concentration étudiante à Rennes
# Auteur : Adam Jouini
# ======================================================================

import os
import requests
import pandas as pd
import geopandas as gpd
import plotly.express as px
from fiona import listlayers

# ----------------------------------------------------------------------
# 1. Chargement des données
# ----------------------------------------------------------------------
def charger_donnees_enseignement_sup():
    """Charge le dataset national des effectifs étudiants."""
    print("\n[1/6] 📦 Chargement des données d’enseignement supérieur...")
    url = (
        "https://huggingface.co/datasets/analysedonneesfoncieresdata/"
        "analyse_fonciere_data/resolve/main/"
        "fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"
    )
    df = pd.read_csv(url, delimiter=';')
    print(f"✅ Données chargées : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    return df


# ----------------------------------------------------------------------
# 2. Filtrage et nettoyage pour Rennes
# ----------------------------------------------------------------------
def filtrer_donnees_rennes(df):
    """Filtre le dataset pour la commune de Rennes et nettoie les coordonnées GPS."""
    print("\n[2/6] 🧹 Filtrage des données pour Rennes...")
    df_rennes = df[df['Commune'] == 'Rennes'].copy()
    print(f"Nombre d'établissements à Rennes avant nettoyage : {len(df_rennes):,}")

    df_rennes = df_rennes.dropna(subset=['gps'])
    df_rennes[['lat', 'lon']] = df_rennes['gps'].str.split(',', expand=True)
    df_rennes['lat'] = df_rennes['lat'].astype(float)
    df_rennes['lon'] = df_rennes['lon'].astype(float)
    df_rennes = df_rennes.dropna(subset=['lat', 'lon'])

    print(f"→ Après nettoyage : {len(df_rennes):,} établissements avec coordonnées valides")

    df_rennes = gpd.GeoDataFrame(
        df_rennes,
        geometry=gpd.points_from_xy(df_rennes.lon, df_rennes.lat),
        crs="EPSG:4326"
    )
    print("✅ Conversion en GeoDataFrame réussie.")
    return df_rennes


# ----------------------------------------------------------------------
# 3. Chargement des IRIS
# ----------------------------------------------------------------------
def charger_iris():
    """Charge les contours IRIS et les références de noms/communes."""
    print("\n[3/6] 🗺️ Chargement des contours IRIS et références...")
    url_iris = (
        "https://huggingface.co/datasets/analysedonneesfoncieresdata/"
        "analyse_fonciere_data/resolve/main/contours-iris-pe.gpkg"
    )
    local_path = "contours-iris-pe.gpkg"

    if not os.path.exists(local_path):
        print("→ Téléchargement du fichier IRIS...")
        r = requests.get(url_iris)
        with open(local_path, "wb") as f:
            f.write(r.content)
        print("✅ Fichier téléchargé et enregistré localement.")
    else:
        print("✅ Fichier IRIS déjà disponible en local.")

    layers = listlayers(local_path)
    iris = gpd.read_file(local_path, layer=layers[0])

    url_ref_iris = (
        "https://huggingface.co/datasets/analysedonneesfoncieresdata/"
        "analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"
    )
    iris_noms = pd.read_excel(url_ref_iris).rename(columns={'CODE_IRIS': 'code_iris'})
    iris = iris.merge(iris_noms[['code_iris', 'LIB_IRIS', 'LIBCOM']], on='code_iris', how='left')

    iris_rennes = iris[iris['nom_commune'].str.contains("Rennes", case=False, na=False)].copy()
    iris_rennes = iris_rennes.to_crs(epsg=4326)
    print(f"✅ {len(iris_rennes):,} IRIS trouvés pour la commune de Rennes.")
    return iris_rennes


# ----------------------------------------------------------------------
# 4. Jointure spatiale et statistiques
# ----------------------------------------------------------------------
def calculer_stats(df_rennes, iris_rennes):
    """Associe les établissements aux IRIS et calcule les indicateurs clés."""
    print("\n[4/6] 📊 Calcul des statistiques par IRIS...")
    df_rennes = df_rennes.to_crs(epsg=4326)

    etabs_par_iris = gpd.sjoin(df_rennes, iris_rennes, how="inner", predicate="within")
    print(f"→ Établissements rattachés à un IRIS : {len(etabs_par_iris):,}")

    iris_stats = iris_rennes.copy()

    iris_stats = iris_stats.merge(
        etabs_par_iris.groupby('LIB_IRIS').size().reset_index(name='nb_etabs'),
        on='LIB_IRIS', how='left'
    ).fillna({'nb_etabs': 0})

    iris_stats = iris_stats.merge(
        etabs_par_iris.groupby('LIB_IRIS')[
            'nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE'
        ].sum().reset_index(name='nb_etudiants'),
        on='LIB_IRIS', how='left'
    ).fillna({'nb_etudiants': 0})

    iris_stats['nb_etabs'] = iris_stats['nb_etabs'].astype(int)
    iris_stats['nb_etudiants'] = iris_stats['nb_etudiants'].astype(int)

    iris_stats = iris_stats.to_crs(epsg=2154)
    iris_stats['area_m2'] = iris_stats.geometry.area
    iris_stats['area_km2'] = iris_stats['area_m2'] / 1_000_000
    iris_stats['etabs_per_km2'] = iris_stats['nb_etabs'] / iris_stats['area_km2']
    iris_stats['students_per_km2'] = iris_stats['nb_etudiants'] / iris_stats['area_km2']

    iris_plot = iris_stats.to_crs(epsg=4326)
    print("✅ Statistiques calculées avec succès.")
    return iris_plot, etabs_par_iris


# ----------------------------------------------------------------------
# 5. Visualisation Plotly
# ----------------------------------------------------------------------
def generer_carte_interactive(iris_plot, df_rennes, etabs_par_iris):
    """Crée une carte interactive montrant la concentration étudiante à Rennes."""
    print("\n[5/6] 🗺️ Génération de la carte interactive...")

    colorscale = "YlOrRd"
    fig = px.choropleth_mapbox(
        iris_plot,
        geojson=iris_plot.__geo_interface__,
        locations=iris_plot.index,
        color='students_per_km2',
        hover_name='LIB_IRIS',
        hover_data=['nb_etabs', 'area_km2', 'nb_etudiants'],
        mapbox_style="carto-positron",
        center={"lat": 48.117, "lon": -1.677},
        zoom=12,
        opacity=0.6,
    )

    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=iris_plot['students_per_km2'].min(),
            cmax=iris_plot['students_per_km2'].max(),
            colorbar=dict(title="Étudiants/km²")
        )
    )

    fig.add_scattermapbox(
        lat=df_rennes['lat'],
        lon=df_rennes['lon'],
        mode='markers',
        marker=dict(size=6, color='blue'),
        text=etabs_par_iris["libellé de l'établissement"],
        name='Établissements supérieurs'
    )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                x=0.0, y=1.05, showactive=True,
                buttons=[
                    dict(
                        label="Densité (étudiants/km²)",
                        method="update",
                        args=[
                            {"z": [iris_plot['students_per_km2']]},
                            {"coloraxis.colorbar.title": "Étudiants/km²"}
                        ]
                    ),
                    dict(
                        label="Nombre d’étudiants",
                        method="update",
                        args=[
                            {"z": [iris_plot['nb_etudiants']]},
                            {"coloraxis.colorbar.title": "Nombre d’étudiants"}
                        ]
                    ),
                ]
            )
        ],
        margin={"r":0, "t":75, "l":0, "b":0},
        title="Concentration étudiante par IRIS – Rennes"
    )

    print("✅ Carte générée avec succès.")
    return fig


# ----------------------------------------------------------------------
# 6. Fonction principale (pipeline complet)
# ----------------------------------------------------------------------
def analyse_concentration_rennes():
    """Pipeline complet : charge les données, calcule les stats et retourne la carte."""
    print("🚀 DÉBUT DE L’ANALYSE : concentration étudiante à Rennes")
    df = charger_donnees_enseignement_sup()
    df_rennes = filtrer_donnees_rennes(df)
    iris_rennes = charger_iris()
    iris_plot, etabs_par_iris = calculer_stats(df_rennes, iris_rennes)
    fig = generer_carte_interactive(iris_plot, df_rennes, etabs_par_iris)
    print("\n✅ ANALYSE TERMINÉE AVEC SUCCÈS ✅")
    return fig, iris_plot, df_rennes
