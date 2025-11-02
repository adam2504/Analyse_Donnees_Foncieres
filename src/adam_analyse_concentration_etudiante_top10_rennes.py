# ============================================================================
# MODULE : widgets_concentration_etudiante_rennes.py
# ============================================================================
# Objectif : visualiser la concentration étudiante à Rennes avec widgets interactifs
# Auteur : Adam Jouini
# ============================================================================

import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display
import plotly.express as px
import pandas as pd
import geopandas as gpd
from src.adam_analyse_concentration_etudiante_carte_rennes import analyse_concentration_rennes


# ============================================================================
# 1. Pipeline d'analyse + récupération des données
# ============================================================================
def charger_donnees_analyse():
    """Charge les résultats d'analyse depuis le premier module."""
    print("🔄 Chargement et préparation des données de concentration étudiante à Rennes...")
    fig, iris_plot, df_rennes = analyse_concentration_rennes()
    print("✅ Données prêtes pour l'affichage interactif.")
    return iris_plot, df_rennes


# ============================================================================
# 2. Génération du graphique interactif
# ============================================================================
def make_fig(iris_rennes_stats, mode="Densité", top_n=10):
    """
    Génère un graphique selon :
      - le mode sélectionné ('Densité' ou 'Nombre d’étudiants')
      - le nombre de top IRIS à afficher
    """
    print(f"📊 Génération du graphique ({mode}, top {top_n})...")

    colorscale = "YlOrRd"

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
        hover_data=['nb_etudiants', 'area_km2', 'nb_etabs'],
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


# ============================================================================
# 3. Création des widgets et affichage
# ============================================================================
def afficher_widgets():
    """Crée et affiche les widgets interactifs basés sur l’analyse Rennes."""
    iris_plot, _ = charger_donnees_analyse()

    print("🧩 Création des widgets interactifs...")

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

    interactive_plot = interactive(
        lambda mode, top_n: make_fig(iris_plot, mode, top_n),
        mode=mode_selector,
        top_n=top_slider
    )

    print("✅ Widgets prêts à l’utilisation !")
    display(interactive_plot)
