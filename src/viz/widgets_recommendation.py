"""
Investment Recommendation Widgets Module
========================================

Interactive 3D visualization widgets for Rennes IRIS investment recommendations.
Combines student density, commercial establishments, and transportation data
for data-driven investment decisions.

Author: Adam (modularized as widget)
"""

import ipywidgets as widgets
from ipywidgets import VBox, HBox, interactive
from IPython.display import display, clear_output
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import requests
import os
import warnings
from pathlib import Path
from fiona import listlayers

warnings.filterwarnings('ignore')

# Get project root directory
script_dir = Path(__file__).parent.parent  # src/viz -> src
project_root = script_dir.parent  # src -> project root


def load_final_recommendation_data():
    """
    Load data for the final investment recommendation analysis.

    Returns:
        tuple: (iris_gdf, commerces_gdf, etudiants_gdf, transports_gdf, merged_gdf)
    """
    print("🔄 Loading final recommendation analysis data...")

    # Import modular functions where available, fallback to original
    try:
        from src.analyses.analyse_commerces import load_commerce_data
        commerces_gdf = load_commerce_data()
        print("✅ Using modular commerce data")
    except ImportError:
        # Inline fallback for commerce loading
        def recup_commerces():
            overpass_url = "http://overpass-api.de/api/interpreter"
            query = """
            [out:json][timeout:90];
            area["name"="Rennes"]["admin_level"="8"]->.searchArea;
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
            r = requests.post(overpass_url, data={"data": query}, timeout=120)
            osm_data = r.json()
            commerces_list=[]
            for e in osm_data['elements']:
                tags = e.get('tags', {})
                if e['type']=='node':
                    lat, lon = e['lat'], e['lon']
                elif 'center' in e:
                    lat, lon = e['center']['lat'], e['center']['lon']
                else:
                    continue
                if tags.get('amenity','') in ['restaurant','fast_food']:
                    cat='Restaurant'
                elif tags.get('amenity','') in ['bar','pub','cafe']:
                    cat='Bar/Café'
                elif tags.get('shop','') in ['supermarket','convenience','grocery']:
                    cat='Supermarché'
                else:
                    cat='Autre'
                commerces_list.append({'categorie':cat,'lat':lat,'lon':lon})
            commerces_gdf_local = gpd.GeoDataFrame(commerces_list, geometry=gpd.points_from_xy([c['lon'] for c in commerces_list],[c['lat'] for c in commerces_list]), crs="EPSG:4326")
            return commerces_gdf_local

        commerces_gdf = recup_commerces()
        print("⚠️ Using fallback commerce data")

    try:
        from src.loaders import load_iris_rennes
        iris_gdf = load_iris_rennes()
        print("✅ Using modular IRIS data")
    except ImportError:
        # Inline fallback for iris loading
        def charger_iris(commune="Rennes"):
            gpkg_path = "contours-iris-pe.gpkg"
            url_gpkg = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/contours-iris-pe.gpkg"
            url_ref = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"

            if not os.path.exists(gpkg_path):
                r = requests.get(url_gpkg)
                with open(gpkg_path, "wb") as f:
                    f.write(r.content)

            layers = listlayers(gpkg_path)
            iris_start = gpd.read_file(gpkg_path, layer=layers[0])
            iris_noms = pd.read_excel(url_ref).rename(columns={'CODE_IRIS':'code_iris'})
            iris_rennes = iris_start[iris_start['nom_commune'].str.contains(commune, case=False, na=False)].copy()
            iris_rennes = iris_rennes.merge(iris_noms[['code_iris','LIB_IRIS','LIBCOM']], on='code_iris', how='left')
            return iris_rennes

        iris_gdf = charger_iris()
        print("⚠️ Using fallback IRIS data")

    # Inline the necessary functions from the old file
    def recup_etudiants():
        """Load student data for Rennes."""
        url = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"
        df = pd.read_csv(url, delimiter=';')
        df = df[df['Commune']=='Rennes'].dropna(subset=['gps'])
        df[['lat','lon']] = df['gps'].str.split(',',expand=True).astype(float)

        # Standardize column names (find the student column and rename to 'nb_etudiants')
        student_cols = [col for col in df.columns if 'étudiants' in col.lower()]
        if student_cols:
            df = df.rename(columns={student_cols[0]: "nb_etudiants"})

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        return gdf

    def recup_transports():
        """Load transportation stops data for Rennes."""
        overpass_url="http://overpass-api.de/api/interpreter"
        query="""
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
        r=requests.post(overpass_url, data={"data":query}, timeout=120)
        osm=r.json()
        lst=[]
        for e in osm['elements']:
            if e['type']=='node':
                lat, lon = e['lat'], e['lon']
            elif 'center' in e:
                lat, lon = e['center']['lat'], e['center']['lon']
            else:
                continue
            lst.append({'lat':lat,'lon':lon})
        gdf = gpd.GeoDataFrame(lst, geometry=gpd.points_from_xy([x['lon'] for x in lst],[x['lat'] for x in lst]), crs="EPSG:4326")
        return gdf

    def calcul_densites(iris_gdf, commerces_gdf, etudiants_gdf, transports_gdf):
        """Calculate density statistics for Rennes IRIS."""
        iris_proj = iris_gdf.to_crs(epsg=2154)

        # Commerces
        commerces_gdf = commerces_gdf.to_crs(iris_gdf.crs)
        c_sjoin = gpd.sjoin(commerces_gdf, iris_gdf[['code_iris','LIB_IRIS','geometry']], how='left', predicate='within')
        stats_com = c_sjoin.groupby('code_iris').size().reset_index(name='total_commerces')
        pivot = c_sjoin.pivot_table(index='code_iris', columns='categorie', aggfunc='size', fill_value=0).reset_index()
        stats_com = stats_com.merge(pivot,on='code_iris',how='left')

        # Étudiants
        etudiants_gdf = etudiants_gdf.to_crs(iris_gdf.crs)
        e_sjoin = gpd.sjoin(etudiants_gdf, iris_gdf[['code_iris','LIB_IRIS','geometry']], how='inner', predicate='within')
        stats_etud = e_sjoin.groupby('code_iris')['nb_etudiants'].sum().reset_index()

        # Transports
        transports_gdf = transports_gdf.to_crs(iris_gdf.crs)
        t_sjoin = gpd.sjoin(transports_gdf, iris_gdf[['code_iris','LIB_IRIS','geometry']], how='left', predicate='intersects')
        stats_trans = t_sjoin.groupby('code_iris').size().reset_index(name='total_arrets')

        # Merge
        merged = iris_proj.merge(stats_com, on='code_iris', how='left').merge(stats_etud, on='code_iris', how='left').merge(stats_trans, on='code_iris', how='left')
        merged = merged.fillna(0)
        merged['score_densite'] = (merged.get('Restaurant',0)*2 + merged.get('Bar/Café',0)*2 + merged.get('Supermarché',0)*1.5)/merged.geometry.area*1e6
        merged['students_per_km2'] = merged['nb_etudiants']/merged.geometry.area*1e6
        merged['densite_arrets_km2'] = merged['total_arrets']/merged.geometry.area*1e6
        return merged

    etudiants_gdf = recup_etudiants()
    transports_gdf = recup_transports()
    merged_gdf = calcul_densites(iris_gdf, commerces_gdf,
                               etudiants_gdf, transports_gdf)

    print("✅ All data loaded and processed for recommendation analysis")
    return iris_gdf, commerces_gdf, etudiants_gdf, transports_gdf, merged_gdf


def create_3d_investment_plot(merged_gdf,
                            color_by="students_per_km2",
                            marker_size=5,
                            show_trends=True):
    """
    Create an interactive 3D scatter plot for investment recommendations.

    Parameters:
    -----------
    merged_gdf : GeoDataFrame
        Processed data with density calculations
    color_by : str
        Column to color markers by (default: students_per_km2)
    marker_size : int
        Size of scatter markers
    show_trends : bool
        Whether to show trend lines

    Returns:
    --------
    plotly Figure : Interactive 3D plot
    """

    # Create hover text
    merged_gdf['hover_text'] = (
        "<b>IRIS:</b> " + merged_gdf['LIB_IRIS'].astype(str) + "<br>" +
        "<b>Étudiants/km²:</b> " + merged_gdf['students_per_km2'].round(1).astype(str) + "<br>" +
        "<b>Commerces/km²:</b> " + merged_gdf['score_densite'].round(1).astype(str) + "<br>" +
        "<b>Transports/km²:</b> " + merged_gdf['densite_arrets_km2'].round(1).astype(str) + "<br>" +
        "<b>Total Étudiants:</b> " + merged_gdf['nb_etudiants'].astype(int).astype(str) + "<br>" +
        "<b>Total Commerces:</b> " + merged_gdf['total_commerces'].astype(int).astype(str)
    )

    # Set color scale and title
    color_scales = {
        "students_per_km2": ("Viridis", "Étudiants/km²"),
        "score_densite": ("Plasma", "Score densité commerces"),
        "densite_arrets_km2": ("Cividis", "Transports/km²"),
        "nb_etudiants": ("Blues", "Nombre d'étudiants")
    }

    colorscale, colorbar_title = color_scales.get(color_by, ("Viridis", color_by))

    fig = go.Figure()

    # Add 3D scatter plot
    fig.add_trace(go.Scatter3d(
        x=merged_gdf['students_per_km2'],
        y=merged_gdf['score_densite'],
        z=merged_gdf['densite_arrets_km2'],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=merged_gdf[color_by],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=colorbar_title)
        ),
        text=merged_gdf['hover_text'],
        hoverinfo='text',
        name='IRIS Sectors'
    ))

    # Add trend lines if requested
    if show_trends:
        # Calculate trends (simplified)
        x_trend = [merged_gdf['students_per_km2'].min(), merged_gdf['students_per_km2'].max()]
        y_trend = [merged_gdf['score_densite'].mean()] * 2
        z_trend = [merged_gdf['densite_arrets_km2'].mean()] * 2

        fig.add_trace(go.Scatter3d(
            x=x_trend, y=y_trend, z=z_trend,
            mode='lines',
            line=dict(color='red', width=4),
            name='Average Trend Line',
            hoverinfo='skip'
        ))

    # Update layout
    fig.update_layout(
        title="🧭 Carte d'Investissement Étudiant - Rennes IRIS",
        scene=dict(
            xaxis_title='Étudiants/km²',
            yaxis_title='Densité Commerciale',
            zaxis_title='Densité Transports/km²',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        height=700
    )

    return fig


def create_investment_recommendations(merged_gdf, top_n=5):
    """
    Generate investment recommendations based on multi-criteria analysis.

    Parameters:
    -----------
    merged_gdf : GeoDataFrame
        Processed data with density calculations
    top_n : int
        Number of top recommendations

    Returns:
    --------
    DataFrame : Top investment recommendations
    """

    # Calculate investment score (weighted combination)
    merged_gdf['investment_score'] = (
        merged_gdf['students_per_km2'] * 0.4 +  # 40% weight on students
        merged_gdf['score_densite'] * 0.3 +     # 30% weight on commerce
        merged_gdf['densite_arrets_km2'] * 0.3   # 30% weight on transport
    )

    # Get top recommendations
    top_investments = merged_gdf.nlargest(top_n, 'investment_score')[[
        'LIB_IRIS', 'code_iris', 'students_per_km2', 'score_densite',
        'densite_arrets_km2', 'nb_etudiants', 'investment_score'
    ]].round(2)

    # Rename columns for display
    top_investments = top_investments.rename(columns={
        'LIB_IRIS': 'Quartier',
        'code_iris': 'Code IRIS',
        'students_per_km2': 'Étudiants/km²',
        'score_densite': 'Commerces/km²',
        'densite_arrets_km2': 'Transports/km²',
        'nb_etudiants': 'Total Étudiants',
        'investment_score': 'Score Investissement'
    })

    return top_investments


def display_investment_analysis_summary(merged_gdf):
    """
    Display summary statistics for investment analysis.

    Parameters:
    -----------
    merged_gdf : GeoDataFrame
        Processed data with density calculations
    """

    print("📊 Résumé de l'Analyse d'Investissement - Rennes")
    print("=" * 50)

    total_students = merged_gdf['nb_etudiants'].sum()
    total_commerce = merged_gdf['total_commerces'].sum()
    total_transport = merged_gdf['total_arrets'].sum()

    print(f"• IRIS analysés: {len(merged_gdf)}")
    print(f"• Total étudiants: {total_students:,.0f}")
    print(f"• Établissements commerciaux: {total_commerce}")
    print(f"• Arrêts de transport: {total_transport}")

    print("\n🏆 Top 3 Secteurs pour l'Investissement Étudiant:")

    top_3 = create_investment_recommendations(merged_gdf, 3)
    display(top_3)


def display_recommendation_widgets(merged_data=None):
    """
    Create and display interactive investment recommendation widgets.

    Parameters:
    -----------
    merged_data : GeoDataFrame, optional
        Pre-processed merged data
    """

    # Load data if not provided
    if merged_data is None:
        print("🔄 Chargement des données d'analyse finale...")
        iris_gdf, commerces_gdf, etudiants_gdf, transports_gdf, merged_data = load_final_recommendation_data()
    else:
        print("⚡ Utilisation des données fournies")

    print("\n🧭 Création de l'interface d'analyse d'investissement interactif...")

    # Create investment recommendations
    recommendations = create_investment_recommendations(merged_data)

    # Widget controls
    color_by = widgets.Dropdown(
        options={
            'Étudiants/km²': 'students_per_km2',
            'Score Commercial': 'score_densite',
            'Densité Transports': 'densite_arrets_km2',
            'Nombre Étudiants': 'nb_etudiants'
        },
        value='students_per_km2',
        description='Coloration :',
        style={'description_width': 'initial'}
    )

    marker_size = widgets.IntSlider(
        value=6,
        min=2,
        max=12,
        step=1,
        description='Taille points :',
        style={'description_width': 'initial'}
    )

    show_trends = widgets.Checkbox(
        value=False,
        description='Lignes tendance',
        style={'description_width': 'initial'}
    )

    view_rotation = widgets.IntSlider(
        value=45,
        min=0,
        max=90,
        step=15,
        description='Rotation (°) :',
        style={'description_width': 'initial'}
    )

    show_recommendations = widgets.ToggleButton(
        value=False,
        description='Afficher Recommandations',
        button_style='info',
        tooltip='Afficher les meilleures opportunités d\'investissement'
    )

    # Output widgets
    plot_output = widgets.Output()
    recommendations_output = widgets.Output()

    def update_plot():
        with plot_output:
            plot_output.clear_output(True)

            fig = create_3d_investment_plot(
                merged_data,
                color_by=color_by.value,
                marker_size=marker_size.value,
                show_trends=show_trends.value
            )

            # Update camera angle
            camera_eye = dict(
                x=1.5 * (90 - view_rotation.value) / 90,
                y=1.5,
                z=1.5 * view_rotation.value / 90
            )
            fig.update_layout(scene_camera=dict(eye=camera_eye))

            fig.show()

    def update_recommendations():
        with recommendations_output:
            recommendations_output.clear_output(True)
            if show_recommendations.value:
                print("🏆 Meilleures Opportunités d'Investissement:")
                display(recommendations.head(10))

    # Update button
    update_button = widgets.Button(
        description='Mettre à jour',
        button_style='primary'
    )
    update_button.on_click(lambda clicked: (update_plot(), update_recommendations()))

    # Auto-update on parameter change
    for widget in [color_by, marker_size, show_trends, view_rotation]:
        widget.observe(lambda change: update_plot() if change['name'] == 'value' else None)

    show_recommendations.observe(lambda change: update_recommendations() if change['name'] == 'value' else None)

    # Layout
    controls_box = VBox([
        HBox([color_by, marker_size]),
        HBox([show_trends, view_rotation]),
        HBox([show_recommendations, update_button])
    ])

    # Initial display
    update_plot()
    update_recommendations()

    display(VBox([
        widgets.HTML("<h3>🧭 Analyse d'Investissement Étudiant - Rennes</h3>"),
        controls_box,
        plot_output,
        recommendations_output
    ]))

    # Summary
    display_investment_analysis_summary(merged_data)

    print("✅ Interface interactive d'analyse d'investissement prête!")


# Convenience function for pipeline execution
def show_investment_recommendations():
    """
    Execute the full investment recommendation pipeline with widgets.
    """
    print("🎯 Recommandations d'Investissement Étudiant - Rennes")
    print("Analyse multi-critères: Étudiants + Commerces + Transports")

    # Execute pipeline
    _, _, _, _, merged_data = load_final_recommendation_data()

    # Show interactive widgets
    display_recommendation_widgets(merged_data)


# Wrapper to match original function signature
def pipeline_3d():
    """
    Wrapper for backward compatibility with original function.
    """
    iris, commerces, etudiants, transports, merged = load_final_recommendation_data()
    print("\n📊 Exécution du pipeline 3D...")
    display_recommendation_widgets(merged)
