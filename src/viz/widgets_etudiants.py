"""
Student Concentration Widgets Module
===================================

Interactive widgets for visualizing student concentration analysis in Rennes.
Provides filtering and dynamic plotting capabilities.

Author: Adam Jouini
"""

import ipywidgets as widgets
from ipywidgets import VBox, interactive
from IPython.display import display
import plotly.express as px
import pandas as pd
import geopandas as gpd

# Import our modular analysis functions
from src.loaders import load_iris_rennes, load_education_rennes
from src.analyses.analyse_etudiants import analyze_student_concentration

# Import plotting defaults
from src.config import DEFAULT_FIG_SIZE


def load_analysis_data():
    """
    Load and prepare analysis data for student concentration widgets.

    Returns:
        tuple: (iris_gdf, education_gdf, analysis_results)
    """
    print("🔄 Loading IRIS and education data for Rennes...")
    iris_gdf = load_iris_rennes()
    education_gdf = load_education_rennes()

    print("🔄 Running student concentration analysis...")
    results = analyze_student_concentration(iris_gdf, education_gdf)

    print("✅ Data and analysis ready for interactive visualization.")
    return iris_gdf, education_gdf, results


def create_interactive_student_plot(stats_gdf, mode="Densité", top_n=10):
    """
    Create an interactive Plotly bar chart for student concentration data.

    Args:
        stats_gdf (GeoDataFrame): Student analysis statistics
        mode (str): 'Densité' or 'Nombre d\'étudiants'
        top_n (int): Number of top IRIS to display
    """

    colorscale = "YlOrRd"

    if mode == "Densité":
        data = stats_gdf.sort_values(by='students_per_km2', ascending=False).head(top_n)
        y_col = "students_per_km2"
        title = f"Top {top_n} IRIS les plus denses en étudiants - Rennes"
        color_title = "Étudiants/km²"
    else:
        data = stats_gdf.sort_values(by='nb_etudiants', ascending=False).head(top_n)
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
        margin=dict(t=60, l=50, r=50, b=50),
        width=DEFAULT_FIG_SIZE[0] * 50,  # Scale for Plotly
        height=DEFAULT_FIG_SIZE[1] * 50,
        xaxis_tickangle=45
    )

    fig.show()


def create_interactive_comparison_plot(stats_gdf, metric1="students_per_km2", metric2="nb_etabs", top_n=15):
    """
    Create a dual-axis plot comparing two metrics.

    Args:
        stats_gdf (GeoDataFrame): Student analysis statistics
        metric1 (str): Primary metric for bars
        metric2 (str): Secondary metric for line
        top_n (int): Number of top IRIS to display
    """

    data = stats_gdf.sort_values(by=metric1, ascending=False).head(top_n)

    fig = px.bar(
        data,
        x='LIB_IRIS',
        y=metric1,
        title=f'Comparison: {metric1} vs {metric2} - Top {top_n} IRIS',
        labels={metric1: metric1.replace('_', ' ').title()}
    )

    # Add secondary metric as line
    fig.add_scatter(
        x=data['LIB_IRIS'],
        y=data[metric2],
        mode='lines+markers',
        name=metric2.replace('_', ' ').title(),
        yaxis='y2',
        line=dict(color='red', width=3)
    )

    fig.update_layout(
        xaxis_title="Quartier (IRIS)",
        yaxis=dict(title=metric1.replace('_', ' ').title()),
        yaxis2=dict(title=metric2.replace('_', ' ').title(), overlaying='y', side='right'),
        margin=dict(t=80, l=50, r=80, b=50),
        width=DEFAULT_FIG_SIZE[0] * 60,
        height=DEFAULT_FIG_SIZE[1] * 60,
        xaxis_tickangle=45
    )

    fig.show()


def create_interactive_student_map(iris_gdf, education_gdf, stats_gdf, color_metric="students_per_km2"):
    """
    Create an interactive choropleth map of student concentration.

    Args:
        iris_gdf (GeoDataFrame): IRIS polygons in WGS84
        education_gdf (GeoDataFrame): Education establishments
        stats_gdf (GeoDataFrame): Student analysis statistics
        color_metric (str): 'students_per_km2' or 'nb_etudiants'
    """

    colorscale = "YlOrRd"

    # Set appropriate titles and labels based on metric
    if color_metric == "students_per_km2":
        color_title = "Étudiants/km²"
        map_title = "Concentration étudiante par IRIS – Rennes"
        hover_template = (
            "<b>%{hovertext}</b><br>" +
            "Étudiants/km²: %{z:.0f}<br>" +
            "Nombre d'étudiants: %{customdata[0]:,}<br>" +
            "Établissements: %{customdata[1]}<br>" +
            "Superficie: %{customdata[2]:.2f} km²<extra></extra>"
        )
    else:
        color_title = "Nombre d'étudiants"
        map_title = "Nombre total d'étudiants par IRIS – Rennes"
        hover_template = (
            "<b>%{hovertext}</b><br>" +
            "Nombre d'étudiants: %{z:,}<br>" +
            "Étudiants/km²: %{customdata[0]:.0f}<br>" +
            "Établissements: %{customdata[1]}<br>" +
            "Superficie: %{customdata[2]:.2f} km²<extra></extra>"
        )

    # Prepare map data
    iris_for_map = iris_gdf.to_crs("EPSG:4326").merge(
        stats_gdf[['code_iris', 'nb_etudiants', 'students_per_km2', 'nb_etabs', 'area_km2']],
        on='code_iris',
        how='left'
    ).fillna(0)

    education_for_map = education_gdf.to_crs("EPSG:4326")

    # Join establishments to IRIS for tooltips
    establishments_with_iris = gpd.sjoin(
        education_for_map,
        iris_for_map[['code_iris', 'LIB_IRIS', 'geometry']],
        how='inner',
        predicate='within'
    )

    # Create the choropleth map
    fig = px.choropleth_mapbox(
        iris_for_map,
        geojson=iris_for_map.__geo_interface__,
        locations=iris_for_map.index,
        color=color_metric,
        hover_name='LIB_IRIS',
        hover_data=['nb_etabs', 'area_km2', 'students_per_km2'],
        mapbox_style="carto-positron",
        center={"lat": 48.117, "lon": -1.677},
        zoom=12,
        opacity=0.6,
    )

    # Update color scale and title
    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=iris_for_map[color_metric].min(),
            cmax=iris_for_map[color_metric].max(),
            colorbar=dict(title=color_title)
        )
    )

    # Add establishment markers
    fig.add_scattermapbox(
        lat=education_for_map['lat'],
        lon=education_for_map['lon'],
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            opacity=0.8
        ),
        text=establishments_with_iris.get("libellé de l'établissement", "Établissement"),
        name='Établissements supérieurs',
        hovertemplate="<b>%{text}</b><extra></extra>"
    )

    # Update hover template
    fig.update_traces(
        hovertemplate=hover_template,
        selector=dict(type='choroplethmapbox')
    )

    # Set overall layout
    fig.update_layout(
        title={
            'text': map_title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        mapbox=dict(
            style="carto-positron",
            center={"lat": 48.117, "lon": -1.677},
            zoom=12
        )
    )

    return fig


def display_student_widgets(analysis_results=None, iris_gdf=None, education_gdf=None):
    """
    Create and display the interactive student concentration widgets
    with both charts and interactive map in one interface.

    Args:
        analysis_results (dict, optional): Pre-computed analysis results
        iris_gdf (GeoDataFrame, optional): IRIS polygons
        education_gdf (GeoDataFrame, optional): Education establishments
    """
    # Load data if not provided
    if any(param is None for param in [analysis_results, iris_gdf, education_gdf]):
        print("🔄 No datasets provided → loading via analysis pipeline...")
        iris_gdf, education_gdf, analysis_results = load_analysis_data()
    else:
        print("⚡ Using provided datasets.")

    print("🧩 Creating combined interactive student concentration widgets...")

    # Extract the stats dataframe for plotting
    stats_gdf = analysis_results['stats']

    # Controls for charts and map
    plot_type = widgets.Dropdown(
        options=['Bar Chart', 'Comparison Dual', 'Carte interactive Plotly'],
        value='Bar Chart',
        description='Type de graphique :',
        style={'description_width': 'initial'}
    )

    mode_selector = widgets.ToggleButtons(
        options=['Densité', 'Nombre d\'étudiants'],
        description='Afficher :',
        button_style='info',
        value='Densité',
        style={'description_width': 'initial'}
    )

    top_slider = widgets.IntSlider(
        value=15,
        min=5,
        max=30,
        step=1,
        description='Top N IRIS :',
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    metric1_selector = widgets.Dropdown(
        options=['students_per_km2', 'nb_etudiants', 'nb_etabs', 'etabs_per_km2'],
        value='nb_etudiants',
        description='Métrique 1 :',
        style={'description_width': 'initial'}
    )

    metric2_selector = widgets.Dropdown(
        options=['students_per_km2', 'nb_etudiants', 'nb_etabs', 'etabs_per_km2'],
        value='etabs_per_km2',
        description='Métrique 2 :',
        style={'description_width': 'initial'}
    )

    zoom_slider = widgets.IntSlider(
        value=12,
        min=10,
        max=15,
        step=1,
        description='Zoom :',
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    # Create controls containers
    charts_controls = VBox()
    map_controls = VBox()
    plot_output = widgets.Output()

    def update_plot():
        with plot_output:
            plot_output.clear_output(True)

            if plot_type.value == 'Bar Chart':
                create_interactive_student_plot(stats_gdf, mode_selector.value, top_slider.value)
            elif plot_type.value == 'Comparison Dual':
                create_interactive_comparison_plot(stats_gdf, metric1_selector.value, metric2_selector.value, top_slider.value)
            elif plot_type.value == 'Carte interactive Plotly':
                color_metric = 'students_per_km2' if mode_selector.value == 'Densité' else 'nb_etudiants'
                fig = create_interactive_student_map(iris_gdf, education_gdf, stats_gdf, color_metric)
                fig.update_layout(
                    mapbox=dict(
                        zoom=zoom_slider.value,
                        center={"lat": 48.117, "lon": -1.677}
                    )
                )
                fig.show()

    # Update button for manual trigger
    update_button = widgets.Button(description='Afficher/Mettre à jour')
    update_button.on_click(lambda clicked: update_plot())

    def update_controls():
        if plot_type.value == 'Bar Chart':
            charts_controls.children = [mode_selector, top_slider, update_button]
            display(VBox([plot_type, charts_controls, plot_output]))
        elif plot_type.value == 'Comparison Dual':
            charts_controls.children = [top_slider, metric1_selector, metric2_selector, update_button]
            display(VBox([plot_type, charts_controls, plot_output]))
        elif plot_type.value == 'Carte interactive Plotly':
            map_controls.children = [mode_selector, zoom_slider, update_button]
            display(VBox([plot_type, map_controls, plot_output]))

    # Observe changes
    plot_type.observe(lambda change: update_controls(), names='value')

    # Initial display
    update_controls()

    print("✅ Combined interactive student concentration widgets ready!")


# Convenience function for quick demonstration
def demo_student_analysis():
    """
    Demo function to quickly show the student concentration analysis with widgets.
    """
    print("🎓 Student Concentration Analysis Demo")
    print("This will load data and create interactive visualizations.")

    # Create sample analysis
    iris_gdf, education_gdf, results = load_analysis_data()

    print(f"\n📊 Found {results['summary'].iloc[0]['total_students']:,} students across {results['summary'].iloc[0]['total_establishments']} establishments")

    # Show summary
    print("\n📈 Analysis Summary:")
    display(results['summary'])

    # Launch interactive widgets
    display_student_widgets(results, iris_gdf, education_gdf)

    return results
