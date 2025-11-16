"""
National Student Analysis Widgets Module
=======================================

Interactive widgets for visualizing French student density analysis at national scale.
Includes maps, rankings, and comparative visualizations.

Author: Adam Jouini
"""

import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interactive, VBox, HBox
from IPython.display import display
import pandas as pd
import plotly.express as px
import numpy as np

# Import our modular components
from src.loaders import load_french_cities_100k
from src.analyses.analyse_etudiants_national import analyze_national_student_density

# Import plotting defaults
from src.config import DEFAULT_FIG_SIZE, DEFAULT_DPI


def load_national_analysis_data():
    """
    Load and prepare analysis data for national student density widgets.

    Returns:
        tuple: (geocoded_data, analysis_results)
    """
    print("🌍 Loading national student analysis data...")

    # Load base data
    data_df = load_french_cities_100k()

    # Run complete analysis
    results = analyze_national_student_density(data_df, add_coordinates=True)

    print("✅ National analysis data ready for visualization.")
    return results['data'], results


def create_ranking_bar_chart(data_df, rank_by='student_density', top_n=15, figsize=None):
    """
    Create a bar chart showing city rankings.

    Args:
        data_df (DataFrame): City data with rankings
        rank_by (str): Column to rank by
        top_n (int): Number of top cities to show
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = DEFAULT_FIG_SIZE

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Get top N cities
    if rank_by == 'student_density':
        title = f'Top {top_n} villes - Densité étudiante'
        ylabel = 'Étudiants/population'
        data = data_df.nlargest(top_n, 'student_density')
        values = data['student_density']
        labels = data['libgeo']
    elif rank_by == 'total_students':
        title = f'Top {top_n} villes - Nombre d\'étudiants'
        ylabel = 'Nombre d\'étudiants'
        data = data_df.nlargest(top_n, 'nombre total d\'étudiants inscrits hors doubles inscriptions université/CPGE')
        values = data['nombre total d\'étudiants inscrits hors doubles inscriptions université/CPGE']
        labels = data['libgeo']
    elif rank_by == 'population':
        title = f'Top {top_n} villes - Population'
        ylabel = 'Population'
        data = data_df.nlargest(top_n, 'p21_pop')
        values = data['p21_pop']
        labels = data['libgeo']
    else:
        raise ValueError(f"Unknown rank_by option: {rank_by}")

    # Create horizontal bar chart
    bars = ax.barh(range(len(labels)), values)

    # Color bars based on density
    if rank_by != 'student_density':
        # Color by density for context
        densities = data['student_density']
        norm = plt.Normalize(densities.min(), densities.max())
        colors = plt.cm.YlOrRd(norm(densities))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(ylabel)
    ax.set_title(title)

    # Add value labels
    for i, (idx, row) in enumerate(data.iterrows()):
        if rank_by == 'student_density':
            ax.text(values.iloc[i] + 0.0001, i, f'{values.iloc[i]:.3f}', va='center')
        else:
            ax.text(values.iloc[i] + max(values) * 0.01, i,
                   f'{values.iloc[i]:,.0f}', va='center')

    plt.tight_layout()
    return fig, ax


def create_scatter_map(data_df, figsize=None):
    """
    Create a scatter map of student density across France.

    Args:
        data_df (DataFrame): Geocoded city data
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = (figsize[0] if figsize else DEFAULT_FIG_SIZE[0] * 1.5,
                  figsize[1] if figsize else DEFAULT_FIG_SIZE[1])

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Filter data with valid coordinates
    map_data = data_df.dropna(subset=['latitude', 'longitude']).copy()

    # Color points by student density
    scatter = ax.scatter(
        map_data['longitude'],
        map_data['latitude'],
        c=map_data['student_density'],
        s=map_data['p21_pop'] / 5000,  # Scale point size by population
        cmap='YlOrRd',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Densité étudiante', rotation=270, labelpad=15)

    # Formatting
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Répartition de la densité étudiante en France')
    ax.grid(True, alpha=0.3)

    # Add text for major cities
    major_cities = map_data.nlargest(5, 'student_density')
    for idx, row in major_cities.iterrows():
        ax.annotate(row['libgeo'],
                   (row['longitude'], row['latitude']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    plt.tight_layout()
    return fig, ax


def create_distribution_histogram(data_df, bins=20, figsize=None):
    """
    Create a histogram of student density distribution.

    Args:
        data_df (DataFrame): City data with student density
        bins (int): Number of histogram bins
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = DEFAULT_FIG_SIZE

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Create histogram
    counts, bins, patches = ax.hist(data_df['student_density'], bins=bins,
                                  alpha=0.7, color='#FF6B6B', edgecolor='black')

    ax.set_xlabel('Densité étudiante')
    ax.set_ylabel('Nombre de villes')
    ax.set_title('Distribution de la densité étudiante en France')

    # Add statistics text
    mean_density = data_df['student_density'].mean()
    median_density = data_df['student_density'].median()

    ax.axvline(mean_density, color='blue', linestyle='--', alpha=0.8,
              label=f'Moyenne: {mean_density:.3f}')
    ax.axvline(median_density, color='green', linestyle=':', alpha=0.8,
              label=f'Médiane: {median_density:.3f}')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def create_plotly_scatter_map(data_df):
    """
    Create an interactive Plotly scatter map.

    Args:
        data_df (DataFrame): Geocoded city data

    Returns:
        plotly.graph_objects.Figure: Interactive map
    """
    # Filter for cities with coordinates
    map_data = data_df.dropna(subset=['latitude', 'longitude']).copy()

    # Create the scatter map
    fig = px.scatter_mapbox(
        map_data,
        lat='latitude',
        lon='longitude',
        color='student_density',
        size='p21_pop',
        hover_name='libgeo',
        hover_data=['student_density', 'p21_pop', 'nb_etudiants'],
        color_continuous_scale='YlOrRd',
        size_max=40,
        zoom=5,
        center={"lat": 46.5, "lon": 2.5},  # Center of France
        title="Répartition géographique des étudiants en France"
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    return fig


def display_national_student_widgets(analysis_results=None, data_df=None):
    """
    Create and display the interactive national student analysis widgets.

    Args:
        analysis_results (dict, optional): Pre-computed analysis results
        data_df (DataFrame, optional): City data with geocoding
    """
    # Load data if not provided
    if analysis_results is None or data_df is None:
        print("🔄 Loading analysis data...")
        data_df, analysis_results = load_national_analysis_data()

    print("🧩 Creating interactive national student analysis widgets...")

    # Extract summary stats
    summary = analysis_results['summary']

    # Plot type selector
    plot_type = widgets.Dropdown(
        options=['Bar Chart Rankings', 'Scatter Map', 'Distribution Histogram', 'Interactive Plotly Map'],
        value='Bar Chart Rankings',
        description='Type de graphique :',
        style={'description_width': 'initial'}
    )

    # Ranking selector (for bar charts)
    ranking_type = widgets.Dropdown(
        options=['student_density', 'total_students', 'population'],
        value='student_density',
        description='Classement par :',
        style={'description_width': 'initial'}
    )

    # Top N slider
    top_n_slider = widgets.IntSlider(
        value=15,
        min=5,
        max=30,
        step=1,
        description='Top N villes :',
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    def update_plot(plot_type, ranking, top_n):
        plt.close('all')  # Clear previous plots

        if plot_type == 'Bar Chart Rankings':
            fig, ax = create_ranking_bar_chart(data_df, ranking, top_n)
        elif plot_type == 'Scatter Map':
            fig, ax = create_scatter_map(data_df)
        elif plot_type == 'Distribution Histogram':
            fig, ax = create_distribution_histogram(data_df)
        elif plot_type == 'Interactive Plotly Map':
            fig = create_plotly_scatter_map(data_df)
            fig.show()
            return
        else:
            print(f"Unknown plot type: {plot_type}")
            return

        plt.show()

    # Create interactive widgets
    interactive_plot = interactive(
        update_plot,
        plot_type=plot_type,
        ranking=ranking_type,
        top_n=top_n_slider
    )

    # Display everything
    display(VBox([
        interactive_plot
    ]))

    print("✅ Interactive national student analysis widgets ready!")


# Convenience function for quick demonstration
def demo_national_student_analysis():
    """
    Demo function to quickly show the national student analysis with widgets.
    """
    print("🇫🇷 National Student Analysis Demo")
    print("This will load comprehensive French student data and create interactive visualizations.")

    try:
        # Create sample analysis
        data_df, analysis_results = load_national_analysis_data()

        print(f"\n📈 Analysis Overview:")
        print(f"- Cities analyzed: {analysis_results['summary']['total_cities']:,}")
        print(f"- Total population: {analysis_results['summary']['total_population']:,}")
        print(f"- Total students: {analysis_results['summary']['total_students']:,}")
        print(f"- Average student density: {analysis_results['summary']['avg_density']:.1%}")

        # Top 5 cities by density
        top_density = analysis_results['metrics']['top_cities_density']
        print(f"\n🏆 Top cities by student density:")
        for i, city in enumerate(top_density[:5], 1):
            print(f"{i}. {city['libgeo']}: {city['student_density']:.2%}")

        # Launch interactive widgets
        display_national_student_widgets(analysis_results, data_df)

        return analysis_results

    except Exception as e:
        print(f"❌ Error creating national analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


# Advanced comparison function
def compare_student_vs_population(data_df, figsize=None):
    """
    Create a comparison scatter plot of student density vs population size.

    Args:
        data_df (DataFrame): City data
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = DEFAULT_FIG_SIZE

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Create scatter plot
    scatter = ax.scatter(
        data_df['p21_pop'],
        data_df['student_density'],
        alpha=0.6,
        c=data_df['student_density'],
        cmap='YlOrRd',
        s=data_df['p21_pop'] / 10000,  # Scale point size
        edgecolors='black',
        linewidth=0.5
    )

    # Add trend line
    z = np.polyfit(data_df['p21_pop'], data_df['student_density'], 1)
    p = np.poly1d(z)
    ax.plot(data_df['p21_pop'], p(data_df['p21_pop']), "r--", alpha=0.8)

    # Format axis
    ax.set_xlabel('Population de la ville')
    ax.set_ylabel('Densité étudiante')
    ax.set_title('Corrélation Population vs Densité Étudiante')

    # Format x-axis with commas
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1%}'))

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Intensité\ncouleur =\ndensité')

    # Add city labels for outliers
    outliers = data_df.nlargest(3, 'student_density')
    for idx, row in outliers.iterrows():
        ax.annotate(row['libgeo'],
                   (row['p21_pop'], row['student_density']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    plt.tight_layout()
    return fig, ax
