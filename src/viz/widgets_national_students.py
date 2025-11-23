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
from src.analyses.national_student_analysis import analyze_national_student_density

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
        title = f'Top {top_n} cities - Student density (cities + 100k inhabitants)'
        ylabel = 'Students/population'
        data = data_df.nlargest(top_n, 'student_density')
        values = data['student_density']
        labels = data['libgeo']
    elif rank_by == 'total_students':
        title = f'Top {top_n} cities - Number of students (cities + 100k inhabitants)'
        ylabel = 'Number of students'
        data = data_df.nlargest(top_n, 'nb_etudiants')
        values = data['nb_etudiants']
        labels = data['libgeo']
    elif rank_by == 'population':
        title = f'Top {top_n} cities - Population (cities + 100k inhabitants)'
        ylabel = 'Population'
        data = data_df.nlargest(top_n, 'p21_pop')
        values = data['p21_pop']
        labels = data['libgeo']
    else:
        raise ValueError(f"Unknown rank_by option: {rank_by}")

    # Create vertical bar chart
    bars = ax.bar(range(len(labels)), values)

    # Color bars based on density
    if rank_by != 'student_density':
        # Color by density for context
        densities = data['student_density']
        norm = plt.Normalize(densities.min(), densities.max())
        colors = plt.cm.YlOrRd(norm(densities))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Cities')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add mean and median lines
    if rank_by == 'student_density':
        col = 'student_density'
        fmt = '.3f'
        mean_label = f'Mean: {data_df[col].mean():{fmt}}'
        median_label = f'Median: {data_df[col].median():{fmt}}'
    elif rank_by == 'total_students':
        col = 'nb_etudiants'
        fmt = ',.0f'
        mean_label = f'Mean: {data_df[col].mean():{fmt}}'
        median_label = f'Median: {data_df[col].median():{fmt}}'
    elif rank_by == 'population':
        col = 'p21_pop'
        fmt = ',.0f'
        mean_label = f'Mean: {data_df[col].mean():{fmt}}'
        median_label = f'Median: {data_df[col].median():{fmt}}'

    mean_val = data_df[col].mean()
    median_val = data_df[col].median()
    ax.axhline(mean_val, color='blue', linestyle='--', alpha=0.8, label=mean_label)
    ax.axhline(median_val, color='green', linestyle=':', alpha=0.8, label=median_label)
    ax.legend()
    ax.grid(False)

    # Add value labels
    for i in range(len(values)):
        if rank_by == 'student_density':
            ax.text(i, values.iloc[i] + 0.0001, f'{values.iloc[i]:.3f}', va='bottom', ha='center')
        else:
            ax.text(i, values.iloc[i] + max(values) * 0.01, f'{values.iloc[i]:,.0f}', va='bottom', ha='center')

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
    fig = px.scatter_map(
        map_data,
        lat="latitude",
        lon="longitude",
        hover_name="libgeo",
        hover_data=[
            'nb_etudiants',
            'student_density',
            'p21_pop'
        ],
        size="student_density",
        zoom=5,
        height=600
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(title="Geographical distribution of students in France")

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
        options=['Bar Chart Rankings', 'Interactive Plotly Map'],
        value='Bar Chart Rankings',
        description='Chart type:',
        style={'description_width': 'initial'}
    )

    # Ranking selector (for bar charts)
    ranking_type = widgets.Dropdown(
        options=['student_density', 'total_students', 'population'],
        value='student_density',
        description='Rank by:',
        style={'description_width': 'initial'}
    )

    # Top N slider
    top_n_slider = widgets.IntSlider(
        value=15,
        min=5,
        max=30,
        step=1,
        description='Top N cities:',
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    def update_plot(plot_type, ranking, top_n):
        plt.close('all')  # Clear previous plots

        if plot_type == 'Bar Chart Rankings':
            fig, ax = create_ranking_bar_chart(data_df, ranking, top_n)
        elif plot_type == 'Interactive Plotly Map':
            fig = create_plotly_scatter_map(data_df)
            fig.show()
            return
        else:
            print(f"Unknown plot type: {plot_type}")
            return

        plt.show()

    # Create controls container and output
    controls_container = VBox()
    plot_output = widgets.Output()

    # Update button for manual trigger
    update_button = widgets.Button(description='Display/Update')

    def update_plot_manual(clicked):
        with plot_output:
            plt.close('all')
            plot_output.clear_output(True)
            if plot_type.value == 'Bar Chart Rankings':
                fig, ax = create_ranking_bar_chart(data_df, ranking_type.value, top_n_slider.value)
                plt.show()
            elif plot_type.value == 'Interactive Plotly Map':
                fig = create_plotly_scatter_map(data_df)
                fig.show()

    update_button.on_click(update_plot_manual)

    # Function to update controls
    def update_controls():
        if plot_type.value == 'Bar Chart Rankings':
            controls_container.children = [plot_type, ranking_type, top_n_slider, update_button]
        else:
            controls_container.children = [plot_type, update_button]

    # Initial controls setup
    update_controls()

    # Observe changes to plot_type
    plot_type.observe(lambda change: update_controls(), names='value')

    # Display UI
    display(VBox([controls_container, plot_output]))

    print("✅ Interactive national student analysis widgets ready!")


# Convenience function for quick demonstration
def demo_national_student_analysis():
    """
    Demo function to quickly show the national student analysis with widgets.
    """
    print("🇫🇷 National Student Analysis Demo")  # Keep flag for France, but translate text?
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
    ax.set_xlabel('City population')
    ax.set_ylabel('Student density')
    ax.set_title('Population vs Student Density Correlation')

    # Format x-axis with commas
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1%}'))

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Color intensity =\ndensity')

    # Add city labels for outliers
    outliers = data_df.nlargest(3, 'student_density')
    for idx, row in outliers.iterrows():
        ax.annotate(row['libgeo'],
                   (row['p21_pop'], row['student_density']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    plt.tight_layout()
    return fig, ax
