"""
Commerce Analysis Widgets Module
================================

Interactive widgets and static plots for visualizing commercial establishments
density and accessibility analysis in Rennes.

Author: Valentine (modularized by Adam Jouini)
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import ipywidgets as widgets
from ipywidgets import interactive, HBox, VBox
from IPython.display import display
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Import our modular analysis functions
from src.loaders import load_iris_rennes
from src.loaders.osm_loader import load_commercial_establishments
from src.analyses.analyse_commerces import analyze_commerce_density

# Import plotting defaults
from src.config import DEFAULT_FIG_SIZE, DEFAULT_DPI


def load_commerce_analysis_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Load and prepare analysis data for commerce widgets.

    Returns:
        tuple: (iris_gdf, commerces_gdf, analysis_results)
    """
    print("🔄 Loading IRIS and commerce data for Rennes...")
    iris_gdf = load_iris_rennes()
    commerces_gdf = load_commercial_establishments()

    print("🔄 Running commerce density analysis...")
    results = analyze_commerce_density(iris_gdf, commerces_gdf)

    print("✅ Commerce data and analysis ready for visualization.")
    return iris_gdf, commerces_gdf, results


def create_top_commerce_count_plot(stats_gdf: gpd.GeoDataFrame, top_n: int = 10, figsize: Optional[Tuple[float, float]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a horizontal bar chart showing top IRIS by commerce count.

    Args:
        stats_gdf (GeoDataFrame): Commerce statistics
        top_n (int): Number of top IRIS to display
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = DEFAULT_FIG_SIZE

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Get top N data sorted ascending for better visualization
    top_data = stats_gdf.nlargest(top_n, 'total_commerces').sort_values('total_commerces', ascending=True)

    # Create horizontal bar chart
    bars = ax.barh(range(len(top_data)), top_data['total_commerces'],
                   color=sns.color_palette("RdYlGn_r", len(top_data)))

    # Customize plot
    ax.set_yticks(range(len(top_data)))
    ax.set_yticklabels(top_data['LIB_IRIS'])
    ax.set_xlabel('Number of businesses')
    ax.set_title(f'Top {top_n} IRIS Rennes - vibrant businesses')

    # Add value labels on bars
    for i, (idx, row) in enumerate(top_data.iterrows()):
        ax.text(row['total_commerces'] + 0.1, i,
                f'{int(row["total_commerces"])}',
                va='center', fontweight='bold')

    plt.tight_layout()
    return fig, ax


def create_commerce_type_distribution_plot(stats_gdf: gpd.GeoDataFrame, top_n: int = 10, figsize: Optional[Tuple[float, float]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grouped bar chart showing commerce types distribution.

    Args:
        stats_gdf (GeoDataFrame): Commerce statistics
        top_n (int): Number of top IRIS to display
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = (DEFAULT_FIG_SIZE[0] * 1.2, DEFAULT_FIG_SIZE[1])

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Get top N by total commerces
    top_data = stats_gdf.nlargest(top_n, 'total_commerces')

    # Set up data for grouped bars
    categories = ['Restaurant', 'Bar/Café', 'Supermarché']
    x = np.arange(len(top_data))
    width = 0.25

    # Create bars for each category
    bars_restaurant = ax.bar(x - width, top_data['Restaurant'], width,
                            label='Restaurants', color='#FF6B6B')
    bars_bar_cafe = ax.bar(x, top_data['Bar/Café'], width,
                          label='Bars/Cafés', color='#4ECDC4')
    bars_super = ax.bar(x + width, top_data['Supermarché'], width,
                       label='Supermarkets', color='#45B7D1')

    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(top_data['LIB_IRIS'], rotation=45, ha='right')
    ax.set_ylabel('Number of businesses')
    ax.set_title(f'Composition of the Top {top_n} IRIS by type of business')
    ax.legend()

    # Add value labels on bars
    def add_value_labels(bars, offset=0):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2.,
                        height + offset, f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')

    add_value_labels(bars_restaurant)
    add_value_labels(bars_bar_cafe)
    add_value_labels(bars_super, 0.5)

    plt.tight_layout()
    return fig, ax


def create_commerce_density_score_plot(stats_gdf, top_n=10, figsize=None):
    """
    Create a plot showing commerce density scores.

    Args:
        stats_gdf (GeoDataFrame): Commerce statistics
        top_n (int): Number of top IRIS to display
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = DEFAULT_FIG_SIZE

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Get top N by density score
    top_data = stats_gdf.nlargest(top_n, 'score_densite').sort_values('score_densite', ascending=True)

    # Create horizontal bar chart
    bars = ax.barh(range(len(top_data)), top_data['score_densite'],
                   color=sns.color_palette("Blues_r", len(top_data)))

    # Customize plot
    ax.set_yticks(range(len(top_data)))
    ax.set_yticklabels(top_data['LIB_IRIS'])
    ax.set_xlabel('Density score (weighted)')
    ax.set_title(f'Top {top_n} IRIS Rennes - commercial density score')

    # Add value labels on bars
    for i, (idx, row) in enumerate(top_data.iterrows()):
        ax.text(row['score_densite'] + 0.01, i,
                f'{row["score_densite"]:.1f}',
                va='center', fontweight='bold')

    plt.tight_layout()
    return fig, ax




def display_commerce_widgets(analysis_results=None, iris_gdf=None, commerces_gdf=None):
    """
    Create and display the interactive commerce analysis widgets.

    Args:
        analysis_results (dict, optional): Pre-computed analysis results
        iris_gdf (GeoDataFrame, optional): IRIS polygons
        commerces_gdf (GeoDataFrame, optional): Commerce establishments
    """
    # Load data if not provided
    if any(param is None for param in [analysis_results, iris_gdf, commerces_gdf]):
        print("🔄 No datasets provided → loading via analysis pipeline...")
        iris_gdf, commerces_gdf, analysis_results = load_commerce_analysis_data()
    else:
        print("⚡ Using provided datasets.")

    print("🧩 Creating interactive commerce widgets...")

    # Extract the stats dataframe for plotting
    stats_gdf = analysis_results['stats']

    # Plot type selector
    plot_type = widgets.Dropdown(
        options=['Commerce Count', 'Type Distribution', 'Density Score'],
        value='Commerce Count',
        description='Chart type:',
        style={'description_width': 'initial'}
    )

    # Top N slider
    top_slider = widgets.IntSlider(
        value=10,
        min=5,
        max=25,
        step=1,
        description='Top N IRIS :',
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    def update_plot(plot_type, top_n):
        plt.close('all')  # Close previous plots

        if plot_type == 'Commerce Count':
            fig, ax = create_top_commerce_count_plot(stats_gdf, top_n)
        elif plot_type == 'Type Distribution':
            fig, ax = create_commerce_type_distribution_plot(stats_gdf, top_n)
        elif plot_type == 'Density Score':
            fig, ax = create_commerce_density_score_plot(stats_gdf, top_n)
        else:
            print(f"Unknown plot type: {plot_type}")
            return

        plt.show()

    # Create the interactive widget
    interactive_plot = interactive(
        update_plot,
        plot_type=plot_type,
        top_n=top_slider
    )

    print("✅ Interactive commerce widgets ready!")
    display(interactive_plot)


def create_commerce_summary_report(analysis_results):
    """
    Create a comprehensive report with multiple commerce visualizations.

    Args:
        analysis_results (dict): Complete analysis results
    """
    print("📊 Generating comprehensive commerce report...")

    stats_gdf = analysis_results['stats']
    summary_df = analysis_results['summary']
    top_by_score = analysis_results['top_by_score']
    top_by_count = analysis_results['top_by_count']

    # Set up the matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rapport d\'Analyse - Commerces Vivants à Rennes',
                fontsize=16, fontweight='bold')

    # 1. Top density score plot (like original bar chart but for score)
    top_score_sorted = top_by_score.sort_values('score_densite', ascending=True)
    axes[0, 0].barh(range(len(top_score_sorted)), top_score_sorted['score_densite'],
                    color=sns.color_palette("RdYlGn_r", len(top_score_sorted)))
    axes[0, 0].set_yticks(range(len(top_score_sorted)))
    axes[0, 0].set_yticklabels(top_score_sorted['LIB_IRIS'], fontsize=8)
    axes[0, 0].set_xlabel('Score de densité')
    axes[0, 0].set_title('Top 10 - Score densité commerciale')

    # 2. Commerce type distribution for top 10 by count
    categories = ['Restaurant', 'Bar/Café', 'Supermarché']
    bottom = np.zeros(len(top_by_count))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, cat in enumerate(categories):
        axes[0, 1].bar(range(len(top_by_count)), top_by_count[cat],
                       bottom=bottom, label=cat, color=colors[i])
        bottom += top_by_count[cat]

    axes[0, 1].set_xticks(range(len(top_by_count)))
    axes[0, 1].set_xticklabels([name[:15] + '...' if len(name) > 15 else name
                                for name in top_by_count['LIB_IRIS']], rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('Nombre de commerces')
    axes[0, 1].set_title('Répartition par type - Top 10')
    axes[0, 1].legend()

    # 3. Accessibility distribution
    accessibility = analysis_results['accessibility']
    accessibility_counts = accessibility['accessibility_score'].value_counts()
    colors_access = {'Very Low': '#d73027', 'Low': '#f46d43', 'Medium': '#fdae61',
                    'High': '#66bd63', 'Very High': '#1a9850'}

    axes[1, 0].pie(accessibility_counts.values,
                   labels=accessibility_counts.index,
                   colors=[colors_access.get(score, '#ccc') for score in accessibility_counts.index],
                   autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Répartition des scores d\'accessibilité')

    # 4. Coverage statistics text
    axes[1, 1].axis('off')
    summary_text = ".2f" ".2f" ".2f"
    axes[1, 1].text(0.1, 0.9, "Statistiques Clés:", fontsize=12, fontweight='bold',
                   transform=axes[1, 1].transAxes, verticalalignment='top')
    axes[1, 1].text(0.1, 0.7, summary_text, fontsize=10,
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))

    plt.tight_layout()
    return fig, axes


# Convenience function for quick demonstration
def demo_commerce_analysis():
    """
    Demo function to quickly show the commerce density analysis with widgets.
    """
    print("🏪 Commerce Density Analysis Demo")
    print("This will load data and create interactive visualizations.")

    # Create analysis
    iris_gdf, commerces_gdf, results = load_commerce_analysis_data()

    print(f"\n🍽️ Found {results['summary'].iloc[0]['total_establishments']:,} establishments across {results['summary'].iloc[0]['total_iris']} IRIS")
    # Create comprehensive report
    fig, axes = create_commerce_summary_report(results)
    plt.show()

    # Launch interactive widgets
    display_commerce_widgets(results, iris_gdf, commerces_gdf)

    return results


# Export matplotlib plots as images
def export_commerce_plots(analysis_results, output_dir='outputs/plots'):
    """
    Export commerce analysis plots to image files.

    Args:
        analysis_results (dict): Analysis results
        output_dir (str): Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    stats_gdf = analysis_results['stats']

    # Top commerce count plot
    fig, ax = create_top_commerce_count_plot(stats_gdf, top_n=15)
    fig.savefig(f'{output_dir}/commerce_count_top15.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Type distribution plot
    fig, ax = create_commerce_type_distribution_plot(stats_gdf, top_n=15)
    fig.savefig(f'{output_dir}/commerce_types_top15.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Density score plot
    fig, ax = create_commerce_density_score_plot(stats_gdf, top_n=15)
    fig.savefig(f'{output_dir}/commerce_density_score_top15.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Summary report
    fig, axes = create_commerce_summary_report(analysis_results)
    fig.savefig(f'{output_dir}/commerce_complete_report.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"✅ Plots exported to {output_dir}/")


# Backwards compatibility functions (matching original script)
def create_original_graph1(analysis_results):
    """Recreate the original Graphique 1: horizontal bar chart of top 10 by commerce count."""
    stats_gdf = analysis_results['stats']
    return create_top_commerce_count_plot(stats_gdf, top_n=10)


def create_original_graph2(analysis_results):
    """Recreate the original Graphique 2: grouped bar chart of commerce types by top 10 IRIS."""
    stats_gdf = analysis_results['stats']
    return create_commerce_type_distribution_plot(stats_gdf, top_n=10)
