"""
Transport Analysis Widgets Module
================================

Interactive widgets and static plots for visualizing public transport coverage
and accessibility analysis in Rennes.

Author: Adam Jouini
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

# Import our modular analysis functions
from src.loaders import load_iris_rennes, load_transport_stops
from src.analyses.analyse_transports import analyze_transport_coverage

# Import plotting defaults
from src.config import DEFAULT_FIG_SIZE, DEFAULT_DPI


def load_transport_analysis_data():
    """
    Load and prepare analysis data for transport widgets.

    Returns:
        tuple: (iris_gdf, transports_gdf, analysis_results)
    """
    print("🔄 Loading IRIS and transport data for Rennes...")
    iris_gdf = load_iris_rennes()
    transports_gdf = load_transport_stops()

    print("🔄 Running transport coverage analysis...")
    results = analyze_transport_coverage(iris_gdf, transports_gdf)

    print("✅ Transport data and analysis ready for visualization.")
    return iris_gdf, transports_gdf, results


def create_top_transport_density_plot(stats_gdf, top_n=10, figsize=None):
    """
    Create a horizontal bar chart showing top IRIS by transport density.

    Args:
        stats_gdf (GeoDataFrame): Transport statistics
        top_n (int): Number of top IRIS to display
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = DEFAULT_FIG_SIZE

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Get top N data sorted ascending for better visualization
    top_data = stats_gdf.nlargest(top_n, 'densite_arrets').sort_values('densite_arrets', ascending=True)

    # Create horizontal bar chart
    bars = ax.barh(range(len(top_data)), top_data['densite_arrets'],
                   color=sns.color_palette("RdYlGn_r", len(top_data)))

    # Customize plot
    ax.set_yticks(range(len(top_data)))
    ax.set_yticklabels(top_data['LIB_IRIS'])
    ax.set_xlabel('Stop density (per km²)')
    ax.set_title(f'Top {top_n} IRIS - Public transport density in Rennes')

    # Add value labels on bars
    for i, (idx, row) in enumerate(top_data.iterrows()):
        ax.text(row['densite_arrets'] + 0.1, i,
                f'{row["densite_arrets"]:.1f}',
                va='center', fontweight='bold')

    plt.tight_layout()
    return fig, ax


def create_transport_type_comparison_plot(stats_gdf, top_n=10, figsize=None):
    """
    Create a grouped bar chart showing transport types distribution.

    Args:
        stats_gdf (GeoDataFrame): Transport statistics
        top_n (int): Number of top IRIS to display
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = DEFAULT_FIG_SIZE

    fig, ax = plt.subplots(figsize=(figsize[0] * 1.2, figsize[1]), dpi=DEFAULT_DPI)

    # Get top N by total stops
    top_data = stats_gdf.nlargest(top_n, 'total_arrets')

    # Set up data for grouped bars
    categories = ['Bus', 'Métro', 'Train']
    x = np.arange(len(top_data))
    width = 0.25

    # Create bars for each category
    bars_bus = ax.bar(x - width, top_data['Bus'], width,
                      label='Bus', color='#FF6B6B')
    bars_metro = ax.bar(x, top_data['Métro'], width,
                        label='Métro', color='#4ECDC4')
    bars_train = ax.bar(x + width, top_data['Train'], width,
                        label='Train', color='#45B7D1')

    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(top_data['LIB_IRIS'], rotation=45, ha='right')
    ax.set_ylabel('Number of stops')
    ax.set_title(f'Distribution by transport type - Top {top_n} IRIS')
    ax.legend()

    # Add value labels on bars
    def add_value_labels(bars, offset=0):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2.,
                        height + offset, f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')

    add_value_labels(bars_bus)
    add_value_labels(bars_metro)
    add_value_labels(bars_train, 0.5)

    plt.tight_layout()
    return fig, ax


def create_accessibility_heatmap(stats_gdf, figsize=None):
    """
    Create a heatmap showing transport accessibility scores.

    Args:
        stats_gdf (GeoDataFrame): Transport statistics
        figsize (tuple, optional): Figure size
    """
    if figsize is None:
        figsize = (figsize[0] if figsize else DEFAULT_FIG_SIZE[0] * 1.5,
                  figsize[1] if figsize else DEFAULT_FIG_SIZE[1])

    fig, ax = plt.subplots(figsize=figsize, dpi=DEFAULT_DPI)

    # Get accessibility data
    accessibility_data = stats_gdf[['LIB_IRIS', 'accessibility_score', 'densite_arrets']].copy()

    # Convert text scores to numeric for color mapping
    score_mapping = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4}
    accessibility_data['score_numeric'] = accessibility_data['accessibility_score'].map(score_mapping)

    # Sort by density for better visualization
    accessibility_data = accessibility_data.sort_values('score_numeric', ascending=False).head(20)

    # Create heatmap
    scatter = ax.scatter(
        accessibility_data['LIB_IRIS'],
        [1] * len(accessibility_data),  # All on same y level
        c=accessibility_data['score_numeric'],
        s=accessibility_data['densite_arrets'] * 10,  # Size based on density
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black'
    )

    # Customize plot
    ax.set_yticks([])
    ax.set_xlabel('Quartier (IRIS)')
    ax.set_title('Accessibilité transports en commun - Rennes')
    plt.xticks(rotation=45, ha='right')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Score d\'accessibilité')
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels(['Très faible', 'Faible', 'Moyen', 'Élevé'])

    # Add legend for size
    sizes = [5, 15, 25, 35]
    labels = [f'{s/10:.1f}' for s in sizes]
    legend_elements = [plt.scatter([], [], s=size*10, c='gray', alpha=0.5, edgecolors='black')
                      for size in sizes]
    ax.legend(legend_elements, labels, title='Densité\n(arrêts/km²)',
             bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig, ax


def display_transport_widgets(analysis_results=None, iris_gdf=None, transports_gdf=None):
    """
    Create and display the interactive transport analysis widgets.

    Args:
        analysis_results (dict, optional): Pre-computed analysis results
        iris_gdf (GeoDataFrame, optional): IRIS polygons
        transports_gdf (GeoDataFrame, optional): Transport stops
    """
    # Load data if not provided
    if any(param is None for param in [analysis_results, iris_gdf, transports_gdf]):
        print("🔄 No datasets provided → loading via analysis pipeline...")
        iris_gdf, transports_gdf, analysis_results = load_transport_analysis_data()
    else:
        print("⚡ Using provided datasets.")

    print("🧩 Creating interactive transport widgets...")

    # Extract the stats dataframe for plotting
    stats_gdf = analysis_results['stats']

    # Plot type selector
    plot_type = widgets.Dropdown(
        options=['Top Density', 'Type Distribution', 'Accessibility Heatmap'],
        value='Top Density',
        description='Type de graphique :',
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

        if plot_type == 'Top Density':
            fig, ax = create_top_transport_density_plot(stats_gdf, top_n)
        elif plot_type == 'Type Distribution':
            fig, ax = create_transport_type_comparison_plot(stats_gdf, top_n)
        elif plot_type == 'Accessibility Heatmap':
            # For heatmap, show top 20 instead of top N
            fig, ax = create_accessibility_heatmap(stats_gdf)
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

    print("✅ Interactive transport widgets ready!")
    display(interactive_plot)


def create_transport_summary_report(analysis_results):
    """
    Create a comprehensive report with multiple transport visualizations.

    Args:
        analysis_results (dict): Complete analysis results
    """
    print("📊 Generating comprehensive transport report...")

    stats_gdf = analysis_results['stats']
    summary_df = analysis_results['summary']
    top_dense = analysis_results['top_dense']
    accessibility = analysis_results['accessibility']

    # Set up the matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rapport d\'Analyse - Transports en Commun à Rennes',
                fontsize=16, fontweight='bold')

    # 1. Top density plot
    top_data = stats_gdf.nlargest(10, 'densite_arrets').sort_values('densite_arrets', ascending=True)
    axes[0, 0].barh(range(len(top_data)), top_data['densite_arrets'],
                    color=sns.color_palette("RdYlGn_r", len(top_data)))
    axes[0, 0].set_yticks(range(len(top_data)))
    axes[0, 0].set_yticklabels(top_data['LIB_IRIS'], fontsize=8)
    axes[0, 0].set_xlabel('Arrêts/km²')
    axes[0, 0].set_title('Top 10 - Densité d\'arrêts')

    # 2. Transport type distribution for top 10
    top_types = stats_gdf.nlargest(10, 'total_arrets')
    categories = ['Bus', 'Métro', 'Train', 'Autre']
    bottom = np.zeros(len(top_types))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95A5A6']
    for i, cat in enumerate(categories):
        axes[0, 1].bar(range(len(top_types)), top_types[cat],
                       bottom=bottom, label=cat, color=colors[i])
        bottom += top_types[cat]

    axes[0, 1].set_xticks(range(len(top_types)))
    axes[0, 1].set_xticklabels([name[:15] + '...' if len(name) > 15 else name
                                for name in top_types['LIB_IRIS']], rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('Nombre d\'arrêts')
    axes[0, 1].set_title('Répartition par type - Top 10')
    axes[0, 1].legend()

    # 3. Accessibility distribution
    accessibility_counts = accessibility['accessibility_score'].value_counts()
    colors_access = {'Very Low': '#d73027', 'Low': '#f46d43',
                    'Medium': '#fdae61', 'High': '#1a9850'}

    axes[1, 0].pie(accessibility_counts.values,
                   labels=accessibility_counts.index,
                   colors=[colors_access.get(score, '#ccc') for score in accessibility_counts.index],
                   autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Répartition des scores d\'accessibilité')

    # 4. Coverage statistics text
    axes[1, 1].axis('off')
    summary_text = f"🚏 Arrêts totaux: {summary_df.iloc[0]['total_stops']:,}\n" \
                   f"🏙️ IRIS total: {summary_df.iloc[0]['total_iris']}\n" \
                   f"🏙️ IRIS avec transports: {summary_df.iloc[0]['iris_with_transports']}\n" \
                   f"📈 Couverture: {summary_df.iloc[0]['coverage_percentage']:.1f}%"

    axes[1, 1].text(0.1, 0.9, "Statistiques Clés:", fontsize=12, fontweight='bold',
                   transform=axes[1, 1].transAxes, verticalalignment='top')
    axes[1, 1].text(0.1, 0.7, summary_text, fontsize=10,
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))

    plt.tight_layout()
    return fig, axes


# Convenience function for quick demonstration
def demo_transport_analysis():
    """
    Demo function to quickly show the transport coverage analysis with widgets.
    """
    print("🚌 Transport Coverage Analysis Demo")
    print("This will load data and create interactive visualizations.")

    # Create analysis
    iris_gdf, transports_gdf, results = load_transport_analysis_data()

    print(f"\n🚇 Found {results['summary'].iloc[0]['total_stops']:,} stops across {results['summary'].iloc[0]['total_iris']} IRIS")
    # Create comprehensive report
    fig, axes = create_transport_summary_report(results)
    plt.show()

    # Launch interactive widgets
    display_transport_widgets(results, iris_gdf, transports_gdf)

    return results


# Export matplotlib plots as images
def export_transport_plots(analysis_results, output_dir='outputs/plots'):
    """
    Export transport analysis plots to image files.

    Args:
        analysis_results (dict): Analysis results
        output_dir (str): Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    stats_gdf = analysis_results['stats']

    # Top density plot
    fig, ax = create_top_transport_density_plot(stats_gdf, top_n=15)
    fig.savefig(f'{output_dir}/transport_density_top15.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Type distribution plot
    fig, ax = create_transport_type_comparison_plot(stats_gdf, top_n=15)
    fig.savefig(f'{output_dir}/transport_types_top15.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Accessibility heatmap
    fig, ax = create_accessibility_heatmap(stats_gdf)
    fig.savefig(f'{output_dir}/transport_accessibility_heatmap.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Summary report
    fig, axes = create_transport_summary_report(analysis_results)
    fig.savefig(f'{output_dir}/transport_complete_report.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"✅ Plots exported to {output_dir}/")
