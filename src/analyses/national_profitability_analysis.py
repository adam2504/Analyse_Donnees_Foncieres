"""
National Profitability Analysis Module
====================================

Business logic for analyzing rental profitability across French student cities.
Analyzes real estate prices, rents, and calculates gross rental yields.

Author: Axel & Valentine (modularized)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import zipfile
import glob
from pathlib import Path
from typing import List, Optional, Dict

# Get project root directory
script_dir = Path(__file__).parent  # src/analyses
project_root = script_dir.parent.parent  # project root

# Ensure data and outputs directories exist
data_dir = project_root / "data"
outputs_dir = project_root / "outputs"
data_dir.mkdir(exist_ok=True)
outputs_dir.mkdir(exist_ok=True)

def download_and_extract_dvf(
    url: str = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/ValeursFoncieres-2024.txt",
    cities: Optional[List[str]] = None,
    max_surface: int = 45,
    min_price: int = 1000,
    max_price: int = 100000,
    output_file: str = str(project_root / "data" / "dvf_appartements_nettoye.csv")
) -> pd.DataFrame:
    """
    Downloads and extracts DVF data for specified cities.

    Parameters:
    -----------
    url : str
        DVF file URL
    cities : list
        List of cities to analyze
    max_surface : int
        Maximum surface area in m²
    min_price : int
        Minimum price per m² to filter outliers
    max_price : int
        Maximum price per m² to filter outliers
    output_file : str
        CSV output file name

    Returns:
    ---------
    df_clean : DataFrame
        Cleaned DataFrame with square meter prices
    """

    if cities is None:
        cities = [
            "PARIS", "LYON", "LILLE", "TOULOUSE", "BORDEAUX", "MARSEILLE", "MONTPELLIER",
            "RENNES", "STRASBOURG", "NANTES", "GRENOBLE", "NANCY", "NICE", "ANGERS",
            "ROUEN", "CLERMONT-FERRAND", "CAEN", "DIJON", "TOURS", "REIMS", "AJACCIO",
            "ANNECY", "TOULON"
        ]

    useful_columns = [
        "Date mutation", "Nature mutation", "Valeur fonciere", "Code postal",
        "Commune", "Type local", "Surface reelle bati"
    ]

    print("="*80)
    print("STEP 1: PURCHASE PRICE EXTRACTION AND ANALYSIS (DVF 2024)")
    print("="*80)

    filtered_data = []
    chunks = pd.read_csv(url, sep="|", low_memory=False, chunksize=20000)

    for i, chunk in enumerate(chunks):
        filter_chunk = chunk[chunk["Commune"].isin(cities)]
        filter_chunk = filter_chunk[
            (filter_chunk["Type local"] == "Appartement") &
            (filter_chunk["Nature mutation"] == "Vente")
        ]

        filter_chunk["Surface reelle bati"] = pd.to_numeric(filter_chunk["Surface reelle bati"], errors="coerce")
        filter_chunk = filter_chunk[filter_chunk["Surface reelle bati"].between(1, max_surface)]

        filter_chunk["Valeur fonciere"] = (
            filter_chunk["Valeur fonciere"].astype(str).str.replace(",", ".").str.replace(" ", "")
        )
        filter_chunk["Valeur fonciere"] = pd.to_numeric(filter_chunk["Valeur fonciere"], errors="coerce")

        filter_chunk = filter_chunk.dropna(subset=["Valeur fonciere", "Surface reelle bati"])
        filter_chunk = filter_chunk[filter_chunk["Valeur fonciere"] > 0]
        filter_chunk = filter_chunk[useful_columns]

        if len(filter_chunk) > 0:
            filtered_data.append(filter_chunk)

    df = pd.concat(filtered_data, ignore_index=True)
    df["Prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]

    df_clean = df[(df['Prix_m2'] >= min_price) & (df['Prix_m2'] <= max_price)].copy()

    print(f"\nExtracted data: {len(df_clean):,} transactions after cleaning")
    print(f"Removed outliers: {len(df) - len(df_clean):,}")

    df_clean = df_clean[[
        "Commune", "Code postal", "Type local", "Surface reelle bati",
        "Valeur fonciere", "Prix_m2", "Date mutation"
    ]]
    df_clean.to_csv(output_file, index=False)

    return df_clean


def calculate_purchase_price_per_city(df_clean, output_file=str(project_root / "data" / "classement_prix_m2_par_ville.csv")):
    """Calculate purchase price statistics per city."""

    city_stats = df_clean.groupby("Commune").agg({
        'Prix_m2': ['count', 'mean', 'min', 'max']
    }).round(0)

    city_stats.columns = ['Sale count', 'Average price €/m²', 'Minimum price €/m²', 'Maximum price €/m²']
    city_stats = city_stats.sort_values('Average price €/m²', ascending=False).reset_index()
    city_stats.insert(0, 'Rank', range(1, len(city_stats) + 1))

    print("\n" + "="*80)
    print("CITY RANKING BY AVERAGE SQUARE METER PRICE")
    print("="*80)
    print(city_stats.to_string(index=False))

    national_average_price = df_clean['Prix_m2'].mean()
    print(f"\n\nNational average price: {national_average_price:,.0f} €/m²")
    print(f"Most expensive city: {city_stats.iloc[0]['Commune']} ({city_stats.iloc[0]['Average price €/m²']:,.0f} €/m²)")
    print(f"Least expensive city: {city_stats.iloc[-1]['Commune']} ({city_stats.iloc[-1]['Average price €/m²']:,.0f} €/m²)")

    city_stats.to_csv(output_file, index=False)

    return city_stats


def visualize_purchase_price(city_stats, df_clean, output_file=str(project_root / "outputs" / "classement_prix_m2_villes.png")):
    """Create a chart of purchase prices per city."""

    chart_stats = city_stats.sort_values('Average price €/m²', ascending=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    chart_cities = chart_stats['Commune']
    average_prices = chart_stats['Average price €/m²']
    y_pos = range(len(chart_cities))

    colors = plt.cm.RdYlGn_r([
        (p - average_prices.min()) / (average_prices.max() - average_prices.min())
        for p in average_prices
    ])
    bars = ax.barh(y_pos, average_prices, height=0.6, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=0.8)

    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height()/2,
                f'{int(average_prices.iloc[i]):,}',
                va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(chart_cities, fontsize=11, fontweight='bold')
    ax.set_xlabel('Price per m² (€)', fontsize=12, fontweight='bold')
    ax.set_title('Average purchase price per m² by city\nApartments ≤ 45m² (DVF 2024)',
                 fontsize=16, fontweight='bold', pad=20)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    national_average_price = df_clean['Prix_m2'].mean()
    ax.axvline(x=national_average_price, color='green', linestyle='--', linewidth=2,
               label=f'National average: {national_average_price:,.0f} €/m²', alpha=0.7)
    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def extract_rents(
    zip_url="https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/data_loyer.zip",
    max_surface=45,
    output_file=str(project_root / "data" / "loyers_moyens_par_ville.csv")
):
    """Extract and analyze rental data from the Observatory."""

    print("="*80)
    print("STEP 2: RENT EXTRACTION AND ANALYSIS")
    print("="*80)

    zip_path = str(project_root / "data" / "data_loyer.zip")
    extraction_folder = str(project_root / "data" / "data_loyer")
    rents_folder = os.path.join(extraction_folder, "data_loyer")

    if not os.path.exists(zip_path):
        response = requests.get(zip_url)
        with open(zip_path, "wb") as f:
            f.write(response.content)

    if not os.path.exists(extraction_folder):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extraction_folder)

    rent_files = glob.glob(os.path.join(rents_folder, "*.csv"))
    rent_results = []

    for rent_file in rent_files:
        try:
            file_name = os.path.basename(rent_file)
            parts = file_name.replace('.csv', '').split('_')
            city_name = parts[4].upper() if len(parts) > 4 else "UNKNOWN"

            df_rent = pd.read_csv(rent_file, encoding='cp1252', sep=';', low_memory=False)
            df_rent.columns = df_rent.columns.str.strip()

            if 'loyer_moyen' in df_rent.columns:
                df_rent['loyer_moyen'] = (
                    df_rent['loyer_moyen'].astype(str).str.replace(',', '.').str.replace(' ', '').replace('', None)
                )
                df_rent['loyer_moyen'] = pd.to_numeric(df_rent['loyer_moyen'], errors='coerce')

            if 'surface_moyenne' in df_rent.columns:
                df_rent['surface_moyenne'] = pd.to_numeric(df_rent['surface_moyenne'], errors='coerce')

            if 'nombre_observations' in df_rent.columns:
                df_rent['nombre_observations'] = pd.to_numeric(df_rent['nombre_observations'], errors='coerce')

            filter_rent = df_rent[
                (
                    (df_rent['Type_habitat'] == 'Appartement') |
                    (df_rent['nombre_pieces_homogene'].isin(['Appart 1P', 'Appart 2P']))
                ) &
                (df_rent['surface_moyenne'].notna()) &
                (df_rent['surface_moyenne'] <= max_surface) &
                (df_rent['loyer_moyen'].notna()) &
                (df_rent['nombre_observations'].notna())
            ]

            if len(filter_rent) > 0:
                average_rent_m2 = (
                    (filter_rent['loyer_moyen'] * filter_rent['nombre_observations']).sum() /
                    filter_rent['nombre_observations'].sum()
                )

                rent_results.append({
                    'Ville': city_name,
                    'Loyer_moyen_m2': round(average_rent_m2, 2),
                    'Nombre_observations': int(filter_rent['nombre_observations'].sum())
                })

        except Exception as e:
            pass

    if not rent_results:
        raise ValueError("No rent data extracted. Check source files or applied filters.")

    df_rents = pd.DataFrame(rent_results)

    df_rents = df_rents.drop_duplicates(subset=['Ville'], keep='first')

    city_mapping = {
        'AIX-MARSEILLE': 'MARSEILLE',
        'AGLO-PARIS': 'PARIS',
    }
    df_rents['Ville'] = df_rents['Ville'].replace(city_mapping)
    df_rents = df_rents.sort_values('Ville')

    print("\n" + "="*80)
    print("AVERAGE RENTS BY CITY")
    print("="*80)
    print(df_rents.to_string(index=False))

    national_average_rent = df_rents['Loyer_moyen_m2'].mean()
    cheapest_city = df_rents.loc[df_rents['Loyer_moyen_m2'].idxmin(), 'Ville']
    most_expensive_city = df_rents.loc[df_rents['Loyer_moyen_m2'].idxmax(), 'Ville']

    print(f"\n\nNational average rent: {national_average_rent:.2f} €/m²")
    print(f"Cheapest city: {cheapest_city} ({df_rents['Loyer_moyen_m2'].min():.2f} €/m²)")
    print(f"Most expensive city: {most_expensive_city} ({df_rents['Loyer_moyen_m2'].max():.2f} €/m²)")

    df_rents.to_csv(output_file, index=False)

    return df_rents


def visualize_rents(df_rents, output_file=str(project_root / "outputs" / "classement_loyers_villes.png")):
    """Create a chart of rents by city."""

    df_rents_chart = df_rents.sort_values('Loyer_moyen_m2', ascending=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    cities_chart = df_rents_chart['Ville']
    rents_chart = df_rents_chart['Loyer_moyen_m2']
    observations = df_rents_chart['Nombre_observations']
    y_pos = range(len(cities_chart))

    colors = plt.cm.RdYlGn_r([
        (r - rents_chart.min()) / (rents_chart.max() - rents_chart.min())
        for r in rents_chart
    ])
    bars = ax.barh(y_pos, rents_chart, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)

    for i, (bar, rent_val, obs) in enumerate(zip(bars, rents_chart, observations)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{rent_val:.2f} €/m²',
                va='center', fontsize=10, fontweight='bold')
        ax.text(1, bar.get_y() + bar.get_height()/2,
                f'({obs:,} obs)',
                va='center', ha='left', fontsize=8, color='white', style='italic')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cities_chart, fontsize=11, fontweight='bold')
    ax.set_xlabel('Average monthly rent (€/m²)', fontsize=12, fontweight='bold')
    ax.set_title('Average monthly rents per m² by city\nApartments ≤ 45m² (2024)',
                 fontsize=16, fontweight='bold', pad=20)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    national_average_rent = df_rents['Loyer_moyen_m2'].mean()
    ax.axvline(x=national_average_rent, color='blue', linestyle='--', linewidth=2,
               label=f'National average: {national_average_rent:.2f} €/m²', alpha=0.7)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    ax.annotate(f'Cheapest city\n{cities_chart.iloc[0]}',
                xy=(rents_chart.iloc[0], 0), xytext=(5, -2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax.annotate(f'Most expensive city\n{cities_chart.iloc[-1]}',
                xy=(rents_chart.iloc[-1], len(cities_chart)-1), xytext=(5, len(cities_chart)-1),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_profitability(
    purchase_file=str(project_root / "data" / "dvf_appartements_nettoye.csv"),
    rents_file=str(project_root / "data" / "loyers_moyens_par_ville.csv"),
    output_file=str(project_root / "data" / "rentabilite_locative_par_ville.csv")
):
    """Calculate gross rental profitability by city."""

    print("="*80)
    print("STEP 3: RENTAL PROFITABILITY CALCULATION")
    print("="*80)

    df_purchase = pd.read_csv(purchase_file, low_memory=False)
    df_rents = pd.read_csv(rents_file)

    purchase_price_per_city = df_purchase.groupby("Commune").agg({
        'Prix_m2': ['mean', 'count']
    }).round(2)
    purchase_price_per_city.columns = ['Prix_achat_moyen_m2', 'Nombre_ventes']
    purchase_price_per_city = purchase_price_per_city.reset_index()

    city_mapping = {
        'AIX-MARSEILLE': 'MARSEILLE',
        'AGLO-PARIS': 'PARIS',
    }
    df_rents['Ville'] = df_rents['Ville'].replace(city_mapping)
    purchase_price_per_city['Commune'] = purchase_price_per_city['Commune'].replace(city_mapping)

    df_profitability = pd.merge(
        purchase_price_per_city,
        df_rents,
        left_on='Commune',
        right_on='Ville',
        how='inner'
    )

    df_profitability['Loyer_annuel_m2'] = (df_profitability['Loyer_moyen_m2'] * 12).round(2)
    df_profitability['Rentabilite_brute_%'] = (
        (df_profitability['Loyer_moyen_m2'] * 12) / df_profitability['Prix_achat_moyen_m2'] * 100
    ).round(2)

    df_profitability = df_profitability.sort_values('Rentabilite_brute_%', ascending=False)
    df_profitability = df_profitability.reset_index(drop=True)
    df_profitability.insert(0, 'Rang', range(1, len(df_profitability) + 1))

    print("\n" + "="*120)
    print("CITY RANKING BY GROSS PROFITABILITY")
    print("="*120)
    print(df_profitability[['Rang', 'Commune', 'Rentabilite_brute_%', 'Loyer_moyen_m2',
                           'Loyer_annuel_m2', 'Prix_achat_moyen_m2', 'Nombre_ventes']].to_string(index=False))

    average_profit = df_profitability['Rentabilite_brute_%'].mean()
    max_profit = df_profitability['Rentabilite_brute_%'].max()
    min_profit = df_profitability['Rentabilite_brute_%'].min()
    most_profitable_city = df_profitability.iloc[0]['Commune']
    least_profitable_city = df_profitability.iloc[-1]['Commune']

    print(f"\n\nAverage profitability: {average_profit:.2f}%")
    print(f"Most profitable city: {most_profitable_city} ({max_profit:.2f}%)")
    print(f"Least profitable city: {least_profitable_city} ({min_profit:.2f}%)")

    df_profitability.to_csv(output_file, index=False)

    return df_profitability


def visualize_profitability(df_profitability, output_file=str(project_root / "outputs" / "analyse_rentabilite_finale.png")):
    """Create profitability charts (bars + scatter)."""

    df_profit_chart = df_profitability.sort_values('Rentabilite_brute_%', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Rental Profitability Analysis - Apartments ≤ 45m² (2024)',
                 fontsize=18, fontweight='bold', y=0.98)

    # Chart 1: Bars
    ax1 = axes[0]
    cities_bar = df_profit_chart['Commune']
    profitability_bar = df_profit_chart['Rentabilite_brute_%']
    y_pos = range(len(cities_bar))

    colors = plt.cm.RdYlGn([
        (p - profitability_bar.min()) / (profitability_bar.max() - profitability_bar.min())
        for p in profitability_bar
    ])
    bars1 = ax1.barh(y_pos, profitability_bar, color=colors, edgecolor='black', linewidth=0.8)

    for i, (bar, profit_val) in enumerate(zip(bars1, profitability_bar)):
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{profit_val:.2f}%',
                va='center', fontsize=11, fontweight='bold')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cities_bar, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Gross profitability (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Gross Profitability by City', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    average_profit = df_profitability['Rentabilite_brute_%'].mean()
    ax1.axvline(x=average_profit, color='red', linestyle='--', linewidth=2.5, alpha=0.8,
               label=f'Average: {average_profit:.2f}%')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # Chart 2: Scatter
    ax2 = axes[1]

    scatter = ax2.scatter(
        df_profitability['Prix_achat_moyen_m2'],
        df_profitability['Loyer_annuel_m2'],
        s=df_profitability['Rentabilite_brute_%'] * 80,
        c=df_profitability['Rentabilite_brute_%'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )

    for idx, row in df_profitability.iterrows():
        ax2.annotate(
            row['Commune'],
            (row['Prix_achat_moyen_m2'], row['Loyer_annuel_m2']),
            fontsize=10, ha='center', fontweight='bold', alpha=0.9
        )

    ax2.set_xlabel('Average purchase price (€/m²)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Annual rent (€/m²)', fontsize=13, fontweight='bold')
    ax2.set_title('Annual Rent vs Purchase Price\n(Bubble size = Profitability)',
                 fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Gross profitability (%)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_national_profitability(
    cities=None,
    max_surface=45,
    min_price=1000,
    max_price=100000
):
    """
    Main function that runs the complete national profitability analysis.

    Parameters:
    -----------
    cities : list, optional
        List of cities to analyze (default: 23 student cities)
    max_surface : int
        Maximum surface area in m²
    min_price : int
        Minimum price per m² to filter outliers
    max_price : int
        Maximum price per m² to filter outliers

    Returns:
    ---------
    dict : Dictionary containing all generated DataFrames
    """

    print("\n")
    print("="*80)
    print("COMPLETE NATIONAL PROFITABILITY ANALYSIS")
    print("Student apartments ≤ 45m²")
    print("="*80)
    print("\n")

    # Step 1: Purchase prices
    df_purchase = download_and_extract_dvf(
        cities=cities,
        max_surface=max_surface,
        min_price=min_price,
        max_price=max_price
    )
    city_stats = calculate_purchase_price_per_city(df_purchase)
    visualize_purchase_price(city_stats, df_purchase)

    # Step 2: Rents
    df_rents = extract_rents(max_surface=max_surface)
    visualize_rents(df_rents)

    # Step 3: Profitability
    df_profitability = calculate_profitability()
    visualize_profitability(df_profitability)

    results = {
        'df_achat': df_purchase,
        'stats_villes': city_stats,
        'df_loyers': df_rents,
        'df_rentabilite': df_profitability
    }

    return results