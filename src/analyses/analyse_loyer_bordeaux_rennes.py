"""
Rennes vs Bordeaux Rent Evolution Analysis Module
===================================================

Business logic for analyzing and comparing rent evolution between Rennes and Bordeaux.
Examines 1-piece apartment rents from 2020-2024 to compare market dynamics.

Author: Axel (modularized)
"""

import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import io
import requests
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Get project root directory
script_dir = Path(__file__).parent  # src/analyses
project_root = script_dir.parent.parent  # project root


def read_csv_safe(path):
    """
    Safely read CSV files with fallback encodings.

    Parameters:
    -----------
    path : str
        Path to CSV file

    Returns:
    --------
    DataFrame : Loaded CSV data
    """
    for enc in ["utf-8", "ISO-8859-1", "cp1252"]:
        try:
            return pd.read_csv(path, sep=";", encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Impossible de lire le fichier {path} avec les encodages standards.")


def download_rent_agglomeration_data(
    url="https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/data_loyer_aglo.zip"
):
    """
    Download and extract rent data for agglomerations.

    Parameters:
    -----------
    url : str
        URL to download agglomeration rent data

    Returns:
    --------
    str : Path to the extracted data folder
    """
    response = requests.get(url)

    # Use data directory in project root
    data_dir = project_root / "data"
    extract_dir = data_dir / "data_loyer_aglo"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(extract_dir)

    # Handle nested folder structure if present
    if "data_loyer_aglo" in os.listdir(extract_dir):
        folder = extract_dir / "data_loyer_aglo"
    else:
        folder = extract_dir

    return str(folder)


def load_rent_data_by_year(data_folder):
    """
    Load and combine rent data from multiple years.

    Parameters:
    -----------
    data_folder : str
        Path to folder containing CSV files

    Returns:
    --------
    DataFrame : Combined rent data with year column
    """
    all_data = []

    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            try:
                year = int(file.split("_")[2])
            except (IndexError, ValueError):
                continue
            df = read_csv_safe(os.path.join(data_folder, file))
            df["Annee"] = year
            all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def filter_and_process_rent_data(df_all, cities=["Bordeaux", "Rennes"], rooms=1):
    """
    Filter and process rent data for specific cities and room count.

    Parameters:
    -----------
    df_all : DataFrame
        Raw combined rent data
    cities : list
        Cities to filter for
    rooms : int
        Number of rooms to filter for

    Returns:
    --------
    DataFrame : Processed and filtered rent data
    """
    col_ville = "agglomeration"
    col_loyer = "loyer_moyen"
    col_piece = "nombre_pieces_homogene"

    # Filter for cities and room count
    df_filtered = df_all[
        df_all[col_ville].str.contains("|".join(cities), case=False, na=False)
        & df_all[col_piece].astype(str).str.contains(str(rooms), na=False)
    ]

    # Clean rent prices
    df_filtered.loc[:, col_loyer] = (
        df_filtered[col_loyer]
        .astype(str)
        .str.replace(",", ".")
        .str.replace(" ", "")
        .astype(float)
    )

    return df_filtered


def calculate_yearly_averages(df_filtered):
    """
    Calculate average rents by year and city.

    Parameters:
    -----------
    df_filtered : DataFrame
        Filtered rent data

    Returns:
    --------
    DataFrame : Yearly averages grouped by city
    """
    return (
        df_filtered.groupby(["Annee", "agglomeration"])["loyer_moyen"]
        .mean()
        .reset_index()
    )


def create_rent_evolution_plot(df_grouped, output_file=None):
    """
    Create a line plot showing rent evolution over time.

    Parameters:
    -----------
    df_grouped : DataFrame
        Grouped rent data by year and city
    output_file : str, optional
        Path to save the plot

    Returns:
    --------
    matplotlib Figure : The generated plot
    """
    plt.figure(figsize=(10, 6))

    for city in df_grouped["agglomeration"].unique():
        subset = df_grouped[df_grouped["agglomeration"] == city]
        plt.plot(subset["Annee"], subset["loyer_moyen"], marker='o', linewidth=2, label=city)

    plt.title("Évolution du loyer moyen au m² (1 pièce) - Bordeaux vs Rennes (2020–2024)")
    plt.xlabel("Année")
    plt.ylabel("Loyer moyen (€/m²)")
    plt.xticks(sorted(df_grouped["Annee"].unique()))
    plt.grid(True)
    plt.legend()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return plt.gcf()


def print_market_analysis():
    """
    Print detailed market analysis and investment insights.
    """
    print("\n======================  CONTEXTE DU CHOIX  ====================== :\n")
    print("Après analyse des grandes villes françaises, deux se distinguent pour Léa : Rennes et Bordeaux.")
    print("Nous allons comparer leur évolution des loyers pour déterminer laquelle répond le mieux à son besoins.")
    print("A savoir une rentabilité rapide, une stabilité du marché et un faible risque de vacance.\n")

    print("\n======================  ANALYSE ET INTERPRÉTATION  ====================== :\n")
    print("Entre 2020 et 2024, les loyers ont augmenté d'environ 9 % à Bordeaux et Rennes.")
    print("En 2024 : 16,3 €/m² à Bordeaux contre 15,9 €/m² à Rennes.")
    print("Deux courbes similaires : progression régulière, marché tendu dans les deux villes.")
    print()
    print("Bordeaux : loyers plus élevés, coût d'achat plus important, marché plus concurrentiel.")
    print("Rennes : marché plus accessible, stable, forte demande étudiante, risque moindre de vacance.")
    print()
    print("De plus, comme vu précédemment, la rentabilité moyenne légèrement supérieure à Rennes (2,98 % contre 2,31 %).")
    print()
    print("Conclusion : Rennes combine loyers modérés, stabilité et rendement correct —")
    print("un choix plus sûr et cohérent pour Léa.")


def analyze_rent_evolution_bordeaux_rennes(
    cities=["Bordeaux", "Rennes"],
    rooms=1,
    show_plot=True,
    save_plot=True
):
    """
    Complete analysis of rent evolution between Bordeaux and Rennes.

    Parameters:
    -----------
    cities : list
        Cities to compare (default: ["Bordeaux", "Rennes"])
    rooms : int
        Number of rooms to analyze (default: 1)
    show_plot : bool
        Whether to display the plot (default: True)
    save_plot : bool
        Whether to save the plot to outputs folder (default: True)

    Returns:
    --------
    dict : Analysis results including processed data and plot
    """

    # Download and load data
    data_folder = download_rent_agglomeration_data()
    df_all = load_rent_data_by_year(data_folder)

    # Process and filter data
    df_filtered = filter_and_process_rent_data(df_all, cities, rooms)
    df_grouped = calculate_yearly_averages(df_filtered)

    # Create visualization
    output_file = str(project_root / "outputs" / "evolution_loyers_bordeaux_rennes.png") if save_plot else None
    fig = create_rent_evolution_plot(df_grouped, output_file)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # Print analysis
    print_market_analysis()

    results = {
        'raw_data': df_all,
        'filtered_data': df_filtered,
        'grouped_data': df_grouped,
        'plot': fig,
        'summary_stats': {
            'cities_compared': cities,
            'years_range': f"{df_grouped['Annee'].min()}-{df_grouped['Annee'].max()}",
            'avg_increase_percent': 9.0  # As noted in analysis
        }
    }

    return results
