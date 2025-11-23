"""
Rennes Districts Profitability Analysis Module
==============================================

Business logic for analyzing rental profitability across Rennes districts.
Calculates and compares gross rental yields by quartier and surface category.

Author: Lucien (modularized)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import zipfile
import io
from pathlib import Path

# Get project root directory
script_dir = Path(__file__).parent  # src/analyses
project_root = script_dir.parent.parent  # project root


def load_district_price_data(url):
    """
    Load Rennes districts price data.

    Parameters:
    -----------
    url : str
        URL for quartier price data

    Returns:
    --------
    DataFrame : Processed price data
    """
    df = pd.read_csv(url, sep=",")
    df.columns = df.columns.str.strip()

    # Convert datatypes
    df["Nb de pièces"] = df["Nb de pièces"].astype(int)
    df["Surface (m²)"] = df["Surface (m²)"].astype(float)
    df["Prix au m² (€)"] = df["Prix au m² (€)"].astype(float)
    df["Prix total estimé (€)"] = df["Prix total estimé (€)"].astype(float)

    return df


def load_rent_data(url_base, filename):
    """
    Load and extract rent data from zip file.

    Parameters:
    -----------
    url_base : str
        Base URL for data
    filename : str
        Filename within the zip to extract

    Returns:
    --------
    DataFrame : Processed rent data
    """
    url = f"{url_base}/data_loyer.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(filename), encoding="latin-1", sep=";")
    df.columns = df.columns.str.strip()

    return df


def filter_student_apartments(df):
    """
    Filter for student apartments (new apartments ≤45m²).

    Parameters:
    -----------
    df : DataFrame
        Price data

    Returns:
    --------
    DataFrame : Filtered data
    """
    return df[
        (df["Type de bien"] == "Appartement") &
        (df["Surface (m²)"] <= 45) &
        (df["Statut"] == "neuf")
    ]


def filter_clean_rent_data(df_rent):
    """
    Filter and clean rent data.

    Parameters:
    -----------
    df_rent : DataFrame
        Raw rent data

    Returns:
    --------
    DataFrame : Filtered and cleaned rent data
    """
    df_filtered = df_rent[
        (
            (df_rent["Type_habitat"] == "Appartement") |
            (df_rent["nombre_pieces_homogene"] == "Appart 1P") |
            (df_rent["nombre_pieces_homogene"] == "Appart 2P")
        ) &
        (df_rent["surface_moyenne"] < 50)
    ]

    df_filtered = df_filtered.sort_values(by="moyenne_loyer_mensuel")
    df_filtered = df_filtered[df_filtered["nombre_logements"].notna() & (df_filtered["nombre_logements"] != 0)]

    return df_filtered


def map_zones_to_quartiers(df_rent):
    """
    Map zone codes to quartier names.

    Parameters:
    -----------
    df_rent : DataFrame
        Rent data with Zone_calcul column

    Returns:
    --------
    DataFrame : Data with Quartier column added
    """
    mapping_zones = {
        "L3500.1.01": "Centre",
        "L3500.1.02": "Thabor-Saint Helier",
        "L3500.1.03": "Lorient-Saint Brieuc",
        "L3500.1.04": "Nord Saint Martin",
        "L3500.1.05": "Maurepas-Patton",
        "L3500.1.06": "Atalante Beaulieu",
        "L3500.1.07": "Francisco-Vern-Poterie",
        "L3500.1.08": "Sud Gare",
        "L3500.1.09": "Cleunay-Arsenal Redon",
        "L3500.1.10": "Villejean-Beauregard",
        "L3500.1.11": "Le Blosne",
        "L3500.1.12": "Bréquigny"
    }

    df_rent["Quartier"] = df_rent["Zone_calcul"].map(mapping_zones).fillna("Inconnu")
    return df_rent


def rename_columns(df_price):
    """
    Rename columns for consistency between datasets.

    Parameters:
    -----------
    df_price : DataFrame
        Price data with French column names

    Returns:
    --------
    DataFrame : Data with English column names
    """
    return df_price.rename(columns={
        "Type de bien": "Type_habitat",
        "Surface (m²)": "surface_moyenne",
        "Prix au m² (€)": "Prix_m2",
        "Prix total estimé (€)": "Prix_total"
    })


def create_surface_categories(ranges=None):
    """
    Create surface category ranges for analysis.

    Parameters:
    -----------
    ranges : list, optional
        List of (min, max, label) tuples

    Returns:
    --------
    list : Surface ranges
    """
    if ranges is None:
        ranges = [
            (20, 35, "30"),
            (35, 50, "45")
        ]
    return ranges


def calculate_median_by_category(df_rent, surface_ranges, numeric_columns):
    """
    Calculate median rents by quartier and surface category.

    Parameters:
    -----------
    df_rent : DataFrame
        Rent data
    surface_ranges : list
        Surface category ranges
    numeric_columns : list
        Columns to calculate medians for

    Returns:
    --------
    DataFrame : Aggregated data by quartier and surface category
    """
    dfs = []
    mapping_zones = {
        "L3500.1.01": "Centre",
        "L3500.1.02": "Thabor-Saint Helier",
        "L3500.1.03": "Lorient-Saint Brieuc",
        "L3500.1.04": "Nord Saint Martin",
        "L3500.1.05": "Maurepas-Patton",
        "L3500.1.06": "Atalante Beaulieu",
        "L3500.1.07": "Francisco-Vern-Poterie",
        "L3500.1.08": "Sud Gare",
        "L3500.1.09": "Cleunay-Arsenal Redon",
        "L3500.1.10": "Villejean-Beauregard",
        "L3500.1.11": "Le Blosne",
        "L3500.1.12": "Bréquigny"
    }

    for min_size, max_size, label in surface_ranges:
        df_temp = df_rent[
            (df_rent["Type_habitat"] == "Appartement") &
            (df_rent["surface_moyenne"].between(min_size, max_size))
        ].copy()

        df_group = (
            df_temp
            .groupby("Zone_calcul", as_index=False)[numeric_columns]
            .median(numeric_only=True)
        )

        df_group["Quartier"] = df_group["Zone_calcul"].map(mapping_zones).fillna("Inconnu")
        df_group["Catégorie_surface_environ"] = label
        dfs.append(df_group)

    return pd.concat(dfs, ignore_index=True)


def merge_price_and_rent_data(df_grouped, df_price):
    """
    Merge price and rent data for profitability calculation.

    Parameters:
    -----------
    df_grouped : DataFrame
        Grouped rent data
    df_price : DataFrame
        Price data

    Returns:
    --------
    DataFrame : Merged data with alignment
    """
    df_grouped["Catégorie_surface_environ"] = df_grouped["Catégorie_surface_environ"].astype(float)
    df_price["Catégorie_surface_environ"] = df_price["surface_moyenne"]  # alignment

    return pd.merge(
        df_grouped,
        df_price,
        on=["Quartier", "Catégorie_surface_environ"],
        how="left"
    )


def calculate_profitability(df_merged):
    """
    Calculate gross rental profitability percentages.

    Parameters:
    -----------
    df_merged : DataFrame
        Merged price and rent data

    Returns:
    --------
    DataFrame : Data with profitability column added
    """
    df_merged["rentabilite_brute_%"] = (df_merged["loyer_mensuel_median"] * 12 / df_merged["Prix_total"]) * 100
    return df_merged.sort_values(by=["rentabilite_brute_%"], ascending=False)


def create_profitability_plot(df_merged, output_file=None):
    """
    Create seaborn bar plot of profitability by quartier and surface category.

    Parameters:
    -----------
    df_merged : DataFrame
        Merged data with profitability
    output_file : str, optional
        Path to save the plot

    Returns:
    --------
    matplotlib Figure : The generated plot
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_merged,
        x="Quartier",
        y="rentabilite_brute_%",
        hue="Catégorie_surface_environ",
        palette="Set2"
    )
    plt.ylabel("Rentabilité brute (%)")
    plt.xlabel("Quartier")
    plt.title("Rentabilité brute par Quartier et surface à Rennes")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Surface (m²)")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return plt.gcf()


def get_top_performers(df_merged, n=3):
    """
    Get top and bottom performers by profitability.

    Parameters:
    -----------
    df_merged : DataFrame
        Data with profitability
    n : int
        Number of top/bottom performers to return

    Returns:
    --------
    tuple : (top_performers, bottom_performers, mean_profitability)
    """
    df_clean = df_merged.dropna(subset=["rentabilite_brute_%"])
    top = df_clean.head(n)
    bottom = df_clean.tail(n)
    mean_profit = df_clean["rentabilite_brute_%"].mean()

    return top, bottom, mean_profit


def print_profitability_summary(top, bottom, mean_profit):
    """
    Print summary of profitability analysis.

    Parameters:
    -----------
    top : DataFrame
        Top performing districts
    bottom : DataFrame
        Bottom performing districts
    mean_profit : float
        Mean profitability percentage
    """
    print(f"À Rennes, la rentabilité moyenne des appartements est d'environ {mean_profit:.2f}%.")
    print(f"Les quartiers les plus rentables sont : {', '.join(top['Quartier'].unique())}.")
    print(f"Les quartiers les moins rentables sont : {', '.join(bottom['Quartier'].unique())}.")
    print("Les logements autour de 30m² sont souvent plus rentables (petits T1/T2 étudiants).")
    print("Les surfaces de 45m² sont plus stables à long terme, avec une vacance locative moindre.")


def analyze_rennes_district_profitability(
    save_plot=True,
    surface_ranges=None,
    show_plot=True
):
    """
    Complete analysis of Rennes district rental profitability.

    Parameters:
    -----------
    save_plot : bool
        Whether to save the plot to outputs folder
    surface_ranges : list, optional
        Custom surface category ranges
    show_plot : bool
        Whether to display the plot

    Returns:
    --------
    dict : Analysis results including processed data and plot
    """

    print("=== ANALYSE DE LA RENTABILITÉ DES QUARTIERS À RENNES ===")

    # Set up data source
    BDD = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main"

    # Load and process price data
    df_price = load_district_price_data(f"{BDD}/rennes_quartiers.csv.txt")
    df_students = filter_student_apartments(df_price)
    df_students = rename_columns(df_students)

    # Load and process rent data
    df_rent = load_rent_data(BDD, 'data_loyer/Base_OP_2024_L3500_Rennes.csv')
    df_rent = filter_clean_rent_data(df_rent)
    df_rent = map_zones_to_quartiers(df_rent)

    # Create surface categories and calculate medians
    if surface_ranges is None:
        surface_ranges = create_surface_categories()

    numeric_columns = [
        "loyer_median", "loyer_moyen",
        "loyer_mensuel_median", "moyenne_loyer_mensuel",
        "surface_moyenne", "nombre_logements"
    ]

    df_grouped = calculate_median_by_category(df_rent, surface_ranges, numeric_columns)

    # Merge data and calculate profitability
    df_merged = merge_price_and_rent_data(df_grouped, df_students)
    df_profit = calculate_profitability(df_merged)

    # Create visualization
    output_file = str(project_root / "outputs" / "rentabilite_quartiers_rennes.png") if save_plot else None
    fig = create_profitability_plot(df_profit, output_file)

    if not show_plot:
        plt.close(fig)
    elif show_plot:
        plt.show()

    # Get performance summary
    top_performers, bottom_performers, avg_profitability = get_top_performers(df_profit)
    print_profitability_summary(top_performers, bottom_performers, avg_profitability)

    results = {
        'price_data': df_students,
        'rent_data': df_rent,
        'merged_data': df_profit,
        'plot': fig,
        'top_performers': top_performers,
        'bottom_performers': bottom_performers,
        'average_profitability': avg_profitability,
        'summary_stats': {
            'total_districts': len(df_profit['Quartier'].unique()),
            'surface_categories': surface_ranges,
            'mean_profitability': avg_profitability
        }
    }

