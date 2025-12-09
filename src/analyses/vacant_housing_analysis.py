"""
Vacant Housing Analysis Module
===============================

Business logic for analyzing vacant housing rates across French communes.
Analyzes INSEE data for housing vacancy rates in major French cities.

Author: Lucien (modularized)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Optional, Dict, Any

# Get project root directory
script_dir = Path(__file__).parent  # src/analyses
project_root = script_dir.parent.parent  # project root

def load_vacant_housing_data(
    insee_vacancy_url: str = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/insee_rp_hist_1968.xlsx",
    population_url: str = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/POPULATION_MUNICIPALE_COMMUNES_FRANCE_lucien.xlsx",
    communes_url: str = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/communes-france-2025.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare vacant housing data from INSEE sources.

    Parameters:
    -----------
    insee_vacancy_url : str
        URL for INSEE housing vacancy data
    population_url : str
        URL for population data
    communes_url : str
        URL for communes coordinates data

    Returns:
    --------
    tuple : (df_vac_processed, df_pop_processed, df_communes_coords)
        Processed DataFrames for analysis
    """

    # Load data
    df_vac = pd.read_excel(insee_vacancy_url, header=1)
    df_pop = pd.read_excel(population_url)
    df_communes = pd.read_csv(communes_url, sep=",", low_memory=False)

    # Prepare communes coordinates
    df_communes_coords = df_communes[[
        "code_insee", "code_postal", "latitude_centre", "longitude_centre"
    ]]

    # Clean vacancy data
    df_vac.columns = ["code_commune", "nom_commune", "annee", "part_log_vacant", "col5", "col6", "col7", "col8"]
    df_vac = df_vac[["code_commune", "nom_commune", "annee", "part_log_vacant"]]
    df_vac = df_vac.dropna(subset=["part_log_vacant"])
    df_vac["part_log_vacant"] = pd.to_numeric(df_vac["part_log_vacant"], errors="coerce")
    df_vac = df_vac.dropna(subset=["part_log_vacant"])
    df_vac = df_vac.sort_values(by="part_log_vacant", ascending=True)

    # Clean population data
    df_pop.columns = ["objectid", "reg", "dep", "cv", "codgeo", "libgeo", "p21_pop"]
    df_pop["codgeo"] = df_pop["codgeo"].astype(str)

    # Convert types
    df_vac["code_commune"] = df_vac["code_commune"].astype(str)
    df_communes_coords.loc[:, "code_insee"] = df_communes_coords["code_insee"].astype(str)

    return df_vac, df_pop, df_communes_coords


def process_vacant_housing_data(df_vac: pd.DataFrame, df_pop: pd.DataFrame, df_communes_coords: pd.DataFrame, min_population: int = 100000) -> pd.DataFrame:
    """
    Process and merge vacant housing data.

    Parameters:
    -----------
    df_vac : DataFrame
        Vacancy data
    df_pop : DataFrame
        Population data
    df_communes_coords : DataFrame
        Communes coordinates
    min_population : int
        Minimum population threshold for communes

    Returns:
    --------
    DataFrame : Processed data with coordinates and vacancy rates
    """

    # Filter communes > min_population habitants
    df_pop_filtered = df_pop[df_pop["p21_pop"] > min_population]
    df_vac_filtered = df_vac.merge(
        df_pop_filtered[["codgeo", "p21_pop"]],
        left_on="code_commune",
        right_on="codgeo",
        how="inner"
    )

    # Keep most recent year per commune
    df_vac_recent = df_vac_filtered.sort_values("annee", ascending=False) \
                                  .groupby("code_commune", as_index=False) \
                                  .first()

    # Merge coordinates
    df_vac_map = df_vac_recent.merge(
        df_communes_coords,
        left_on="code_commune",
        right_on="code_insee",
        how="left"
    ).dropna(subset=["latitude_centre", "longitude_centre"])

    return df_vac_map


def create_vacancy_map(df_vac_map: pd.DataFrame) -> go.Figure:
    """
    Create an interactive Plotly map of vacant housing rates.

    Parameters:
    -----------
    df_vac_map : DataFrame
        Vacancy data with coordinates

    Returns:
    --------
    plotly Figure : Interactive map
    """

    fig = px.scatter_mapbox(
        df_vac_map,
        lat="latitude_centre",
        lon="longitude_centre",
        size="part_log_vacant",
        color="part_log_vacant",
        hover_name="nom_commune",
        hover_data={"p21_pop": True, "part_log_vacant": True, "code_postal": True, "annee": True},
        zoom=5,
        height=500,
        color_continuous_scale="OrRd"
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    return fig


def create_vacancy_histogram(df_vac_map: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Create a histogram of communes with lowest vacancy rates.

    Parameters:
    -----------
    df_vac_map : DataFrame
        Vacancy data with coordinates
    top_n : int
        Number of top communes to show

    Returns:
    --------
    plotly Figure : Histogram
    """

    top_low_vac = df_vac_map.sort_values("part_log_vacant", ascending=True).head(top_n)
    fig_hist = px.bar(
        top_low_vac,
        x="nom_commune",
        y="part_log_vacant",
        text="part_log_vacant",
        hover_data={"p21_pop": True, "code_postal": True, "annee": True},
        labels={"part_log_vacant": "Vacant housing rate", "nom_commune": "Commune"},
        title=f"Top {top_n} communes with the lowest vacant housing rates",
        color="part_log_vacant",
        color_continuous_scale="Blues",
        height=600
    )
    fig_hist.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    return fig_hist


def print_vacancy_summary(df_vac_map: pd.DataFrame, top_n: int = 20) -> None:
    """
    Print summary statistics for vacancy analysis.

    Parameters:
    -----------
    df_vac_map : DataFrame
        Vacancy data with coordinates
    top_n : int
        Number for summary display
    """

    mean_vacancy = df_vac_map["part_log_vacant"].mean()
    top_low_vac = df_vac_map.sort_values("part_log_vacant", ascending=True).head(top_n)
    commune_min = top_low_vac.iloc[0]["nom_commune"]
    vac_min = top_low_vac.iloc[0]["part_log_vacant"]

    print(f"On average, large French communes have a vacancy rate of about {mean_vacancy:.2f}%.")
    print(f"The commune with the **lowest rate** of vacant housing is {commune_min}, with only {vac_min:.2f}% vacancy.")
    print("These cities are generally very dynamic and sought after: investing in these areas limits the risks of rental vacancy.")
    print("Conversely, the reddest points on the map indicate cities where supply exceeds demand, which can hinder short-term profitability.")


def analyze_vacant_housing(min_population: int = 100000, top_n: int = 20) -> Dict[str, Any]:
    """
    Complete vacant housing analysis pipeline.

    Parameters:
    -----------
    min_population : int
        Minimum population threshold for communes
    top_n : int
        Number of top communes to display

    Returns:
    --------
    dict : Analysis results including plots and summary data
    """

    print("=== VACANT HOUSING ANALYSIS IN FRANCE ===")

    # Load and prepare data
    df_vac, df_pop, df_communes_coords = load_vacant_housing_data()

    # Process data
    df_vac_processed = process_vacant_housing_data(df_vac, df_pop, df_communes_coords, min_population)

    # Create visualizations
    vacancy_map = create_vacancy_map(df_vac_processed)
    vacancy_hist = create_vacancy_histogram(df_vac_processed, top_n)

    # Show results
    vacancy_map.show()
    vacancy_hist.show()

    # Print summary
    print_vacancy_summary(df_vac_processed, top_n)

    results = {
        'processed_data': df_vac_processed,
        'vacancy_map': vacancy_map,
        'vacancy_histogram': vacancy_hist,
        'mean_vacancy_rate': df_vac_processed["part_log_vacant"].mean()
    }
