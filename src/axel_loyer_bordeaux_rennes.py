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
script_dir = Path(__file__).parent  # src
project_root = script_dir.parent  # project root (since this file is directly in src/)
data_dir = project_root / "data"

def analyse_loyer_bordeaux_rennes():
    """
    Analyse et affiche l'évolution du loyer moyen au m² (1 pièce)
    pour Bordeaux et Rennes entre 2020 et 2024.
    """


    url = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/data_loyer_aglo.zip"
    response = requests.get(url)

    # Create folder in project data directory
    extract_dir = data_dir / "data_loyer_aglo"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(extract_dir)

    if "data_loyer_aglo" in os.listdir(extract_dir):
        folder = os.path.join(extract_dir, "data_loyer_aglo")
    else:
        folder = extract_dir

    def read_csv_safe(path):
        for enc in ["utf-8", "ISO-8859-1", "cp1252"]:
            try:
                return pd.read_csv(path, sep=";", encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Impossible de lire le fichier {path} avec les encodages standards.")

    # lecture fichiers
    all_data = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            try:
                year = int(file.split("_")[2])
            except (IndexError, ValueError):
                continue
            df = read_csv_safe(os.path.join(folder, file))
            df["Annee"] = year
            all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    #filtrage données
    col_ville = "agglomeration"
    col_loyer = "loyer_moyen"
    col_piece = "nombre_pieces_homogene"

    df_filtered = df_all[
        df_all[col_ville].str.contains("Bordeaux|Rennes", case=False, na=False)
        & df_all[col_piece].astype(str).str.contains("1", na=False)
    ]

    df_filtered.loc[:, col_loyer] = (
        df_filtered[col_loyer]
        .astype(str)
        .str.replace(",", ".")
        .str.replace(" ", "")
        .astype(float)
    )

    #moyenne par année +par ville
    df_grouped = (
        df_filtered.groupby(["Annee", col_ville])[col_loyer]
        .mean()
        .reset_index()
    )

    print("\n======================  CONTEXTE DU CHOIX  ====================== :\n")
    print("Après analyse des grandes villes françaises, deux se distinguent pour Léa : Rennes et Bordeaux.")
    print("Nous allons comparer leur évolution des loyers pour déterminer laquelle répond le mieux à son besoins.")
    print("A savoir une rentabilité rapide, une stabilité du marché et un faible risque de vacance.\n")

   
    plt.figure(figsize=(10, 6))
    for city in df_grouped[col_ville].unique():
        subset = df_grouped[df_grouped[col_ville] == city]
        plt.plot(subset["Annee"], subset[col_loyer], marker='o', linewidth=2, label=city)

    plt.title("Évolution du loyer moyen au m² (1 pièce) - Bordeaux vs Rennes (2020–2024)")
    plt.xlabel("Année")
    plt.ylabel("Loyer moyen (€/m²)")
    plt.xticks(sorted(df_grouped["Annee"].unique()))
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\n======================  ANALYSE ET INTERPRÉTATION  ====================== :\n")
    print("Entre 2020 et 2024, les loyers ont augmenté d’environ 9 % à Bordeaux et Rennes.")
    print("En 2024 : 16,3 €/m² à Bordeaux contre 15,9 €/m² à Rennes.")
    print("Deux courbes similaires : progression régulière, marché tendu dans les deux villes.")
    print()
    print("Bordeaux : loyers plus élevés, coût d’achat plus important, marché plus concurrentiel.")
    print("Rennes : marché plus accessible, stable, forte demande étudiante, risque moindre de vacance.")
    print()
    print("De plus, comme vu précédemment, la rentabilité moyenne légèrement supérieure à Rennes (2,98 % contre 2,31 %).")
    print()
    print("Conclusion : Rennes combine loyers modérés, stabilité et rendement correct —")
    print("un choix plus sûr et cohérent pour Léa.")
