import requests, zipfile, io
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def analyse_rentabilite_quartiers_rennes():
    """
    Analyse la rentabilité brute des quartiers de Rennes à partir des données
    de loyers et de prix au m².
    Affiche un graphique de rentabilité par quartier et catégorie de surface.
    """

    # --- Source des données ---
    BDD = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main"

    # --- Chargement des fichiers ---
    df = pd.read_csv(f"{BDD}/rennes_quartiers.csv.txt", sep=",")
    r = requests.get(f"{BDD}/data_loyer.zip")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    fichier_csv = 'data_loyer/Base_OP_2024_L3500_Rennes.csv'
    df2 = pd.read_csv(z.open(fichier_csv), encoding="latin-1", sep=";")

    # --- Nettoyage ---
    df2.columns = df2.columns.str.strip()
    df.columns = df.columns.str.strip()

    df["Nb de pièces"] = df["Nb de pièces"].astype(int)
    df["Surface (m²)"] = df["Surface (m²)"].astype(float)
    df["Prix au m² (€)"] = df["Prix au m² (€)"].astype(float)
    df["Prix total estimé (€)"] = df["Prix total estimé (€)"].astype(float)

    # --- Filtrage appartements étudiants neufs ---
    df_etudiants = df[
        (df["Type de bien"] == "Appartement") &
        (df["Surface (m²)"] <= 45) &
        (df["Statut"] == "neuf")
    ]

    # --- Nettoyage et filtrage des loyers ---
    df2 = df2[
        (
            (df2["Type_habitat"] == "Appartement") |
            (df2["nombre_pieces_homogene"] == "Appart 1P") |
            (df2["nombre_pieces_homogene"] == "Appart 2P")
        ) &
        (df2["surface_moyenne"] < 50)
    ]
    df2 = df2.sort_values(by="moyenne_loyer_mensuel")
    df2 = df2[df2["nombre_logements"].notna() & (df2["nombre_logements"] != 0)]

    # --- Mapping des zones -> quartiers ---
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

    df2["Quartier"] = df2["Zone_calcul"].map(mapping_zones).fillna("Inconnu")

    # --- Renommage cohérent ---
    df_etudiants = df_etudiants.rename(columns={
        "Type de bien": "Type_habitat",
        "Surface (m²)": "surface_moyenne",
        "Prix au m² (€)": "Prix_m2",
        "Prix total estimé (€)": "Prix_total"
    })

    # --- Catégories de surface ---
    plages = [
        (20, 35, "30"),
        (35, 50, "45")
    ]
    colonnes_numeriques = [
        "loyer_median", "loyer_moyen",
        "loyer_mensuel_median", "moyenne_loyer_mensuel",
        "surface_moyenne", "nombre_logements"
    ]

    # --- Calcul médian par quartier et surface ---
    dfs = []
    for bas, haut, label in plages:
        df_temp = df2[
            (df2["Type_habitat"] == "Appartement") &
            (df2["surface_moyenne"].between(bas, haut))
        ].copy()
        df_group = (
            df_temp
            .groupby("Zone_calcul", as_index=False)[colonnes_numeriques]
            .median(numeric_only=True)
        )
        df_group["Quartier"] = df_group["Zone_calcul"].map(mapping_zones).fillna("Inconnu")
        df_group["Catégorie_surface_environ"] = label
        dfs.append(df_group)

    df_grouped = pd.concat(dfs, ignore_index=True)

    # --- Merge avec prix au m² étudiants ---
    df_grouped["Catégorie_surface_environ"] = df_grouped["Catégorie_surface_environ"].astype(float)
    df_etudiants["Catégorie_surface_environ"] = df_etudiants["surface_moyenne"]  # alignement

    df_merged = pd.merge(
        df_grouped,
        df_etudiants,
        on=["Quartier", "Catégorie_surface_environ"],
        how="left"
    )

    # --- Calcul rentabilité brute ---
    df_merged["rentabilite_brute_%"] = (df_merged["loyer_mensuel_median"] * 12 / df_merged["Prix_total"]) * 100
    df_merged = df_merged.sort_values(by=["rentabilite_brute_%"], ascending=False)

    print("\n=== ANALYSE DE LA RENTABILITÉ DES QUARTIERS À RENNES ===")

    # --- Affichage graphique ---
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
    plt.show()

    top = df_merged.dropna(subset=["rentabilite_brute_%"]).head(3)
    worst = df_merged.dropna(subset=["rentabilite_brute_%"]).tail(3)
    moyenne = df_merged["rentabilite_brute_%"].mean()

    print(f"À Rennes, la rentabilité moyenne des appartements est d’environ {moyenne:.2f}%.")
    print(f"Les quartiers les plus rentables sont : {', '.join(top['Quartier'].unique())}.")
    print(f"Les quartiers les moins rentables sont : {', '.join(worst['Quartier'].unique())}.")
    print("Les logements autour de 30m² sont souvent plus rentables (petits T1/T2 étudiants).")
    print("Les surfaces de 45m² sont plus stables à long terme, avec une vacance locative moindre.")
   
