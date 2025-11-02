import pandas as pd
import plotly.express as px

def analyse_logements_vacants():
    """
    Analyse les logements vacants en France et affiche :
    - Une carte interactive Plotly avec le taux de vacance
    - Un histogramme des communes avec le plus faible taux
    """

    # --- Chemin de base des données ---
    BDD = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main"

    # --- Chargement des données ---
    df_vac = pd.read_excel(f"{BDD}/insee_rp_hist_1968.xlsx", header=1)
    df_pop = pd.read_excel(f"{BDD}/POPULATION_MUNICIPALE_COMMUNES_FRANCE_lucien.xlsx")
    df_communes = pd.read_csv(f"{BDD}/communes-france-2025.csv", sep=",", low_memory=False)

    # --- Préparation des données ---
    df_communes_coords = df_communes[[
        "code_insee", "code_postal", "latitude_centre", "longitude_centre"
    ]]
    df_vac.columns = ["code_commune", "nom_commune", "annee", "part_log_vacant", "col5", "col6", "col7", "col8"]
    df_vac = df_vac[["code_commune", "nom_commune", "annee", "part_log_vacant"]]
    df_pop.columns = ["objectid", "reg", "dep", "cv", "codgeo", "libgeo", "p21_pop"]

    # Nettoyage
    df_vac = df_vac.dropna(subset=["part_log_vacant"])
    df_vac["part_log_vacant"] = pd.to_numeric(df_vac["part_log_vacant"], errors="coerce")
    df_vac = df_vac.dropna(subset=["part_log_vacant"])
    df_vac = df_vac.sort_values(by="part_log_vacant", ascending=True)

    df_pop["codgeo"] = df_pop["codgeo"].astype(str)
    df_vac["code_commune"] = df_vac["code_commune"].astype(str)
    df_communes_coords.loc[:, "code_insee"] = df_communes_coords["code_insee"].astype(str)

    # Filtrage communes > 100k habitants
    df_pop_100k = df_pop[df_pop["p21_pop"] > 100000]
    df_vac_100k = df_vac.merge(
        df_pop_100k[["codgeo", "p21_pop"]],
        left_on="code_commune",
        right_on="codgeo",
        how="inner"
    )

    # Garder l’année la plus récente par commune
    df_vac_100k_recent = df_vac_100k.sort_values("annee", ascending=False) \
                                     .groupby("code_commune", as_index=False) \
                                     .first()

    # Merge coordonnées
    df_vac_map = df_vac_100k_recent.merge(
        df_communes_coords,
        left_on="code_commune",
        right_on="code_insee",
        how="left"
    ).dropna(subset=["latitude_centre", "longitude_centre"])
    
    print("=== ANALYSE DES LOGEMENTS VACANTS EN FRANCE ===")

    # --- Affichage de la carte Plotly ---
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
    fig.show()

    # --- Histogramme ---
    top10_low_vac = df_vac_map.sort_values("part_log_vacant", ascending=True).head(20)
    fig_hist = px.bar(
        top10_low_vac,
        x="nom_commune",
        y="part_log_vacant",
        text="part_log_vacant",
        hover_data={"p21_pop": True, "code_postal": True, "annee": True},
        labels={"part_log_vacant": "Taux logements vacants", "nom_commune": "Commune"},
        title="Top 20 communes avec le plus petit taux de logements vacants",
        color="part_log_vacant",
        color_continuous_scale="Blues",
        height=600
    )
    fig_hist.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_hist.show()
    
    moy_vacance = df_vac_map["part_log_vacant"].mean()
    commune_min = top10_low_vac.iloc[0]["nom_commune"]
    vac_min = top10_low_vac.iloc[0]["part_log_vacant"]
    print(f"En moyenne, les grandes communes françaises ont un taux de vacance d’environ {moy_vacance:.2f}%.")
    print(f"La commune avec le **plus faible taux** de logements vacants est {commune_min}, avec seulement {vac_min:.2f}% de vacance.")
    print("Ces villes sont généralement très dynamiques et recherchées : investir dans ces zones limite les risques de vacance locative.")
    print("À l’inverse, les points les plus rouges sur la carte indiquent des villes où l’offre dépasse la demande, ce qui peut freiner la rentabilité à court terme.")
    
