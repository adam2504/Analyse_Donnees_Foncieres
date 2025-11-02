# ============================================================================
# ANALYSE DE LA DENSITÉ ÉTUDIANTE PAR VILLE (POPULATION > 100K)
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import plotly.express as px


# ============================================================================
# 1. Chargement des données
# ============================================================================
def charger_donnees_population():
    print("📥 Chargement des données de population communale...")
    url_df_pop_communales = (
        "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/"
        "resolve/main/POPULATION_MUNICIPALE_COMMUNES_FRANCE.xlsx"
    )

    df = pd.read_excel(url_df_pop_communales)
    df.drop(
        ['p13_pop', 'p14_pop', 'p15_pop', 'p16_pop', 'p17_pop', 'p18_pop', 'p19_pop', 'p20_pop'],
        axis=1,
        inplace=True,
    )

    df['libgeo'] = df['libgeo'].replace(
        to_replace=[r'Paris.*', r'Lyon.*', r'Marseille.*'],
        value=['Paris', 'Lyon', 'Marseille'],
        regex=True,
    )

    df_grouped = df.groupby('libgeo', as_index=False).agg({
        'objectid': 'first',
        'reg': 'first',
        'dep': 'first',
        'cv': 'first',
        'codgeo': 'first',
        'p21_pop': 'sum'
    })

    df_pop_100k = df_grouped[df_grouped['p21_pop'] > 100000]
    df_pop_100k = df_pop_100k.sort_values(by='p21_pop', ascending=False)

    print(f"✅ {len(df_pop_100k)} villes avec plus de 100k habitants chargées.")
    return df_pop_100k


def charger_donnees_etudiants():
    print("📥 Chargement des données d'enseignement supérieur...")
    url_df_enseignement_sup = (
        "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/"
        "resolve/main/fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"
    )

    df = pd.read_csv(url_df_enseignement_sup, delimiter=';')
    df = df[~df['département'].str.contains("Étranger")]

    df['dep'] = df['département'].str[:2].str.strip()

    df_treated = df.groupby('dep', as_index=False)[[
        'nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE',
        'dont femmes',
        'dont hommes'
    ]].sum()

    print("✅ Données d’enseignement supérieur nettoyées et agrégées.")
    return df_treated


# ============================================================================
# 2. Fusion et traitement
# ============================================================================
def fusionner_donnees(df_pop, df_etudiants):
    print("🔗 Fusion des données population ↔ étudiants...")
    df_joined = pd.merge(df_pop, df_etudiants, on='dep', how='inner')
    df_joined['Densité étudiante'] = (
        df_joined['nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE']
        / df_joined['p21_pop']
    )
    print(f"✅ Fusion réalisée ({len(df_joined)} correspondances trouvées).")
    return df_joined


# ============================================================================
# 3. Ajout des coordonnées géographiques
# ============================================================================
def ajouter_coordonnees(df):
    print("🌍 Ajout des coordonnées géographiques via API Geocode (peut prendre du temps)...")

    geolocator = Nominatim(user_agent="adam_geocoder_ece_paris")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    def get_coordinates(ville):
        try:
            if pd.isna(ville):
                return pd.Series([None, None])
            location = geocode(f"{ville}, France")
            if location:
                return pd.Series([location.latitude, location.longitude])
            else:
                return pd.Series([None, None])
        except Exception as e:
            print(f"⚠️ Erreur pour la ville '{ville}': {e}")
            return pd.Series([None, None])

    df[["latitude", "longitude"]] = df["libgeo"].apply(get_coordinates)
    print("✅ Coordonnées ajoutées.")
    return df


# ============================================================================
# 4. Visualisations
# ============================================================================
def generer_graphiques(df):
    print("📊 Génération des graphiques...")

    df_top_students = df.sort_values(
        by='nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE',
        ascending=False
    ).head(20)

    df_top_density = df.sort_values(by='Densité étudiante', ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    plt.bar(df_top_students['libgeo'], df_top_students[
        'nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE'])
    plt.title('Top 20 villes par nombre d’étudiants (population > 100k)')
    plt.xlabel('Villes')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Nombre d’étudiants')
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(df_top_density['libgeo'], df_top_density['Densité étudiante'])
    plt.title('Top 20 villes par densité étudiante')
    plt.xlabel('Villes')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Densité étudiante')
    plt.grid(False)
    plt.show()

    print("✅ Graphiques générés.")


def generer_carte(df):
    print("🗺️ Génération de la carte interactive...")

    fig = px.scatter_map(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="libgeo",
        hover_data=[
            "nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE",
            "Densité étudiante",
        ],
        size="Densité étudiante",
        zoom=5,
        height=600
    )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(title="Répartition géographique des étudiants en France")

    print("✅ Carte interactive prête.")
    fig.show()


# ============================================================================
# 5. Fonction principale (pipeline complet)
# ============================================================================
def analyse_densite_etudiante():
    df_pop = charger_donnees_population()
    df_etud = charger_donnees_etudiants()
    df_joined = fusionner_donnees(df_pop, df_etud)
    df_geo = ajouter_coordonnees(df_joined)
    generer_graphiques(df_geo)
    generer_carte(df_geo)

    print("🎯 Analyse terminée avec succès.")
    return df_geo
