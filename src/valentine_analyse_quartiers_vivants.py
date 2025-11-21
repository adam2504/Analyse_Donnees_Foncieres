"""
Analyse des IRIS de Rennes les plus denses en commerces vivants
================================================================
Version modulaire : affichage direct des graphiques sans sauvegarde

Utilise les modules modulaires:
- loaders/ : chargement des données IRIS et OSM
- analyses/analyse_commerces.py : logique d'analyse des commerces
- viz/widgets_commerces.py : visualisation des résultats
"""

import sys
from pathlib import Path

# Add parent directory to path for relative imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

from src.loaders import load_iris_rennes
from src.loaders.osm_loader import load_commercial_establishments
from src.analyses.analyse_commerces import analyze_commerce_density
from src.viz.widgets_commerces import create_original_graph1, create_original_graph2


def analyse_commerces_rennes():
    """
    Analyse modulaire des commerces vivants à Rennes utilisant les modules spécialisés.
    """
    print("=" * 80)
    print("ANALYSE MODULAIRE DES IRIS - COMMERCES VIVANTS A RENNES")
    print("=" * 80)

    # ============================================================================
    # CHARGEMENT DES DONNÉES VIA MODULES
    # ============================================================================
    print("\nCHARGEMENT DES DONNÉES")
    print("-" * 60)

    try:
        # Chargement des IRIS via le module dédié
        iris_gdf = load_iris_rennes()

        # Chargement des commerces via le module OSM
        commerces_gdf = load_commercial_establishments()

        print("✅ Données chargées avec succès")

    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        return None, None, None

    # ============================================================================
    # ANALYSE VIA MODULE SPÉCIALISÉ
    # ============================================================================
    print("\nANALYSE DES DENSITÉS COMMERCIALES")
    print("-" * 60)

    try:
        # Utilisation du module d'analyse spécialisé
        results = analyze_commerce_density(iris_gdf, commerces_gdf)

        print("✅ Analyse terminée")
        print(f"📊 {results['summary'].iloc[0]['total_establishments']} établissements analysés")
        print(f"🏆 Score de densité moyen: {results['summary'].iloc[0]['avg_density_score']:.1f}")

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse : {e}")
        return None, None, None

    # ============================================================================
    # VISUALISATION VIA MODULES SPÉCIALISÉS
    # ============================================================================
    print("\nGÉNÉRATION DES VISUALISATIONS")
    print("-" * 60)

    try:
        # Création des graphiques originaux via les fonctions backward-compatibility
        fig1, ax1 = create_original_graph1(results)
        fig2, ax2 = create_original_graph2(results)

        print("✅ Graphiques générés")

        return fig1, fig2, results['stats']

    except Exception as e:
        print(f"❌ Erreur lors de la génération des graphiques : {e}")
        return None, None, None


if __name__ == "__main__":
    # Lancement de l'analyse modulaire
    fig1, fig2, stats_iris = analyse_commerces_rennes()

    if fig1 is not None and fig2 is not None:
        print("\n" + "=" * 80)
        print("ANALYSE TERMINÉE - Graphiques disponibles dans les variables fig1, fig2")
        print("=" * 80)
    else:
        print("\n❌ L'analyse a échoué")
