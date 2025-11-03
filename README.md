# Analyse Données Foncières - Investissement Locatif Étudiant

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/adam2504/Analyse_Donnees_Foncieres)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Un projet d'analyse de données immobilières spécialisé dans l'investissement locatif étudiant en France, avec un focus particulier sur la ville de Rennes.

## 🎯 Contexte du Projet

Ce projet répond aux besoins d'une jeune investisseuse (Léa, 24 ans) cherchant à réaliser son premier investissement locatif dans une ville étudiante dynamique. L'objectif est d'identifier les meilleures opportunités d'investissement dans des studios et T1 de ≤45m² avec un budget total de 200 000 €.

## 📊 Analyses Réalisées

### Vue Nationale (France)
- **Évolution des prix au m²** : Analyse des tendances immobilières sur 5 ans
- **Rentabilité brute** : Évaluation de la performance locative par ville
- **Concentration étudiante** : Cartographie des villes universitaires
- **Taux de vacance locative** : Anticipation des périodes de faible occupation

### Focus Rennes
- **Rentabilité par quartier** : Comparaison des performances locatives
- **Densité étudiante** : Cartes interactives de concentration universitaire
- **Transports en commun** : Analyse de l'accessibilité et attractivité
- **Quartiers vivants** : Densité de commerces (restaurants, bars, supermarchés)
- **Analyse 3D intégrée** : Visualisation combinée de tous les critères

## 🏗️ Architecture du Projet

```
Analyse_Donnees_Foncieres/
├── src/                          # Scripts d'analyse Python
│   ├── adam_analyse_*.py         # Analyses de concentration étudiante
│   ├── lucien_*.py              # Analyses de rentabilité et vacance
│   ├── valentine_*.py           # Analyses de quartiers vivants
│   ├── axel_*.py                # Analyses comparatives et fonctions
│   └── analyse_finale_*.py      # Pipeline d'analyse intégrée
├── notebooks/                   # Notebooks Jupyter
│   ├── persona_interactive_notebook.ipynb  # Workflow complet
│   └── raw_exploratory_notebook.ipynb      # Explorations brutes
├── data/                        # Données locales (si présentes)
├── outputs/                     # Résultats et visualisations
├── requirements.txt             # Dépendances Python
├── Sources_Donnees.txt          # Description des sources de données
└── README.md                    # Ce fichier
```

## 📦 Installation et Configuration

### Prérequis
- Python 3.8+
- pip pour la gestion des packages

### Installation
```bash
# Cloner le repository
git clone https://github.com/adam2504/Analyse_Donnees_Foncieres.git
cd Analyse_Donnees_Foncieres

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances Principales
- `pandas` - Manipulation de données
- `geopandas` - Analyse géospatiale
- `plotly` - Visualisations interactives
- `matplotlib` / `seaborn` - Graphiques statiques
- `requests` - Téléchargement de données
- `osmnx` - Données OpenStreetMap

## 🚀 Utilisation

### Workflow Complet (Persona Léa)
Le notebook principal `notebooks/persona_interactive_notebook.ipynb` exécute toutes les analyses :

```python
# Exemple d'utilisation d'une analyse spécifique
from src.adam_analyse_concentration_etudiante_carte_rennes import analyse_concentration_rennes

# Générer la carte interactive de concentration étudiante
fig, iris_plot, df_rennes = analyse_concentration_rennes()
fig.show()
```

### Analyses Individuelles

#### 1. Rentabilité Nationale
```python
from src.axel_valentine_rentabilite_functions import analyse_complete_rentabilite
resultats = analyse_complete_rentabilite()
```

#### 2. Densité Étudiante France
```python
from src.adam_analyse_etudiante import analyse_densite_etudiante
df_resultat = analyse_densite_etudiante()
```

#### 3. Logements Vacants
```python
from src.lucien_analyse_logements_vacants import analyse_logements_vacants
analyse_logements_vacants()
```

#### 4. Comparaison Rennes/Bordeaux
```python
from src.axel_loyer_bordeaux_rennes import analyse_loyer_bordeaux_rennes
analyse_loyer_bordeaux_rennes()
```

#### 5. Rentabilité par Quartier Rennes
```python
from src.lucien_rentabilité_quartiers_rennes import analyse_rentabilite_quartiers_rennes
analyse_rentabilite_quartiers_rennes()
```

#### 6. Concentration Étudiante Rennes
```python
from src.adam_analyse_concentration_etudiante_carte_rennes import analyse_concentration_rennes
fig, iris_plot, df_rennes = analyse_concentration_rennes()
```

#### 7. Transports Rennes
```python
from src.adam_analyse_concentration_transports_rennes import analyse_transports_rennes
fig1, fig2, stats_iris = analyse_transports_rennes()
```

#### 8. Commerces Rennes
```python
from src.valentine_analyse_quartiers_vivants import analyse_commerces_rennes
fig1, fig2, stats_iris = analyse_commerces_rennes()
```

#### 9. Analyse Intégrée 3D
```python
from src.analyse_finale_iris_rennes import pipeline_3d
merged_data = pipeline_3d()
```

## 📊 Sources de Données

- **DVF (Demandes de Valeurs Foncières)** : Transactions immobilières géolocalisées
- **Data Éducation Supérieure** : Effectifs étudiants par établissement
- **Population Communale** : Données démographiques INSEE
- **OpenStreetMap** : Données géographiques via Overpass API
- **Contours IRIS** : Découpage territorial INSEE

Toutes les sources sont documentées dans `Sources_Donnees.txt`.

## 🎨 Visualisations

Le projet génère plusieurs types de visualisations :

- **Cartes interactives** : Concentration étudiante et commerces (Plotly)
- **Graphiques 3D** : Analyse intégrée multi-critères
- **Barplots** : Rentabilité par quartier
- **Cartes choroplèthes** : Densités par IRIS
- **Tableaux de bord** : Widgets interactifs pour exploration

## 📈 Résultats Clés

### Recommandations d'Investissement
- **Budget cible** : 160 000 - 180 000 € d'achat
- **Rentabilité visée** : ≥ 2,5 % brut
- **Focus** : Studios/T1 ≤45m² dans zones étudiantes denses

### Points Forts Rennes
- Concentration étudiante importante
- Marché locatif dynamique
- Bonne accessibilité transports
- Quartiers commerçants attractifs

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Auteurs

- **Adam JOUINI** - Analyses de concentration étudiante
- **Lucien RIVAT** - Analyses de rentabilité et vacance locative
- **Valentine MELLONE** - Analyses de quartiers vivants
- **Axel THOUMYRE** - Analyses comparatives et fonctions utilitaires

## 🙏 Remerciements

- Données ouvertes du gouvernement français (data.gouv.fr)
- Communauté OpenStreetMap
- INSEE pour les données territoriales
- Hugging Face pour l'hébergement des datasets

---

**Note** : Ce projet est développé dans le cadre d'un projet académique d'analyse de données. Les analyses fournissent des insights mais ne constituent pas des conseils financiers personnalisés.
