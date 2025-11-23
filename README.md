# Real Estate Data Analysis - Student Rental Investment

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/adam2504/Analyse_Donnees_Foncieres)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)

A real estate data analysis project specializing in student rental investment in France, with a particular focus on the city of Rennes.

## 🎯 Project Context

This project addresses the needs of a young investor (Léa, 24 years old) seeking to make her first rental investment in a dynamic university city. The objective is to identify the best investment opportunities in studios and T1 apartments of ≤45m² with a total budget of 200,000 €.

## 📊 Analyses Conducted

### National View (France)
- **Square Meter Price Evolution** : Analysis of real estate trends over 5 years
- **Gross Profitability** : Evaluation of rental performance by city
- **Student Concentration** : Mapping of university cities
- **Rental Vacancy Rate** : Anticipation of periods with low occupancy

### Rennes Focus
- **Profitability by District** : Comparison of rental performances
- **Student Density** : Interactive maps of university concentration
- **Public Transportation** : Analysis of accessibility and attractiveness
- **Lively Districts** : Density of businesses (restaurants, bars, supermarkets)
- **Integrated 3D Analysis** : Combined visualization of all criteria

## 🏗️ Project Architecture

```
Analyse_Donnees_Foncieres/
├── src/                          # Python analysis scripts
│   ├── adam_analyse_*.py         # Student concentration analyses
│   ├── lucien_*.py              # Profitability and vacancy analyses
│   ├── valentine_*.py           # Lively districts analyses
│   ├── axel_*.py                # Comparative analyses and functions
│   └── analyse_finale_*.py      # Integrated analysis pipeline
├── notebooks/                   # Jupyter notebooks
│   ├── persona_interactive_notebook.ipynb  # Complete workflow
│   └── raw_exploratory_notebook.ipynb      # Raw explorations
├── data/                        # Local data (if present)
├── outputs/                     # Results and visualizations
├── requirements.txt             # Python dependencies
├── Data_Sources.txt          # Data sources description
└── README.md                    # This file
```

## 📦 Installation and Configuration

### Prerequisites
- Python 3.8+
- pip for package management

### Installation
```bash
# Clone the repository
git clone https://github.com/adam2504/Analyse_Donnees_Foncieres.git
cd Analyse_Donnees_Foncieres

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies
- `pandas` - Data manipulation
- `geopandas` - Geospatial analysis
- `plotly` - Interactive visualizations
- `matplotlib` / `seaborn` - Static charts
- `requests` - Data downloading
- `osmnx` - OpenStreetMap data

## 🚀 Usage

### Complete Workflow (Léa Persona)
The main notebook `notebooks/persona_interactive_notebook.ipynb` runs all analyses:

```python
# Example usage of a specific analysis
from src.adam_analyse_concentration_etudiante_carte_rennes import analyse_concentration_rennes

# Generate the interactive map of student concentration
fig, iris_plot, df_rennes = analyse_concentration_rennes()
fig.show()
```

### Individual Analyses

#### 1. National Profitability
```python
from src.axel_valentine_rentabilite_functions import analyse_complete_rentabilite
resultats = analyse_complete_rentabilite()
```

#### 2. Student Density France
```python
from src.adam_analyse_etudiante import analyse_densite_etudiante
df_resultat = analyse_densite_etudiante()
```

#### 3. Vacant Housing
```python
from src.lucien_analyse_logements_vacants import analyse_logements_vacants
analyse_logements_vacants()
```

#### 4. Rennes/Bordeaux Comparison
```python
from src.axel_loyer_bordeaux_rennes import analyse_loyer_bordeaux_rennes
analyse_loyer_bordeaux_rennes()
```

#### 5. Profitability by Rennes District
```python
from src.lucien_rentabilité_quartiers_rennes import analyse_rentabilite_quartiers_rennes
analyse_rentabilite_quartiers_rennes()
```

#### 6. Student Concentration Rennes
```python
from src.adam_analyse_concentration_etudiante_carte_rennes import analyse_concentration_rennes
fig, iris_plot, df_rennes = analyse_concentration_rennes()
```

#### 7. Rennes Transportation
```python
from src.adam_analyse_concentration_transports_rennes import analyse_transports_rennes
fig1, fig2, stats_iris = analyse_transports_rennes()
```

#### 8. Rennes Businesses
```python
from src.valentine_analyse_quartiers_vivants import analyse_commerces_rennes
fig1, fig2, stats_iris = analyse_commerces_rennes()
```

#### 9. Integrated 3D Analysis
```python
from src.analyse_finale_iris_rennes import pipeline_3d
merged_data = pipeline_3d()
```

## 📊 Data Sources

- **DVF (Demands for Land Values)** : Geolocationed real estate transactions
- **Higher Education Data** : Number of students per institution
- **Municipal Population** : INSEE demographic data
- **OpenStreetMap** : Geographic data via Overpass API
- **IRIS Contours** : INSEE territorial subdivision

All sources are documented in `Data_Sources.txt`.

## 🎨 Visualizations

The project generates several types of visualizations:

- **Interactive Maps** : Student concentration and businesses (Plotly)
- **3D Charts** : Integrated multi-criteria analysis
- **Barplots** : Profitability by district
- **Choropleth Maps** : Densities by IRIS
- **Dashboards** : Interactive widgets for exploration

## 📈 Key Results

### Investment Recommendations
- **Target Budget** : 160,000 - 180,000 € purchase
- **Targeted Profitability** : ≥ 2.5% gross
- **Focus** : Studios/T1 ≤45m² in dense student areas

### Rennes Strengths
- Significant student concentration
- Dynamic rental market
- Good transportation accessibility
- Attractive commercial districts

## 👥 Authors

- **Adam JOUINI** - Student concentration analyses
- **Lucien RIVAT** - Profitability and rental vacancy analyses
- **Valentine MELLONE** - Lively districts analyses
- **Axel THOUMYRE** - Comparative analyses and utility functions

## 🙏 Acknowledgments

- French government open data (data.gouv.fr)
- OpenStreetMap community
- INSEE for territorial data
- Hugging Face for dataset hosting

---

**Note** : This project was developed as part of an academic data analysis project. The analyses provide insights but do not constitute personalized financial advice.
