# 🏠 Projet Data Science — Analyse des Données Foncières pour l’Investissement Immobilier

## 🎯 Contexte

Ce projet a pour objectif de mettre en pratique les compétences acquises en **data science** en développant une **solution d’aide à la décision pour un investisseur immobilier**.  
À partir des **données foncières publiques françaises** (issues de [data.gouv.fr](https://www.data.gouv.fr)), l’idée est d’explorer, d’analyser et de visualiser les tendances du marché immobilier afin de formuler des **recommandations pertinentes d’investissement**.

---

## 🎓 Persona cible

**Persona choisi :** *Léa, jeune investisseuse étudiante*

**Profil :**
- 👩 24 ans, diplômée de l'EM Lyon
- 💼 Première expérience professionnelle après 2 ans d'alternance
- 💰 Aide parentale pour le financement + épargne personnelle (~15 000 €)
- 🎯 Objectif : réaliser un **premier investissement locatif** dans une **ville étudiante dynamique**

Léa souhaite trouver le **meilleur investissement locatif étudiant** possible avec un **budget global de 200 000 €**, en analysant la rentabilité brute dans les **principales villes étudiantes françaises** (studios et T1 ≤45m²).

Elle recherche un outil interactif pour :
- Évaluer le **taux de vacance locative** en France pour anticiper les périodes creuses (notamment l'été où les étudiants quittent les logements)
- Visualiser les **villes à forte concentration étudiante** en France
- Analyser l'**évolution du prix au m² à l'achat et des loyers étudiants** en France
- Etudier **la rentabilité moyenne en France** en 2024
- Analyser la **dynamique du marché immobilier local : croissance ou baisse des prix et loyers sur les 5 dernières années** (entre Rennes et Bordeaux)
- Comparer **les quartiers les plus rentables (rentabilité brute)**
- Analyser la **localisation/nombre des transports en commun** pour identifier les zones les plus attractives pour les étudiants
- Analyser **la localisation des universités/grandes écoles**
- Analyser les **quartiers vivants (nombre de resto, bars, et supermarchés)**
- Fournir une **recommandation finale : "où investir avec 200k€ ?"**

### 💰 Hypothèses financières

| Élément | Montant estimé |
|----------|----------------|
| Prix d'achat visé | 160 000 – 180 000 € |
| Apport personnel | 15 000 € |
| Prêt immobilier estimé | 180 000 € sur 20 ans |
| Budget total (frais inclus) | **≈ 200 000 €** |
| Objectif de rentabilité brute | **≥ 5 %** |

---

### 🧭 Objectifs du projet

Créer un outil interactif permettant à Léa de :

1. Analyser la **rentabilité locative brute** pour appartements étudiants ≤45m² dans **23 grandes villes françaises**
2. Explorer visuellement les **villes à forte concentration étudiante** et analyser les **taux de vacance locative**
3. Obtenir un **classement des villes** par rentabilité, prix et loyers pour décider où investir avec 200k€

**Étapes détaillées :**

1. **Collecte et préparation des données**
   - Téléchargement et filtrage des données foncières DVF pour les 23 villes étudiantes françaises
   - Traitement des données de loyers (Observatoire des Loyers)
   - Nettoyage, agrégation par ville et calcul des prix/locations moyens

2. **Analyse de rentabilité**
   - Calcul de la rentabilité brute = (Loyer annuel / Prix d'achat) × 100
   - Fusion des données achats et loyers
   - Classement des villes par rentabilité décroissante

3. **Visualisation**
   - Graphiques de rentabilité et prix par ville
   - Cartes interactives des concentrations étudiantes et taux de vacance
   - Tableaux de bord interactifs pour l'exploration

4. **Recommandations d'investissement**
   - Identification des villes les plus rentables avec un budget 200k€
   - Analyse des risques (vacance locative, tendance marché)
   - Focus sur les appartements ≤45m² adaptés aux étudiants

---

## 📊 Données utilisées

**Source principale :**  
- [Demandes de Valeurs Foncières (DVF)](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/)

**Données complémentaires (optionnelles) :**
- Population & démographie : INSEE  
- Revenus médians & fiscalité locale : data.gouv.fr  
- Transports & attractivité régionale : API Geo, data transports  

---

## 🛠️ Technologies et Librairies utilisées

- **Langage :** Python  
- **Outils :** Jupyter Notebook  
- **Librairies principales :**
  - `pandas` → manipulation et nettoyage de données  
  - `matplotlib` / `seaborn` → visualisations statistiques  
  - `folium` ou `plotly` → cartes interactives  
  - `numpy` → calculs numériques  
  - `ipywidgets` → interactivité (filtres dynamiques)
