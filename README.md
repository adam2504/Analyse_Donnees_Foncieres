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

---

## 🧩 Objectifs du projet

1. **Collecte et préparation des données**
   - Téléchargement des données foncières DVF (Demandes de Valeurs Foncières)
   - Nettoyage et transformation : gestion des valeurs manquantes, normalisation des types de biens, géocodage éventuel
   - Agrégation par commune et type de bien

2. **Analyse exploratoire**
   - Étude de la répartition géographique des prix
   - Identification des communes les plus dynamiques
   - Analyse des tendances temporelles des prix au m²

3. **Visualisation**
   - Graphiques d’évolution des prix par commune et par type de bien  
   - Carte interactive des prix moyens et de la rentabilité estimée  
   - Histogrammes, boxplots et heatmaps selon les variables principales  

4. **Recommandations**
   - Classement des communes selon leur potentiel d’investissement  
   - Suggestions de zones à surveiller pour les années à venir  
   - Mise en avant des types de biens les plus performants  

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
