# 🏠 Projet Data Science — Analyse des Données Foncières pour l’Investissement Immobilier

## 🎯 Contexte

Ce projet a pour objectif de mettre en pratique les compétences acquises en **data science** en développant une **solution d’aide à la décision pour un investisseur immobilier**.  
À partir des **données foncières publiques françaises** (issues de [data.gouv.fr](https://www.data.gouv.fr)), l’idée est d’explorer, d’analyser et de visualiser les tendances du marché immobilier afin de formuler des **recommandations pertinentes d’investissement**.

---

## 👤 Persona cible

**Persona choisi :** *Julien, investisseur locatif en région parisienne*  

Julien souhaite identifier les **villes d’Île-de-France les plus rentables** pour investir dans un bien locatif.  
Il recherche un outil interactif lui permettant de :
- Visualiser les **tendances de prix** dans les communes franciliennes  
- Identifier les **zones à forte rentabilité locative**  
- Comparer les **types de biens les plus attractifs (studios, T2, maisons, etc.)**  
- Déterminer les **zones à éviter (prix trop élevés, faible rendement)**  

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
