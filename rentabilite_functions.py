# ============================================================================
# FONCTIONS D'ANALYSE DE RENTABILITÉ LOCATIVE
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import zipfile
import glob


def telecharger_et_extraire_dvf(
    url="https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/ValeursFoncieres-2024.txt",
    villes=None,
    surface_max=45,
    prix_min=1000,
    prix_max=100000,
    output_file="dvf_appartements_nettoye.csv"
):
    """
    Télécharge et extrait les données DVF pour les villes spécifiées.
    
    Paramètres:
    -----------
    url : str
        URL du fichier DVF
    villes : list
        Liste des villes à analyser
    surface_max : int
        Surface maximale en m²
    prix_min : int
        Prix minimum au m² pour filtrer les aberrations
    prix_max : int
        Prix maximum au m² pour filtrer les aberrations
    output_file : str
        Nom du fichier CSV de sortie
        
    Retourne:
    ---------
    df_clean : DataFrame
        DataFrame nettoyé avec les prix au m²
    """
    
    if villes is None:
        villes = [
            "PARIS", "LYON", "LILLE", "TOULOUSE", "BORDEAUX", "MARSEILLE", "MONTPELLIER", 
            "RENNES", "STRASBOURG", "NANTES", "GRENOBLE", "NANCY", "NICE", "ANGERS", 
            "ROUEN", "CLERMONT-FERRAND", "CAEN", "DIJON", "TOURS", "REIMS", "AJACCIO", 
            "ANNECY", "TOULON"
        ]
    
    colonnes_utiles = [
        "Date mutation", "Nature mutation", "Valeur fonciere", "Code postal",
        "Commune", "Type local", "Surface reelle bati"
    ]
    
    print("="*80)
    print("ÉTAPE 1 : EXTRACTION ET ANALYSE DES PRIX D'ACHAT (DVF 2024)")
    print("="*80)
    
    data_filtree = []
    chunks = pd.read_csv(url, sep="|", low_memory=False, chunksize=20000)
    
    for i, chunk in enumerate(chunks):
        filtre = chunk[chunk["Commune"].isin(villes)]
        filtre = filtre[
            (filtre["Type local"] == "Appartement") &
            (filtre["Nature mutation"] == "Vente")
        ]
        
        filtre["Surface reelle bati"] = pd.to_numeric(filtre["Surface reelle bati"], errors="coerce")
        filtre = filtre[filtre["Surface reelle bati"].between(1, surface_max)]
        
        filtre["Valeur fonciere"] = (
            filtre["Valeur fonciere"].astype(str).str.replace(",", ".").str.replace(" ", "")
        )
        filtre["Valeur fonciere"] = pd.to_numeric(filtre["Valeur fonciere"], errors="coerce")
        
        filtre = filtre.dropna(subset=["Valeur fonciere", "Surface reelle bati"])
        filtre = filtre[filtre["Valeur fonciere"] > 0]
        filtre = filtre[colonnes_utiles]
        
        if len(filtre) > 0:
            data_filtree.append(filtre)
    
    df = pd.concat(data_filtree, ignore_index=True)
    df["Prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]
    
    df_clean = df[(df['Prix_m2'] >= prix_min) & (df['Prix_m2'] <= prix_max)].copy()
    
    print(f"\nDonnées extraites : {len(df_clean):,} transactions après nettoyage")
    print(f"Lignes supprimées (aberrantes) : {len(df) - len(df_clean):,}")
    
    df_clean = df_clean[[
        "Commune", "Code postal", "Type local", "Surface reelle bati",
        "Valeur fonciere", "Prix_m2", "Date mutation"
    ]]
    df_clean.to_csv(output_file, index=False)
    
    return df_clean


def calculer_prix_achat_par_ville(df_clean, output_file="classement_prix_m2_par_ville.csv"):
    """Calcule les statistiques de prix d'achat par ville."""
    
    stats_villes = df_clean.groupby("Commune").agg({
        'Prix_m2': ['count', 'mean', 'min', 'max']
    }).round(0)
    
    stats_villes.columns = ['Nombre de ventes', 'Prix moyen €/m²', 'Prix min €/m²', 'Prix max €/m²']
    stats_villes = stats_villes.sort_values('Prix moyen €/m²', ascending=False).reset_index()
    stats_villes.insert(0, 'Rang', range(1, len(stats_villes) + 1))
    
    print("\n" + "="*80)
    print("CLASSEMENT DES VILLES PAR PRIX MOYEN AU M²")
    print("="*80)
    print(stats_villes.to_string(index=False))
    
    prix_moyen_national = df_clean['Prix_m2'].mean()
    print(f"\n\nPrix moyen national : {prix_moyen_national:,.0f} €/m²")
    print(f"Ville la plus chère : {stats_villes.iloc[0]['Commune']} ({stats_villes.iloc[0]['Prix moyen €/m²']:,.0f} €/m²)")
    print(f"Ville la moins chère : {stats_villes.iloc[-1]['Commune']} ({stats_villes.iloc[-1]['Prix moyen €/m²']:,.0f} €/m²)")
    
    stats_villes.to_csv(output_file, index=False)
    
    return stats_villes


def visualiser_prix_achat(stats_villes, df_clean, output_file="classement_prix_m2_villes.png"):
    """Crée un graphique des prix d'achat par ville."""
    
    stats_graph = stats_villes.sort_values('Prix moyen €/m²', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    villes_graph = stats_graph['Commune']
    prix_moyens = stats_graph['Prix moyen €/m²']
    y_pos = range(len(villes_graph))
    
    colors = plt.cm.RdYlGn_r([
        (p - prix_moyens.min()) / (prix_moyens.max() - prix_moyens.min()) 
        for p in prix_moyens
    ])
    bars = ax.barh(y_pos, prix_moyens, height=0.6, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=0.8)
    
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height()/2, 
                f'{int(prix_moyens.iloc[i]):,} €', 
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(villes_graph, fontsize=11, fontweight='bold')
    ax.set_xlabel('Prix au m² (€)', fontsize=12, fontweight='bold')
    ax.set_title('Prix moyen d\'achat au m² par ville\nAppartements ≤ 45m² (DVF 2024)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    prix_moyen_national = df_clean['Prix_m2'].mean()
    ax.axvline(x=prix_moyen_national, color='green', linestyle='--', linewidth=2, 
               label=f'Moyenne nationale: {prix_moyen_national:,.0f} €/m²', alpha=0.7)
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def extraire_loyers(
    url_zip="https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/data_loyer.zip",
    surface_max=45,
    output_file="loyers_moyens_par_ville.csv"
):
    """Extrait et analyse les données de loyers de l'Observatoire."""
    
    print("="*80)
    print("ÉTAPE 2 : EXTRACTION ET ANALYSE DES LOYERS")
    print("="*80)
    
    zip_path = "data_loyer.zip"
    dossier_extraction = "data_loyer"
    dossier_loyers = os.path.join(dossier_extraction, "data_loyer")
    
    if not os.path.exists(zip_path):
        response = requests.get(url_zip)
        with open(zip_path, "wb") as f:
            f.write(response.content)
    
    if not os.path.exists(dossier_extraction):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dossier_extraction)
    
    fichiers_loyers = glob.glob(os.path.join(dossier_loyers, "Base_OP_*.csv"))
    resultats_loyers = []
    
    for fichier in fichiers_loyers:
        try:
            nom_fichier = os.path.basename(fichier)
            parties = nom_fichier.replace('.csv', '').split('_')
            nom_ville = parties[4].upper() if len(parties) > 4 else "INCONNU"
            
            df_loyer = pd.read_csv(fichier, encoding='cp1252', sep=';', low_memory=False)
            df_loyer.columns = df_loyer.columns.str.strip()
            
            if 'loyer_moyen' in df_loyer.columns:
                df_loyer['loyer_moyen'] = (
                    df_loyer['loyer_moyen'].astype(str).str.replace(',', '.').str.replace(' ', '').replace('', None)
                )
                df_loyer['loyer_moyen'] = pd.to_numeric(df_loyer['loyer_moyen'], errors='coerce')
            
            if 'surface_moyenne' in df_loyer.columns:
                df_loyer['surface_moyenne'] = pd.to_numeric(df_loyer['surface_moyenne'], errors='coerce')
            
            if 'nombre_observations' in df_loyer.columns:
                df_loyer['nombre_observations'] = pd.to_numeric(df_loyer['nombre_observations'], errors='coerce')
            
            filtre = df_loyer[
                (
                    (df_loyer['Type_habitat'] == 'Appartement') |
                    (df_loyer['nombre_pieces_homogene'].isin(['Appart 1P', 'Appart 2P']))
                ) &
                (df_loyer['surface_moyenne'].notna()) &
                (df_loyer['surface_moyenne'] <= surface_max) &
                (df_loyer['loyer_moyen'].notna()) &
                (df_loyer['nombre_observations'].notna())
            ]
            
            if len(filtre) > 0:
                loyer_moyen_m2 = (
                    (filtre['loyer_moyen'] * filtre['nombre_observations']).sum() / 
                    filtre['nombre_observations'].sum()
                )
                
                resultats_loyers.append({
                    'Ville': nom_ville,
                    'Loyer_moyen_m2': round(loyer_moyen_m2, 2),
                    'Nombre_observations': int(filtre['nombre_observations'].sum())
                })
        
        except Exception as e:
            pass
    
    df_loyers = pd.DataFrame(resultats_loyers)
    df_loyers = df_loyers.drop_duplicates(subset=['Ville'], keep='first')
    
    mapping_villes = {
        'AIX-MARSEILLE': 'MARSEILLE',
        'AGLO-PARIS': 'PARIS',
    }
    df_loyers['Ville'] = df_loyers['Ville'].replace(mapping_villes)
    df_loyers = df_loyers.sort_values('Ville')
    
    print("\n" + "="*80)
    print("LOYERS MOYENS PAR VILLE")
    print("="*80)
    print(df_loyers.to_string(index=False))
    
    loyer_moyen_national = df_loyers['Loyer_moyen_m2'].mean()
    ville_min = df_loyers.loc[df_loyers['Loyer_moyen_m2'].idxmin(), 'Ville']
    ville_max = df_loyers.loc[df_loyers['Loyer_moyen_m2'].idxmax(), 'Ville']
    
    print(f"\n\nLoyer moyen national : {loyer_moyen_national:.2f} €/m²")
    print(f"Ville la moins chère : {ville_min} ({df_loyers['Loyer_moyen_m2'].min():.2f} €/m²)")
    print(f"Ville la plus chère : {ville_max} ({df_loyers['Loyer_moyen_m2'].max():.2f} €/m²)")
    
    df_loyers.to_csv(output_file, index=False)
    
    return df_loyers


def visualiser_loyers(df_loyers, output_file="classement_loyers_villes.png"):
    """Crée un graphique des loyers par ville."""
    
    df_loyers_graph = df_loyers.sort_values('Loyer_moyen_m2', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    villes = df_loyers_graph['Ville']
    loyers = df_loyers_graph['Loyer_moyen_m2']
    observations = df_loyers_graph['Nombre_observations']
    y_pos = range(len(villes))
    
    colors = plt.cm.RdYlGn_r([
        (l - loyers.min()) / (loyers.max() - loyers.min()) 
        for l in loyers
    ])
    bars = ax.barh(y_pos, loyers, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    for i, (bar, loyer, obs) in enumerate(zip(bars, loyers, observations)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{loyer:.2f} €/m²', 
                va='center', fontsize=10, fontweight='bold')
        ax.text(1, bar.get_y() + bar.get_height()/2, 
                f'({obs:,} obs)', 
                va='center', ha='left', fontsize=8, color='white', style='italic')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(villes, fontsize=11, fontweight='bold')
    ax.set_xlabel('Loyer moyen (€/m²)', fontsize=12, fontweight='bold')
    ax.set_title('Loyers mensuels moyens par ville\nAppartements ≤ 45m² (2024)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    loyer_moyen_national = df_loyers['Loyer_moyen_m2'].mean()
    ax.axvline(x=loyer_moyen_national, color='blue', linestyle='--', linewidth=2, 
               label=f'Moyenne nationale: {loyer_moyen_national:.2f} €/m²', alpha=0.7)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    ax.annotate(f'Ville la moins chère\n{villes.iloc[0]}', 
                xy=(loyers.iloc[0], 0), xytext=(5, -2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    ax.annotate(f'Ville la plus chère\n{villes.iloc[-1]}', 
                xy=(loyers.iloc[-1], len(villes)-1), xytext=(5, len(villes)-1),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def calculer_rentabilite(
    fichier_achat="dvf_appartements_nettoye.csv",
    fichier_loyers="loyers_moyens_par_ville.csv",
    output_file="rentabilite_locative_par_ville.csv"
):
    """Calcule la rentabilité locative brute par ville."""
    
    print("="*80)
    print("ÉTAPE 3 : CALCUL DE LA RENTABILITÉ LOCATIVE")
    print("="*80)
    
    df_achat = pd.read_csv(fichier_achat, low_memory=False)
    df_loyers = pd.read_csv(fichier_loyers)
    
    prix_achat_ville = df_achat.groupby("Commune").agg({
        'Prix_m2': ['mean', 'count']
    }).round(2)
    prix_achat_ville.columns = ['Prix_achat_moyen_m2', 'Nombre_ventes']
    prix_achat_ville = prix_achat_ville.reset_index()
    
    mapping_villes = {
        'AIX-MARSEILLE': 'MARSEILLE',
        'AGLO-PARIS': 'PARIS',
    }
    df_loyers['Ville'] = df_loyers['Ville'].replace(mapping_villes)
    prix_achat_ville['Commune'] = prix_achat_ville['Commune'].replace(mapping_villes)
    
    df_rentabilite = pd.merge(
        prix_achat_ville,
        df_loyers,
        left_on='Commune',
        right_on='Ville',
        how='inner'
    )
    
    df_rentabilite['Loyer_annuel_m2'] = (df_rentabilite['Loyer_moyen_m2'] * 12).round(2)
    df_rentabilite['Rentabilite_brute_%'] = (
        (df_rentabilite['Loyer_moyen_m2'] * 12) / df_rentabilite['Prix_achat_moyen_m2'] * 100
    ).round(2)
    
    df_rentabilite = df_rentabilite.sort_values('Rentabilite_brute_%', ascending=False)
    df_rentabilite = df_rentabilite.reset_index(drop=True)
    df_rentabilite.insert(0, 'Rang', range(1, len(df_rentabilite) + 1))
    
    print("\n" + "="*120)
    print("CLASSEMENT DES VILLES PAR RENTABILITÉ BRUTE")
    print("="*120)
    print(df_rentabilite[['Rang', 'Commune', 'Rentabilite_brute_%', 'Loyer_moyen_m2', 
                           'Loyer_annuel_m2', 'Prix_achat_moyen_m2', 'Nombre_ventes']].to_string(index=False))
    
    rent_moyenne = df_rentabilite['Rentabilite_brute_%'].mean()
    rent_max = df_rentabilite['Rentabilite_brute_%'].max()
    rent_min = df_rentabilite['Rentabilite_brute_%'].min()
    ville_max = df_rentabilite.iloc[0]['Commune']
    ville_min = df_rentabilite.iloc[-1]['Commune']
    
    print(f"\n\nRentabilité moyenne : {rent_moyenne:.2f}%")
    print(f"Ville la plus rentable : {ville_max} ({rent_max:.2f}%)")
    print(f"Ville la moins rentable : {ville_min} ({rent_min:.2f}%)")
    
    df_rentabilite.to_csv(output_file, index=False)
    
    return df_rentabilite


def visualiser_rentabilite(df_rentabilite, output_file="analyse_rentabilite_finale.png"):
    """Crée les graphiques de rentabilité (barres + scatter)."""
    
    df_rent_graph = df_rentabilite.sort_values('Rentabilite_brute_%', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Analyse de Rentabilité Locative - Appartements ≤ 45m² (2024)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Graphique 1 : Barres
    ax1 = axes[0]
    villes = df_rent_graph['Commune']
    rentabilite = df_rent_graph['Rentabilite_brute_%']
    y_pos = range(len(villes))
    
    colors = plt.cm.RdYlGn([
        (r - rentabilite.min()) / (rentabilite.max() - rentabilite.min()) 
        for r in rentabilite
    ])
    bars1 = ax1.barh(y_pos, rentabilite, color=colors, edgecolor='black', linewidth=0.8)
    
    for i, (bar, val) in enumerate(zip(bars1, rentabilite)):
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(villes, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Rentabilité brute (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Rentabilité Brute par Ville', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    rent_moyenne = df_rentabilite['Rentabilite_brute_%'].mean()
    ax1.axvline(x=rent_moyenne, color='red', linestyle='--', linewidth=2.5, alpha=0.8, 
               label=f'Moyenne: {rent_moyenne:.2f}%')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Graphique 2 : Scatter
    ax2 = axes[1]
    
    scatter = ax2.scatter(
        df_rentabilite['Prix_achat_moyen_m2'], 
        df_rentabilite['Loyer_annuel_m2'],
        s=df_rentabilite['Rentabilite_brute_%'] * 80,
        c=df_rentabilite['Rentabilite_brute_%'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )
    
    for idx, row in df_rentabilite.iterrows():
        ax2.annotate(
            row['Commune'], 
            (row['Prix_achat_moyen_m2'], row['Loyer_annuel_m2']),
            fontsize=10, ha='center', fontweight='bold', alpha=0.9
        )
    
    ax2.set_xlabel('Prix d\'achat moyen (€/m²)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Loyer annuel (€/m²)', fontsize=13, fontweight='bold')
    ax2.set_title('Loyer Annuel vs Prix d\'Achat\n(Taille des bulles = Rentabilité)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Rentabilité brute (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def analyse_complete_rentabilite(
    villes=None,
    surface_max=45,
    prix_min=1000,
    prix_max=100000
):
    """
    Fonction principale qui exécute l'analyse complète de rentabilité.
    
    Paramètres:
    -----------
    villes : list, optional
        Liste des villes à analyser (par défaut : 23 villes étudiantes)
    surface_max : int
        Surface maximale en m²
    prix_min : int
        Prix minimum au m² pour filtrer les aberrations
    prix_max : int
        Prix maximum au m² pour filtrer les aberrations
        
    Retourne:
    ---------
    dict : Dictionnaire contenant tous les DataFrames générés
    """
    
    print("\n")
    print("="*80)
    print("ANALYSE COMPLÈTE DE RENTABILITÉ LOCATIVE")
    print("Appartements étudiants ≤ 45m²")
    print("="*80)
    print("\n")
    
    # Étape 1
    df_achat = telecharger_et_extraire_dvf(
        villes=villes,
        surface_max=surface_max,
        prix_min=prix_min,
        prix_max=prix_max
    )
    stats_villes = calculer_prix_achat_par_ville(df_achat)
    visualiser_prix_achat(stats_villes, df_achat)
    
    
    # Étape 2
    df_loyers = extraire_loyers(surface_max=surface_max)
    visualiser_loyers(df_loyers)
    
    
    # Étape 3
    df_rentabilite = calculer_rentabilite()
    visualiser_rentabilite(df_rentabilite)
    
    