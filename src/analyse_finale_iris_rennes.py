# analyse_iris_rennes_3d.py
import os
import pandas as pd
import geopandas as gpd
import requests
import plotly.graph_objects as go
from fiona import listlayers
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1️⃣ CHARGEMENT DES IRIS
# ===========================
def charger_iris(commune="Rennes"):
    gpkg_path = "contours-iris-pe.gpkg"
    url_gpkg = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/contours-iris-pe.gpkg"
    url_ref = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/reference_IRIS_geo2025.xlsx"

    if not os.path.exists(gpkg_path):
        r = requests.get(url_gpkg)
        with open(gpkg_path, "wb") as f:
            f.write(r.content)

    layers = listlayers(gpkg_path)
    iris_start = gpd.read_file(gpkg_path, layer=layers[0])
    iris_noms = pd.read_excel(url_ref).rename(columns={'CODE_IRIS':'code_iris'})
    iris_rennes = iris_start[iris_start['nom_commune'].str.contains(commune, case=False, na=False)].copy()
    iris_rennes = iris_rennes.merge(iris_noms[['code_iris','LIB_IRIS','LIBCOM']], on='code_iris', how='left')
    return iris_rennes

# ===========================
# 2️⃣ RÉCUPÉRATION COMMERCES
# ===========================
def recup_commerces():
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = """
    [out:json][timeout:90];
    area["name"="Rennes"]["admin_level"="8"]->.searchArea;
    (
      node["amenity"="restaurant"](area.searchArea);
      node["amenity"="fast_food"](area.searchArea);
      node["amenity"="bar"](area.searchArea);
      node["amenity"="pub"](area.searchArea);
      node["amenity"="cafe"](area.searchArea);
      node["shop"="supermarket"](area.searchArea);
      node["shop"="convenience"](area.searchArea);
      node["shop"="grocery"](area.searchArea);
    );
    out center;
    """
    r = requests.post(overpass_url, data={"data": query}, timeout=120)
    osm_data = r.json()
    commerces_list=[]
    for e in osm_data['elements']:
        tags = e.get('tags', {})
        if e['type']=='node':
            lat, lon = e['lat'], e['lon']
        elif 'center' in e:
            lat, lon = e['center']['lat'], e['center']['lon']
        else:
            continue
        if tags.get('amenity','') in ['restaurant','fast_food']:
            cat='Restaurant'
        elif tags.get('amenity','') in ['bar','pub','cafe']:
            cat='Bar/Café'
        elif tags.get('shop','') in ['supermarket','convenience','grocery']:
            cat='Supermarché'
        else:
            cat='Autre'
        commerces_list.append({'categorie':cat,'lat':lat,'lon':lon})
    commerces_gdf = gpd.GeoDataFrame(commerces_list, geometry=gpd.points_from_xy([c['lon'] for c in commerces_list],[c['lat'] for c in commerces_list]), crs="EPSG:4326")
    return commerces_gdf

# ===========================
# 3️⃣ RÉCUPÉRATION ÉTUDIANTS
# ===========================
def recup_etudiants():
    url = "https://huggingface.co/datasets/analysedonneesfoncieresdata/analyse_fonciere_data/resolve/main/fr-esr-atlas_regional-effectifs-d-etudiants-inscrits-detail_etablissements.csv"
    df = pd.read_csv(url, delimiter=';')
    df = df[df['Commune']=='Rennes'].dropna(subset=['gps'])
    df[['lat','lon']] = df['gps'].str.split(',',expand=True).astype(float)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    return gdf

# ===========================
# 4️⃣ RÉCUPÉRATION TRANSPORT
# ===========================
def recup_transports():
    overpass_url="http://overpass-api.de/api/interpreter"
    query="""
    [out:json][timeout:90];
    area["name"="Rennes"]["admin_level"="8"]->.searchArea;
    (
      node["public_transport"="stop_position"](area.searchArea);
      node["highway"="bus_stop"](area.searchArea);
      node["railway"="station"](area.searchArea);
      node["railway"="halt"](area.searchArea);
      node["railway"="subway_entrance"](area.searchArea);
    );
    out center;
    """
    r=requests.post(overpass_url, data={"data":query}, timeout=120)
    osm=r.json()
    lst=[]
    for e in osm['elements']:
        if e['type']=='node':
            lat, lon = e['lat'], e['lon']
        elif 'center' in e:
            lat, lon = e['center']['lat'], e['center']['lon']
        else:
            continue
        lst.append({'lat':lat,'lon':lon})
    gdf = gpd.GeoDataFrame(lst, geometry=gpd.points_from_xy([x['lon'] for x in lst],[x['lat'] for x in lst]), crs="EPSG:4326")
    return gdf

# ===========================
# 5️⃣ CALCUL DES DENSITÉS ET MERGE
# ===========================
def calcul_densites(iris_gdf, commerces_gdf, etudiants_gdf, transports_gdf):
    iris_proj = iris_gdf.to_crs(epsg=2154)
    
    # Commerces
    commerces_gdf = commerces_gdf.to_crs(iris_gdf.crs)
    c_sjoin = gpd.sjoin(commerces_gdf, iris_gdf[['code_iris','LIB_IRIS','geometry']], how='left', predicate='within')
    stats_com = c_sjoin.groupby('code_iris').size().reset_index(name='total_commerces')
    pivot = c_sjoin.pivot_table(index='code_iris', columns='categorie', aggfunc='size', fill_value=0).reset_index()
    stats_com = stats_com.merge(pivot,on='code_iris',how='left')
    
    # Étudiants
    etudiants_gdf = etudiants_gdf.to_crs(iris_gdf.crs)
    e_sjoin = gpd.sjoin(etudiants_gdf, iris_gdf[['code_iris','LIB_IRIS','geometry']], how='inner', predicate='within')
    stats_etud = e_sjoin.groupby('code_iris')['nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE'].sum().reset_index()
    stats_etud = stats_etud.rename(columns={'nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE':'nb_etudiants'})
    
    # Transports
    transports_gdf = transports_gdf.to_crs(iris_gdf.crs)
    t_sjoin = gpd.sjoin(transports_gdf, iris_gdf[['code_iris','LIB_IRIS','geometry']], how='left', predicate='intersects')
    stats_trans = t_sjoin.groupby('code_iris').size().reset_index(name='total_arrets')
    
    # Merge
    merged = iris_proj.merge(stats_com, on='code_iris', how='left').merge(stats_etud, on='code_iris', how='left').merge(stats_trans, on='code_iris', how='left')
    merged = merged.fillna(0)
    merged['score_densite'] = (merged.get('Restaurant',0)*2 + merged.get('Bar/Café',0)*2 + merged.get('Supermarché',0)*1.5)/merged.geometry.area*1e6
    merged['students_per_km2'] = merged['nb_etudiants']/merged.geometry.area*1e6
    merged['densite_arrets_km2'] = merged['total_arrets']/merged.geometry.area*1e6
    return merged

# ===========================
# 6️⃣ GRAPHIQUE 3D
# ===========================
def plot_3d(merged):
    fig = go.Figure(data=[go.Scatter3d(
        x=merged['students_per_km2'],
        y=merged['score_densite'],
        z=merged['densite_arrets_km2'],
        mode='markers',
        marker=dict(size=5,color=merged['students_per_km2'],colorscale='Viridis',showscale=True),
        text=merged['LIB_IRIS']
    )])
    fig.update_layout(title="Densités Étudiants - Commerces - Transports à Rennes",
                      scene=dict(xaxis_title='Étudiants/km²',
                                 yaxis_title='Commerces/km²',
                                 zaxis_title='Transports/km²'))
    fig.show()

# ===========================
# 7️⃣ PIPELINE COMPLET
# ===========================
def pipeline_3d():
    iris = charger_iris()
    commerces = recup_commerces()
    etudiants = recup_etudiants()
    transports = recup_transports()
    merged = calcul_densites(iris, commerces, etudiants, transports)
    plot_3d(merged)
    return merged
