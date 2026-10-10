from pathlib import Path
import os

base_dir = Path.cwd()
matplotlib_cache = base_dir / ".matplotlib_cache"
matplotlib_cache.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.vectorized import contains

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

try:
    import hdbscan
    HDBSCAN_BACKEND = "hdbscan"
except Exception:
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
        HDBSCAN_BACKEND = "sklearn"
    except Exception:
        HDBSCAN_BACKEND = None

plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

salida_base = Path("outputs_etapa54")
salida_figuras = salida_base / "figuras"
salida_tablas = salida_base / "tablas"
salida_datasets = salida_base / "datasets"
for carpeta in [salida_figuras, salida_tablas, salida_datasets]:
    carpeta.mkdir(parents=True, exist_ok=True)

def guardar_figura(nombre):
    ruta = salida_figuras / nombre
    plt.tight_layout()
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    return ruta

def normalizar_texto(x):
    if pd.isna(x):
        return np.nan
    import unicodedata, re
    texto = str(x).strip().upper()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto if texto else np.nan

def resumen_clusters(labels):
    labels = np.asarray(labels)
    ruido = int((labels == -1).sum())
    clusters = sorted([c for c in np.unique(labels) if c != -1])
    tamanios = pd.Series(labels[labels != -1]).value_counts() if len(clusters) else pd.Series(dtype=int)
    return {
        "numero_clusters": len(clusters),
        "puntos_ruido": ruido,
        "porcentaje_ruido": round(ruido / len(labels) * 100, 2) if len(labels) else 0,
        "tamano_cluster_mas_grande": int(tamanios.max()) if not tamanios.empty else 0,
        "tamano_promedio_clusters": round(float(tamanios.mean()), 2) if not tamanios.empty else 0,
    }

def plot_clusters(gdf_m, limite_m, columna_cluster, titulo, nombre_figura):
    fig, ax = plt.subplots(figsize=(9, 9))
    limite_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#333333", linewidth=1.0)
    ruido = gdf_m[gdf_m[columna_cluster] == -1]
    clusters = gdf_m[gdf_m[columna_cluster] != -1]
    if not ruido.empty:
        ruido.plot(ax=ax, color="#BDBDBD", markersize=6, alpha=0.35, label="Ruido")
    if not clusters.empty:
        for cluster_id, datos in clusters.groupby(columna_cluster):
            datos.plot(ax=ax, markersize=7, alpha=0.65, label=f"Cluster {cluster_id}")
    ax.set_title(titulo)
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) <= 15:
        ax.legend(fontsize=8, loc="best")
    return guardar_figura(nombre_figura)

def preparar_geometria_clusters(gdf_m, columna_cluster, buffer_m=120):
    datos = gdf_m[gdf_m[columna_cluster].notna() & (gdf_m[columna_cluster] != -1)].copy()
    if datos.empty:
        return gpd.GeoDataFrame(columns=[columna_cluster, "geometry"], geometry="geometry", crs=gdf_m.crs)

    geometrias = []
    for cluster_id, grupo in datos.groupby(columna_cluster):
        union = grupo.geometry.unary_union
        hull = union.convex_hull
        geom = hull.buffer(buffer_m)
        geometrias.append({columna_cluster: int(cluster_id), "geometry": geom})
    return gpd.GeoDataFrame(geometrias, geometry="geometry", crs=gdf_m.crs)

def resumen_detallado_clusters(gdf_m, columna_cluster, nombre_tecnica):
    datos = gdf_m[gdf_m[columna_cluster].notna() & (gdf_m[columna_cluster] != -1)].copy()
    if datos.empty:
        return pd.DataFrame(columns=[
            "tecnica", "cluster", "registros", "coordenadas_unicas",
            "modalidad_predominante", "registros_modalidad_predominante", "porcentaje_modalidad_predominante",
            "tipo_predominante", "subtipo_predominante", "centroide_longitud", "centroide_latitud"
        ])

    filas = []
    datos_wgs = datos.to_crs(epsg=4326)
    for cluster_id, grupo_m in datos.groupby(columna_cluster):
        grupo_wgs = datos_wgs.loc[grupo_m.index]
        coords_unicas = grupo_wgs.geometry.apply(lambda g: (round(g.x, 7), round(g.y, 7))).nunique()

        def predominante(columna):
            if columna not in grupo_m.columns:
                return ("No disponible", 0, 0.0)
            conteo = grupo_m[columna].fillna("No especificado").astype(str).value_counts()
            if conteo.empty:
                return ("No disponible", 0, 0.0)
            valor = conteo.index[0]
            n = int(conteo.iloc[0])
            pct = round(n / len(grupo_m) * 100, 2)
            return valor, n, pct

        modalidad, n_modalidad, pct_modalidad = predominante("modalidad_hecho")
        tipo, _, _ = predominante("tipo_hecho")
        subtipo, _, _ = predominante("subtipo_hecho")
        centroide = grupo_m.geometry.unary_union.centroid
        centroide_wgs = gpd.GeoSeries([centroide], crs=gdf_m.crs).to_crs(epsg=4326).iloc[0]

        filas.append({
            "tecnica": nombre_tecnica,
            "cluster": int(cluster_id),
            "registros": int(len(grupo_m)),
            "coordenadas_unicas": int(coords_unicas),
            "modalidad_predominante": modalidad,
            "registros_modalidad_predominante": n_modalidad,
            "porcentaje_modalidad_predominante": pct_modalidad,
            "tipo_predominante": tipo,
            "subtipo_predominante": subtipo,
            "centroide_longitud": round(float(centroide_wgs.x), 6),
            "centroide_latitud": round(float(centroide_wgs.y), 6),
        })

    return pd.DataFrame(filas).sort_values("registros", ascending=False).reset_index(drop=True)

def plot_clusters_limpio(gdf_m, limite_m, columna_cluster, titulo, nombre_figura, buffer_m=120):
    clusters = gdf_m[gdf_m[columna_cluster].notna() & (gdf_m[columna_cluster] != -1)].copy()
    if clusters.empty:
        print(f"No hay clusters para graficar en {columna_cluster}.")
        return None

    poligonos = preparar_geometria_clusters(gdf_m, columna_cluster, buffer_m=buffer_m)
    fig, ax = plt.subplots(figsize=(9, 9))
    limite_m.plot(ax=ax, facecolor="#FAFAFA", edgecolor="#404040", linewidth=0.9)
    poligonos.plot(ax=ax, column=columna_cluster, cmap="tab20", alpha=0.24, edgecolor="#1F1F1F", linewidth=1.2)

    for cluster_id, grupo in clusters.groupby(columna_cluster):
        grupo.plot(ax=ax, markersize=13, alpha=0.78, label=f"Cluster {int(cluster_id)}")
        centro = grupo.geometry.unary_union.centroid
        ax.annotate(
            f"C{int(cluster_id)}\nn={len(grupo)}",
            xy=(centro.x, centro.y),
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#555555", alpha=0.85)
        )

    ax.set_title(titulo)
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal", adjustable="box")
    if clusters[columna_cluster].nunique() <= 12:
        ax.legend(fontsize=8, loc="best", frameon=True)
    return guardar_figura(nombre_figura)

def plot_clusters_zoom_urbano(gdf_m, limite_m, columna_cluster, titulo, nombre_figura, buffer_m=120, margen_m=450):
    clusters = gdf_m[gdf_m[columna_cluster].notna() & (gdf_m[columna_cluster] != -1)].copy()
    if clusters.empty:
        print(f"No hay clusters para graficar en {columna_cluster}.")
        return None

    poligonos = preparar_geometria_clusters(gdf_m, columna_cluster, buffer_m=buffer_m)
    minx, miny, maxx, maxy = clusters.total_bounds
    fig, ax = plt.subplots(figsize=(9, 9))
    limite_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#555555", linewidth=0.8)
    poligonos.plot(ax=ax, column=columna_cluster, cmap="tab20", alpha=0.28, edgecolor="#111111", linewidth=1.4)

    for cluster_id, grupo in clusters.groupby(columna_cluster):
        grupo.plot(ax=ax, markersize=16, alpha=0.82, label=f"Cluster {int(cluster_id)}")
        centro = grupo.geometry.unary_union.centroid
        ax.annotate(
            f"C{int(cluster_id)} | {len(grupo)}",
            xy=(centro.x, centro.y),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#555555", alpha=0.88)
        )

    ax.set_xlim(minx - margen_m, maxx + margen_m)
    ax.set_ylim(miny - margen_m, maxy + margen_m)
    ax.set_title(titulo)
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal", adjustable="box")
    if clusters[columna_cluster].nunique() <= 12:
        ax.legend(fontsize=8, loc="best", frameon=True)
    return guardar_figura(nombre_figura)

# %%
# Rutas configurables. 
ruta_puntos = Path("outputs_etapa3/datasets/puntos_dataset_delimitado_distrito_huaral.geojson")
ruta_distrito = Path("outputs_etapa3/datasets/distrito_huaral_gadm41.geojson")
ruta_gadm = Path("gadm41_PER.gpkg")

if ruta_puntos.exists():
    gdf_puntos = gpd.read_file(ruta_puntos)
else:
    ruta_csv = Path("outputs_etapa3/datasets/dataset_modalidades_actuales_delimitado.csv")
    if not ruta_csv.exists():
        raise FileNotFoundError("No se encontró el dataset analítico. Configurar ruta_puntos o ruta_csv.")
    df_tmp = pd.read_csv(ruta_csv)
    df_tmp["lat_hecho"] = pd.to_numeric(df_tmp["lat_hecho"], errors="coerce")
    df_tmp["long_hecho"] = pd.to_numeric(df_tmp["long_hecho"], errors="coerce")
    df_tmp = df_tmp.dropna(subset=["lat_hecho", "long_hecho"])
    gdf_puntos = gpd.GeoDataFrame(df_tmp, geometry=gpd.points_from_xy(df_tmp["long_hecho"], df_tmp["lat_hecho"]), crs="EPSG:4326")

if gdf_puntos.crs is None:
    gdf_puntos = gdf_puntos.set_crs(epsg=4326)
else:
    gdf_puntos = gdf_puntos.to_crs(epsg=4326)

if ruta_distrito.exists():
    gdf_distrito = gpd.read_file(ruta_distrito).to_crs(epsg=4326)
else:
    if not ruta_gadm.exists():
        raise FileNotFoundError("No se encontró distrito_huaral_gadm41.geojson ni gadm41_PER.gpkg.")
    gdf_peru = gpd.read_file(ruta_gadm, layer="ADM_ADM_3")
    gdf_peru["departamento_norm"] = gdf_peru["NAME_1"].apply(normalizar_texto)
    gdf_peru["provincia_norm"] = gdf_peru["NAME_2"].apply(normalizar_texto)
    gdf_peru["distrito_norm"] = gdf_peru["NAME_3"].apply(normalizar_texto)
    gdf_distrito = gdf_peru[
        (gdf_peru["departamento_norm"] == "LIMA")
        & (gdf_peru["provincia_norm"] == "HUARAL")
        & (gdf_peru["distrito_norm"] == "HUARAL")
    ].copy().to_crs(epsg=4326)

columnas_interpretacion = [
    "fecha_hora_hecho", "año_hecho", "mes_hecho", "tipo_hecho",
    "subtipo_hecho", "modalidad_hecho", "grupo_id", "grupo",
    "lat_hecho", "long_hecho", "geometry"
]
columnas_existentes = [c for c in columnas_interpretacion if c in gdf_puntos.columns]
gdf_puntos = gdf_puntos[columnas_existentes].copy()

if "grupo_id" not in gdf_puntos.columns or "grupo" not in gdf_puntos.columns:
    def asignar_grupo_analitico_tecnicas(tipo_hecho):
        tipo = normalizar_texto(tipo_hecho)
        if tipo == "LEY DE VIOLENCIA CONTRA LA MUJER Y GRUPOS VULNERABLES":
            return pd.Series({"grupo_id": "g1", "grupo": "Violencia contra la mujer y grupos vulnerables"})
        if tipo == "PATRIMONIO (DELITO)":
            return pd.Series({"grupo_id": "g2", "grupo": "Patrimonio"})
        return pd.Series({"grupo_id": "g3", "grupo": "Otros hechos de baja frecuencia"})
    grupos_tmp = gdf_puntos["tipo_hecho"].apply(asignar_grupo_analitico_tecnicas)
    gdf_puntos = pd.concat([gdf_puntos.drop(columns=["grupo_id", "grupo"], errors="ignore"), grupos_tmp], axis=1)

gdf_puntos["lat_hecho"] = pd.to_numeric(gdf_puntos["lat_hecho"], errors="coerce")
gdf_puntos["long_hecho"] = pd.to_numeric(gdf_puntos["long_hecho"], errors="coerce")
total_original = len(gdf_puntos)
gdf_puntos_validos = gdf_puntos.dropna(subset=["lat_hecho", "long_hecho"]).copy()
gdf_puntos_validos = gdf_puntos_validos[
    gdf_puntos_validos["lat_hecho"].between(-90, 90)
    & gdf_puntos_validos["long_hecho"].between(-180, 180)
].copy()

try:
    gdf_join = gpd.sjoin(gdf_puntos_validos, gdf_distrito[["geometry"]], how="left", predicate="within")
except TypeError:
    gdf_join = gpd.sjoin(gdf_puntos_validos, gdf_distrito[["geometry"]], how="left", op="within")

gdf_modelo = gdf_join[gdf_join["index_right"].notna()].drop(columns=["index_right"], errors="ignore").copy()

gdf_distrito_m = gdf_distrito.to_crs(epsg=32718)
gdf_modelo_m = gdf_modelo.to_crs(epsg=32718)
coords_m = np.column_stack([gdf_modelo_m.geometry.x, gdf_modelo_m.geometry.y])

coord_counts = (
    gdf_modelo
    .assign(long_red=lambda x: x.geometry.x.round(7), lat_red=lambda x: x.geometry.y.round(7))
    .groupby(["long_red", "lat_red"])
    .size()
)

resumen_preparacion = pd.DataFrame({
    "indicador": [
        "Total de registros cargados",
        "Registros sin coordenadas o coordenadas inválidas excluidos",
        "Puntos dentro del distrito de Huaral",
        "Coordenadas únicas",
        "Máximo de registros repetidos en una misma coordenada",
        "Grupos analíticos disponibles"
    ],
    "cantidad": [
        total_original,
        total_original - len(gdf_puntos_validos),
        len(gdf_modelo),
        int(coord_counts.shape[0]),
        int(coord_counts.max()) if not coord_counts.empty else 0,
        int(gdf_modelo["grupo_id"].nunique()) if "grupo_id" in gdf_modelo.columns else 0
    ]
})
resumen_preparacion.to_excel(salida_tablas / "resumen_541_preparacion_espacial.xlsx", index=False)
resumen_preparacion

# %%
fig, ax = plt.subplots(figsize=(8, 8))
gdf_distrito_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#333333", linewidth=1.0)
gdf_modelo_m.plot(ax=ax, markersize=5, alpha=0.25, color="#4C78A8")
ax.set_title("Puntos válidos para modelamiento geoespacial - distrito de Huaral")
ax.set_xlabel("Este (m)")
ax.set_ylabel("Norte (m)")
ax.set_aspect("equal", adjustable="box")
guardar_figura("fig_541_puntos_modelamiento_huaral.png")

# %% [markdown]
# ## 5.4.2. Preparación de grupos analíticos para el modelamiento geoespacial
# %%
orden_grupos = ["g1", "g2", "g3"]
orden_grupos_tecnicas = ["g1", "g2"]
nombres_grupos = {
    "g1": "Violencia contra la mujer y grupos vulnerables",
    "g2": "Patrimonio",
    "g3": "Otros hechos de baja frecuencia",
}
colores_grupos = {
    "g1": "#E15759",
    "g2": "#4C78A8",
    "g3": "#59A14F",
}

if "grupo_id" not in gdf_modelo_m.columns or "grupo" not in gdf_modelo_m.columns:
    raise KeyError("El dataset espacial debe contener las columnas grupo_id y grupo.")

gdf_modelo_m["grupo_id"] = gdf_modelo_m["grupo_id"].astype(str)
gdf_modelo_m["grupo"] = gdf_modelo_m["grupo"].astype(str)

grupos_modelamiento = (
    gdf_modelo_m
    .groupby(["grupo_id", "grupo"], dropna=False)
    .agg(
        registros=("geometry", "size"),
        coordenadas_unicas=("geometry", lambda s: s.apply(lambda g: (round(g.x, 2), round(g.y, 2))).nunique())
    )
    .reset_index()
    .sort_values("grupo_id")
)
grupos_modelamiento["aplicar_tecnicas_geoespaciales"] = grupos_modelamiento["grupo_id"].isin(orden_grupos_tecnicas)
grupos_modelamiento["criterio_inclusion"] = np.where(
    grupos_modelamiento["aplicar_tecnicas_geoespaciales"],
    "Incluido en KDE, DBSCAN, HDBSCAN y círculos de 100 m",
    "Excluido del modelamiento geoespacial por baja frecuencia"
)
grupos_modelamiento.to_excel(salida_tablas / "resumen_542_grupos_modelamiento_geoespacial.xlsx", index=False)
grupos_modelamiento

# %%
def nombre_archivo_grupo(grupo_id):
    return str(grupo_id).lower().replace(" ", "_")

def coords_de_gdf(gdf):
    return np.column_stack([gdf.geometry.x, gdf.geometry.y])

def modalidad_predominante(df):
    if "modalidad_hecho" not in df.columns or df.empty:
        return "No disponible", 0, 0.0
    conteo = df["modalidad_hecho"].fillna("No especificado").astype(str).value_counts()
    valor = conteo.index[0]
    n = int(conteo.iloc[0])
    pct = round(n / len(df) * 100, 2)
    return valor, n, pct

def seleccionar_parametros_dbscan(coords, objetivo_clusters=6):
    min_samples = 15 if len(coords) >= 300 else 5
    eps_values = [75, 100, 150, 200, 300, 500, 750, 1000]
    filas = []
    for eps in eps_values:
        modelo = DBSCAN(eps=eps, min_samples=min_samples)
        labels = modelo.fit_predict(coords)
        resumen = resumen_clusters(labels)
        resumen.update({"eps_m": eps, "min_samples": min_samples})
        filas.append(resumen)
    evaluacion = pd.DataFrame(filas)
    candidatos = evaluacion[evaluacion["numero_clusters"] > 0].copy()
    if candidatos.empty:
        seleccion = evaluacion.iloc[0]
    else:
        candidatos["criterio"] = (candidatos["porcentaje_ruido"] - 20).abs() + (candidatos["numero_clusters"] - objetivo_clusters).abs() * 2
        seleccion = candidatos.sort_values("criterio").iloc[0]
    return evaluacion, int(seleccion["eps_m"]), int(seleccion["min_samples"])

def seleccionar_parametros_hdbscan(coords):
    if HDBSCAN_BACKEND is None or len(coords) < 10:
        return pd.DataFrame(), None, None
    posibles_min_cluster_size = [v for v in [5, 10, 15, 30, 50, 75, 100] if v < len(coords)]
    posibles_min_samples = [v for v in [5, 10, 15, 20] if v < len(coords)]
    if not posibles_min_cluster_size or not posibles_min_samples:
        return pd.DataFrame(), None, None
    filas = []
    for min_cluster_size in posibles_min_cluster_size:
        for min_samples in posibles_min_samples:
            if min_samples > min_cluster_size:
                continue
            if HDBSCAN_BACKEND == "hdbscan":
                modelo = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            else:
                modelo = SklearnHDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = modelo.fit_predict(coords)
            resumen = resumen_clusters(labels)
            resumen.update({"min_cluster_size": min_cluster_size, "min_samples": min_samples})
            filas.append(resumen)
    evaluacion = pd.DataFrame(filas)
    if evaluacion.empty:
        return evaluacion, None, None
    candidatos = evaluacion[evaluacion["numero_clusters"] > 0].copy()
    if candidatos.empty:
        seleccion = evaluacion.iloc[0]
    else:
        candidatos["criterio"] = (candidatos["porcentaje_ruido"] - 20).abs() + (candidatos["numero_clusters"] - 6).abs() * 2
        seleccion = candidatos.sort_values("criterio").iloc[0]
    return evaluacion, int(seleccion["min_cluster_size"]), int(seleccion["min_samples"])

def zonas_calientes_radio_fijo(gdf_grupo_m, grupo_id, grupo, radio_m=100, min_separacion_m=200, max_circulos=10):
    puntos = gdf_grupo_m.copy().reset_index(drop=True)
    candidatos = (
        puntos
        .assign(x_red=lambda x: x.geometry.x.round(2), y_red=lambda x: x.geometry.y.round(2))
        .drop_duplicates(subset=["x_red", "y_red"])
        .reset_index(drop=True)
    )
    filas = []
    for idx, centro in candidatos.iterrows():
        distancias = puntos.geometry.distance(centro.geometry)
        dentro = puntos[distancias <= radio_m].copy()
        if dentro.empty:
            continue
        modalidad, n_modalidad, pct_modalidad = modalidad_predominante(dentro)
        centro_wgs = gpd.GeoSeries([centro.geometry], crs=puntos.crs).to_crs(epsg=4326).iloc[0]
        filas.append({
            "grupo_id": grupo_id,
            "grupo": grupo,
            "id_candidato": int(idx),
            "registros": int(len(dentro)),
            "coordenadas_unicas": int(dentro.geometry.apply(lambda g: (round(g.x, 2), round(g.y, 2))).nunique()),
            "modalidad_predominante": modalidad,
            "registros_modalidad_predominante": n_modalidad,
            "porcentaje_modalidad_predominante": pct_modalidad,
            "centro_longitud": round(float(centro_wgs.x), 6),
            "centro_latitud": round(float(centro_wgs.y), 6),
            "geometry": centro.geometry,
        })
    if not filas:
        return gpd.GeoDataFrame(columns=["grupo_id", "grupo", "zona_caliente", "radio_m", "geometry"], geometry="geometry", crs=gdf_grupo_m.crs)
    candidatos = gpd.GeoDataFrame(filas, geometry="geometry", crs=gdf_grupo_m.crs)
    candidatos = candidatos.sort_values(["registros", "coordenadas_unicas"], ascending=[False, False]).reset_index(drop=True)
    seleccionados = []
    for _, candidato in candidatos.iterrows():
        if len(seleccionados) >= max_circulos:
            break
        if not seleccionados or min(candidato.geometry.distance(prev.geometry) for prev in seleccionados) >= min_separacion_m:
            seleccionados.append(candidato)
    zonas = gpd.GeoDataFrame(seleccionados, geometry="geometry", crs=gdf_grupo_m.crs).reset_index(drop=True)
    zonas["zona_caliente"] = [f"{grupo_id.upper()}-ZC-{i+1}" for i in range(len(zonas))]
    zonas["radio_m"] = radio_m
    zonas["geometry"] = zonas.geometry.buffer(radio_m)
    return zonas[[
        "grupo_id", "grupo", "zona_caliente", "radio_m", "registros", "coordenadas_unicas",
        "modalidad_predominante", "registros_modalidad_predominante", "porcentaje_modalidad_predominante",
        "centro_longitud", "centro_latitud", "geometry"
    ]]

def tamano_proporcional(valor, max_valor, minimo=70, maximo=900):
    if max_valor <= 0:
        return minimo
    return minimo + (maximo - minimo) * np.sqrt(valor / max_valor)

def coordenadas_unicas_conteo(gdf_m):
    if gdf_m.empty:
        return gpd.GeoDataFrame(columns=["cantidad", "geometry"], geometry="geometry", crs=gdf_m.crs)
    tmp = gdf_m.copy()
    tmp["x_red"] = tmp.geometry.x.round(2)
    tmp["y_red"] = tmp.geometry.y.round(2)
    conteo = tmp.groupby(["x_red", "y_red"], as_index=False).size().rename(columns={"size": "cantidad"})
    return gpd.GeoDataFrame(
        conteo,
        geometry=gpd.points_from_xy(conteo["x_red"], conteo["y_red"]),
        crs=gdf_m.crs
    )

def aplicar_zoom_a_geometria(ax, gdf_m, margen_m=450):
    if gdf_m.empty:
        return
    minx, miny, maxx, maxy = gdf_m.total_bounds
    ax.set_xlim(minx - margen_m, maxx + margen_m)
    ax.set_ylim(miny - margen_m, maxy + margen_m)

def plot_puntos_proporcionales_zoom(gdf_m, limite_m, titulo, nombre_figura, color="#2F4858", margen_m=450):
    puntos = coordenadas_unicas_conteo(gdf_m)
    if puntos.empty:
        print(f"No hay puntos para graficar: {titulo}")
        return None
    max_cantidad = puntos["cantidad"].max()
    puntos["tamano"] = puntos["cantidad"].apply(lambda x: tamano_proporcional(x, max_cantidad, minimo=55, maximo=950))

    fig, ax = plt.subplots(figsize=(9, 9))
    limite_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#404040", linewidth=0.9)
    puntos.plot(ax=ax, markersize=puntos["tamano"], color=color, alpha=0.34, edgecolor="#111111", linewidth=0.35)

    principales = puntos.sort_values("cantidad", ascending=False).head(8)
    for _, fila in principales.iterrows():
        ax.annotate(
            int(fila["cantidad"]),
            xy=(fila.geometry.x, fila.geometry.y),
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#555555", alpha=0.86)
        )
    aplicar_zoom_a_geometria(ax, gdf_m, margen_m=margen_m)
    ax.set_title(titulo)
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal", adjustable="box")
    return guardar_figura(nombre_figura)

def plot_clusters_proporcionales_zoom(gdf_m, limite_m, columna_cluster, titulo, nombre_figura, buffer_m=120, margen_m=450):
    clusters = gdf_m[gdf_m[columna_cluster].notna() & (gdf_m[columna_cluster] != -1)].copy()
    if clusters.empty:
        print(f"No hay clusters para graficar en {columna_cluster}.")
        return None

    poligonos = preparar_geometria_clusters(gdf_m, columna_cluster, buffer_m=buffer_m)
    resumen = clusters.groupby(columna_cluster).size().rename("registros").reset_index()
    centroides = []
    for cluster_id, grupo_cluster in clusters.groupby(columna_cluster):
        centro = grupo_cluster.geometry.unary_union.centroid
        n = int(len(grupo_cluster))
        centroides.append({columna_cluster: cluster_id, "registros": n, "geometry": centro})
    centroides = gpd.GeoDataFrame(centroides, geometry="geometry", crs=gdf_m.crs)
    max_registros = centroides["registros"].max()
    centroides["tamano"] = centroides["registros"].apply(lambda x: tamano_proporcional(x, max_registros, minimo=90, maximo=1000))

    fig, ax = plt.subplots(figsize=(9, 9))
    limite_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#555555", linewidth=0.8)
    poligonos.plot(ax=ax, column=columna_cluster, cmap="tab20", alpha=0.28, edgecolor="#111111", linewidth=1.4)
    clusters.plot(ax=ax, markersize=5, alpha=0.22, color="#2F4858")
    centroides.plot(ax=ax, markersize=centroides["tamano"], alpha=0.42, color="#F28E2B", edgecolor="#7F0000", linewidth=0.7)

    for _, fila in centroides.iterrows():
        ax.annotate(
            f"C{int(fila[columna_cluster])}\nn={int(fila['registros'])}",
            xy=(fila.geometry.x, fila.geometry.y), ha="center", va="center",
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#7F0000", alpha=0.88)
        )
    aplicar_zoom_a_geometria(ax, clusters, margen_m=margen_m)
    ax.set_title(titulo)
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal", adjustable="box")
    return guardar_figura(nombre_figura)

def plot_zonas_zoom_proporcional(zonas_m, puntos_m, limite_m, titulo, nombre_figura, margen_m=450):
    if zonas_m.empty:
        print(f"No hay zonas calientes para graficar: {titulo}")
        return None
    zonas = zonas_m.copy()
    max_registros = zonas["registros"].max()
    centroides = zonas.copy()
    centroides["geometry"] = centroides.geometry.centroid
    centroides["tamano"] = centroides["registros"].apply(lambda x: tamano_proporcional(x, max_registros, minimo=90, maximo=1050))

    fig, ax = plt.subplots(figsize=(9, 9))
    limite_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#555555", linewidth=0.8)
    zonas.plot(ax=ax, column="registros", cmap="YlOrRd", alpha=0.36, edgecolor="#7F0000", linewidth=1.5)
    puntos_en_zonas = puntos_m[puntos_m.geometry.apply(lambda g: zonas.contains(g).any())]
    if not puntos_en_zonas.empty:
        puntos_en_zonas.plot(ax=ax, markersize=5, alpha=0.20, color="#2F4858")
    centroides.plot(ax=ax, markersize=centroides["tamano"], color="#F28E2B", alpha=0.38, edgecolor="#7F0000", linewidth=0.8)

    for _, fila in centroides.iterrows():
        ax.annotate(
            f"{fila['zona_caliente']}\nn={int(fila['registros'])}",
            xy=(fila.geometry.x, fila.geometry.y), ha="center", va="center",
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#7F0000", alpha=0.9)
        )
    aplicar_zoom_a_geometria(ax, zonas, margen_m=margen_m)
    ax.set_title(titulo)
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal", adjustable="box")
    return guardar_figura(nombre_figura)

# %% [markdown]
# ## 5.4.3. KDE por grupo analítico
# 

# %%
resultados_kde_grupos = []
for grupo_id in orden_grupos_tecnicas:
    datos_grupo = gdf_modelo_m[gdf_modelo_m["grupo_id"] == grupo_id].copy()
    if datos_grupo.empty:
        continue
    grupo = datos_grupo["grupo"].iloc[0]
    coords = coords_de_gdf(datos_grupo)
    coords_eval = coords
    if len(coords) > 3000:
        rng = np.random.default_rng(42)
        coords_eval = coords[rng.choice(len(coords), size=3000, replace=False)]
    bandwidths = [100, 150, 200, 300, 500, 750, 1000]
    if len(coords_eval) >= 20:
        grid = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=min(5, len(coords_eval)))
        grid.fit(coords_eval)
        bandwidth = int(grid.best_params_["bandwidth"])
        score_cv = float(grid.best_score_)
    else:
        bandwidth = 200
        score_cv = np.nan
    resultados_kde_grupos.append({"grupo_id": grupo_id, "grupo": grupo, "registros": len(datos_grupo), "bandwidth_m": bandwidth, "score_cv": score_cv})

    minx, miny, maxx, maxy = gdf_distrito_m.total_bounds
    x_grid = np.linspace(minx, maxx, 220)
    y_grid = np.linspace(miny, maxy, 220)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(coords)
    zz = np.exp(kde.score_samples(grid_coords)).reshape(xx.shape)
    mascara = contains(gdf_distrito_m.geometry.union_all() if hasattr(gdf_distrito_m.geometry, "union_all") else gdf_distrito_m.geometry.unary_union, grid_coords[:, 0], grid_coords[:, 1]).reshape(xx.shape)
    zz = np.where(mascara, zz, np.nan)

    fig, ax = plt.subplots(figsize=(9, 9))
    gdf_distrito_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#333333", linewidth=1.0)
    cont = ax.contourf(xx, yy, zz, levels=15, cmap="YlOrRd", alpha=0.75)
    datos_grupo.plot(ax=ax, markersize=4, alpha=0.20, color="#2F4858")
    fig.colorbar(cont, ax=ax, shrink=0.72, label="Densidad estimada KDE")
    ax.set_title(f"KDE - {grupo_id}: {grupo}")
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal", adjustable="box")
    guardar_figura(f"fig_542_kde_densidad_{nombre_archivo_grupo(grupo_id)}.png")

    plot_puntos_proporcionales_zoom(
        datos_grupo,
        gdf_distrito_m,
        f"KDE y concentración proporcional - zoom urbano - {grupo_id}: {grupo}",
        f"fig_542b_kde_zoom_urbano_proporcional_{nombre_archivo_grupo(grupo_id)}.png",
        color=colores_grupos.get(grupo_id, "#2F4858"),
        margen_m=450
    )

resultados_kde_grupos = pd.DataFrame(resultados_kde_grupos)
resultados_kde_grupos.to_excel(salida_tablas / "kde_resultados_bandwidth_por_grupo.xlsx", index=False)
resultados_kde_grupos

# %% [markdown]
# ## 5.4.4. DBSCAN por grupo analítico
# 

# %%
resultados_dbscan_grupos = []
evaluaciones_dbscan_grupos = []
resumen_detallado_dbscan_grupos = []
gdf_resultados_grupos_m = gdf_modelo_m.copy()
gdf_resultados_grupos_m["cluster_dbscan"] = np.nan

for grupo_id in orden_grupos_tecnicas:
    datos_grupo = gdf_modelo_m[gdf_modelo_m["grupo_id"] == grupo_id].copy()
    if datos_grupo.empty:
        continue
    grupo = datos_grupo["grupo"].iloc[0]
    coords = coords_de_gdf(datos_grupo)
    evaluacion, eps_dbscan, min_samples_dbscan = seleccionar_parametros_dbscan(coords)
    evaluacion.insert(0, "grupo", grupo)
    evaluacion.insert(0, "grupo_id", grupo_id)
    evaluaciones_dbscan_grupos.append(evaluacion)

    modelo = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
    labels = modelo.fit_predict(coords)
    datos_grupo["cluster_dbscan"] = labels
    gdf_resultados_grupos_m.loc[datos_grupo.index, "cluster_dbscan"] = labels
    resumen = resumen_clusters(labels)
    resumen.update({"grupo_id": grupo_id, "grupo": grupo, "eps_usado_m": eps_dbscan, "min_samples_usado": min_samples_dbscan})
    resultados_dbscan_grupos.append(resumen)

    detalle = resumen_detallado_clusters(datos_grupo, "cluster_dbscan", "DBSCAN")
    if not detalle.empty:
        detalle.insert(0, "grupo", grupo)
        detalle.insert(0, "grupo_id", grupo_id)
        resumen_detallado_dbscan_grupos.append(detalle)
        poligonos = preparar_geometria_clusters(datos_grupo, "cluster_dbscan", buffer_m=120)
        poligonos.insert(0, "grupo", grupo)
        poligonos.insert(0, "grupo_id", grupo_id)
        poligonos.to_crs(epsg=4326).to_file(salida_datasets / f"dbscan_poligonos_buffers_clusters_{nombre_archivo_grupo(grupo_id)}.geojson", driver="GeoJSON")

    plot_clusters_limpio(
        datos_grupo,
        gdf_distrito_m,
        "cluster_dbscan",
        f"DBSCAN - {grupo_id}: {grupo}",
        f"fig_543_dbscan_clusters_{nombre_archivo_grupo(grupo_id)}.png",
        buffer_m=120
    )

    plot_clusters_proporcionales_zoom(
        datos_grupo,
        gdf_distrito_m,
        "cluster_dbscan",
        f"DBSCAN - zoom urbano con buffers y círculos proporcionales - {grupo_id}: {grupo}",
        f"fig_543b_dbscan_zoom_urbano_proporcional_{nombre_archivo_grupo(grupo_id)}.png",
        buffer_m=120,
        margen_m=450
    )

resultados_dbscan_grupos = pd.DataFrame(resultados_dbscan_grupos)
evaluaciones_dbscan_grupos = pd.concat(evaluaciones_dbscan_grupos, ignore_index=True) if evaluaciones_dbscan_grupos else pd.DataFrame()
resumen_detallado_dbscan_grupos = pd.concat(resumen_detallado_dbscan_grupos, ignore_index=True) if resumen_detallado_dbscan_grupos else pd.DataFrame()

resultados_dbscan_grupos.to_excel(salida_tablas / "dbscan_resumen_modelos_por_grupo.xlsx", index=False)
evaluaciones_dbscan_grupos.to_excel(salida_tablas / "dbscan_evaluacion_parametros_por_grupo.xlsx", index=False)
resumen_detallado_dbscan_grupos.to_excel(salida_tablas / "dbscan_resumen_detallado_clusters_por_grupo.xlsx", index=False)
resultados_dbscan_grupos

# %% [markdown]
# ## 5.4.5. HDBSCAN por grupo analítico
# 

# %%
print("Backend HDBSCAN disponible:", HDBSCAN_BACKEND if HDBSCAN_BACKEND is not None else "No disponible")
resultados_hdbscan_grupos = []
evaluaciones_hdbscan_grupos = []
resumen_detallado_hdbscan_grupos = []
gdf_resultados_grupos_m["cluster_hdbscan"] = np.nan

for grupo_id in orden_grupos_tecnicas:
    datos_grupo = gdf_modelo_m[gdf_modelo_m["grupo_id"] == grupo_id].copy()
    if datos_grupo.empty:
        continue
    grupo = datos_grupo["grupo"].iloc[0]
    coords = coords_de_gdf(datos_grupo)
    evaluacion, min_cluster_size, min_samples = seleccionar_parametros_hdbscan(coords)
    if not evaluacion.empty:
        evaluacion.insert(0, "grupo", grupo)
        evaluacion.insert(0, "grupo_id", grupo_id)
        evaluaciones_hdbscan_grupos.append(evaluacion)
    if min_cluster_size is None:
        resultados_hdbscan_grupos.append({
            "grupo_id": grupo_id, "grupo": grupo, "numero_clusters": np.nan,
            "puntos_ruido": np.nan, "porcentaje_ruido": np.nan,
            "tamano_cluster_mas_grande": np.nan, "tamano_promedio_clusters": np.nan,
            "min_cluster_size_usado": np.nan, "min_samples_usado": np.nan,
            "nota": "HDBSCAN no disponible o grupo con pocos registros"
        })
        continue
    if HDBSCAN_BACKEND == "hdbscan":
        modelo = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    else:
        modelo = SklearnHDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = modelo.fit_predict(coords)
    datos_grupo["cluster_hdbscan"] = labels
    gdf_resultados_grupos_m.loc[datos_grupo.index, "cluster_hdbscan"] = labels
    resumen = resumen_clusters(labels)
    resumen.update({"grupo_id": grupo_id, "grupo": grupo, "min_cluster_size_usado": min_cluster_size, "min_samples_usado": min_samples})
    resultados_hdbscan_grupos.append(resumen)

    detalle = resumen_detallado_clusters(datos_grupo, "cluster_hdbscan", "HDBSCAN")
    if not detalle.empty:
        detalle.insert(0, "grupo", grupo)
        detalle.insert(0, "grupo_id", grupo_id)
        resumen_detallado_hdbscan_grupos.append(detalle)
        poligonos = preparar_geometria_clusters(datos_grupo, "cluster_hdbscan", buffer_m=120)
        poligonos.insert(0, "grupo", grupo)
        poligonos.insert(0, "grupo_id", grupo_id)
        poligonos.to_crs(epsg=4326).to_file(salida_datasets / f"hdbscan_poligonos_buffers_clusters_{nombre_archivo_grupo(grupo_id)}.geojson", driver="GeoJSON")

    plot_clusters_limpio(
        datos_grupo,
        gdf_distrito_m,
        "cluster_hdbscan",
        f"HDBSCAN - {grupo_id}: {grupo}",
        f"fig_544_hdbscan_clusters_{nombre_archivo_grupo(grupo_id)}.png",
        buffer_m=120
    )

    plot_clusters_proporcionales_zoom(
        datos_grupo,
        gdf_distrito_m,
        "cluster_hdbscan",
        f"HDBSCAN - zoom urbano con buffers y círculos proporcionales - {grupo_id}: {grupo}",
        f"fig_544b_hdbscan_zoom_urbano_proporcional_{nombre_archivo_grupo(grupo_id)}.png",
        buffer_m=120,
        margen_m=450
    )

resultados_hdbscan_grupos = pd.DataFrame(resultados_hdbscan_grupos)
evaluaciones_hdbscan_grupos = pd.concat(evaluaciones_hdbscan_grupos, ignore_index=True) if evaluaciones_hdbscan_grupos else pd.DataFrame()
resumen_detallado_hdbscan_grupos = pd.concat(resumen_detallado_hdbscan_grupos, ignore_index=True) if resumen_detallado_hdbscan_grupos else pd.DataFrame()

resultados_hdbscan_grupos.to_excel(salida_tablas / "hdbscan_resumen_modelos_por_grupo.xlsx", index=False)
evaluaciones_hdbscan_grupos.to_excel(salida_tablas / "hdbscan_evaluacion_parametros_por_grupo.xlsx", index=False)
resumen_detallado_hdbscan_grupos.to_excel(salida_tablas / "hdbscan_resumen_detallado_clusters_por_grupo.xlsx", index=False)
resultados_hdbscan_grupos

# %% [markdown]
# ## 5.4.6. Zonas calientes con círculos de 100 m por grupo analítico
# 

# %%
zonas_calientes_grupos = []
for grupo_id in orden_grupos_tecnicas:
    datos_grupo = gdf_modelo_m[gdf_modelo_m["grupo_id"] == grupo_id].copy()
    if datos_grupo.empty:
        continue
    grupo = datos_grupo["grupo"].iloc[0]
    zonas = zonas_calientes_radio_fijo(datos_grupo, grupo_id, grupo, radio_m=100, min_separacion_m=200, max_circulos=10)
    if zonas.empty:
        continue
    zonas_calientes_grupos.append(zonas)
    zonas.to_crs(epsg=4326).to_file(salida_datasets / f"zonas_calientes_circulos_100m_{nombre_archivo_grupo(grupo_id)}.geojson", driver="GeoJSON")

    fig, ax = plt.subplots(figsize=(9, 9))
    gdf_distrito_m.plot(ax=ax, facecolor="#FAFAFA", edgecolor="#404040", linewidth=0.9)
    puntos_en_zonas = datos_grupo[datos_grupo.geometry.apply(lambda g: zonas.contains(g).any())]
    if not puntos_en_zonas.empty:
        puntos_en_zonas.plot(ax=ax, color=colores_grupos.get(grupo_id, "#2F4858"), markersize=9, alpha=0.45, label="Registros dentro de círculos")
    zonas.plot(
        ax=ax, column="registros", cmap="YlOrRd", alpha=0.38,
        edgecolor="#7F0000", linewidth=1.5, legend=True,
        legend_kwds={"label": "Registros dentro del radio de 100 m", "shrink": 0.7}
    )
    for _, fila in zonas.iterrows():
        centro = fila.geometry.centroid
        ax.annotate(
            f"{fila['zona_caliente']}\nn={int(fila['registros'])}",
            xy=(centro.x, centro.y), ha="center", va="center",
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7F0000", alpha=0.88)
        )
    ax.set_title(f"Círculos críticos 100 m - {grupo_id}: {grupo}")
    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    guardar_figura(f"fig_545_circulos_100m_{nombre_archivo_grupo(grupo_id)}.png")

    plot_zonas_zoom_proporcional(
        zonas,
        datos_grupo,
        gdf_distrito_m,
        f"Círculos críticos 100 m - zoom urbano proporcional - {grupo_id}: {grupo}",
        f"fig_545b_circulos_100m_zoom_urbano_proporcional_{nombre_archivo_grupo(grupo_id)}.png",
        margen_m=450
    )

zonas_calientes_grupos = pd.concat(zonas_calientes_grupos, ignore_index=True) if zonas_calientes_grupos else gpd.GeoDataFrame()
if not zonas_calientes_grupos.empty:
    pd.DataFrame(zonas_calientes_grupos.drop(columns="geometry")).to_excel(salida_tablas / "zonas_calientes_circulos_100m_por_grupo.xlsx", index=False)
    zonas_calientes_grupos.to_crs(epsg=4326).to_file(salida_datasets / "zonas_calientes_circulos_100m_por_grupo.geojson", driver="GeoJSON")
zonas_calientes_grupos.drop(columns="geometry", errors="ignore") if not zonas_calientes_grupos.empty else zonas_calientes_grupos

# %% [markdown]
# ## 5.4.7. Comparación de técnicas por grupo analítico
# 

# %%
comparacion_tecnicas_grupos = []
for _, fila in grupos_modelamiento[grupos_modelamiento["aplicar_tecnicas_geoespaciales"]].iterrows():
    grupo_id = fila["grupo_id"]
    grupo = fila["grupo"]
    kde_fila = resultados_kde_grupos[resultados_kde_grupos["grupo_id"] == grupo_id]
    db_fila = resultados_dbscan_grupos[resultados_dbscan_grupos["grupo_id"] == grupo_id]
    hdb_fila = resultados_hdbscan_grupos[resultados_hdbscan_grupos["grupo_id"] == grupo_id]
    zonas_fila = zonas_calientes_grupos[zonas_calientes_grupos["grupo_id"] == grupo_id] if not zonas_calientes_grupos.empty else pd.DataFrame()
    comparacion_tecnicas_grupos.extend([
        {
            "grupo_id": grupo_id,
            "grupo": grupo,
            "tecnica": "KDE",
            "parametros_usados": "No disponible" if kde_fila.empty else f"bandwidth={int(kde_fila.iloc[0]['bandwidth_m'])} m",
            "numero_clusters_o_zonas": "Superficie continua",
            "registros_ruido_o_no_asignados_pct": np.nan,
            "utilidad_principal": "Visualizar concentración espacial relativa dentro del grupo"
        },
        {
            "grupo_id": grupo_id,
            "grupo": grupo,
            "tecnica": "DBSCAN",
            "parametros_usados": "No disponible" if db_fila.empty else f"eps={int(db_fila.iloc[0]['eps_usado_m'])} m; min_samples={int(db_fila.iloc[0]['min_samples_usado'])}",
            "numero_clusters_o_zonas": np.nan if db_fila.empty else db_fila.iloc[0]["numero_clusters"],
            "registros_ruido_o_no_asignados_pct": np.nan if db_fila.empty else db_fila.iloc[0]["porcentaje_ruido"],
            "utilidad_principal": "Identificar agrupamientos espaciales del grupo con radio fijo"
        },
        {
            "grupo_id": grupo_id,
            "grupo": grupo,
            "tecnica": "HDBSCAN",
            "parametros_usados": "No disponible" if hdb_fila.empty or pd.isna(hdb_fila.iloc[0].get("numero_clusters")) else f"min_cluster_size={int(hdb_fila.iloc[0]['min_cluster_size_usado'])}; min_samples={int(hdb_fila.iloc[0]['min_samples_usado'])}",
            "numero_clusters_o_zonas": np.nan if hdb_fila.empty else hdb_fila.iloc[0].get("numero_clusters", np.nan),
            "registros_ruido_o_no_asignados_pct": np.nan if hdb_fila.empty else hdb_fila.iloc[0].get("porcentaje_ruido", np.nan),
            "utilidad_principal": "Comparar agrupamientos con densidades variables dentro del grupo"
        },
        {
            "grupo_id": grupo_id,
            "grupo": grupo,
            "tecnica": "Círculos 100 m",
            "parametros_usados": "radio=100 m; separación mínima entre centros=200 m",
            "numero_clusters_o_zonas": int(len(zonas_fila)) if not zonas_fila.empty else 0,
            "registros_ruido_o_no_asignados_pct": np.nan,
            "utilidad_principal": "Delimitar zonas calientes interpretables con radio máximo fijo"
        }
    ])
comparacion_tecnicas_grupos = pd.DataFrame(comparacion_tecnicas_grupos)
comparacion_tecnicas_grupos.to_excel(salida_tablas / "comparacion_547_tecnicas_geoespaciales_por_grupo.xlsx", index=False)

gdf_resultados_grupos_m.to_crs(epsg=4326).to_file(salida_datasets / "puntos_resultados_tecnicas_por_grupo.geojson", driver="GeoJSON")
comparacion_tecnicas_grupos
