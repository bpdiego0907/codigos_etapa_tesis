from pathlib import Path
import os
import warnings

base_dir = Path.cwd()
matplotlib_cache = base_dir / ".matplotlib_cache"
matplotlib_cache.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.ops import unary_union
from libpysal.weights import Queen
from sklearn.cluster import DBSCAN

try:
    import contextily as cx
    CONTEXTILY_DISPONIBLE = True
except Exception:
    CONTEXTILY_DISPONIBLE = False

try:
    import folium
    FOLIUM_DISPONIBLE = True
except Exception:
    FOLIUM_DISPONIBLE = False

plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

CRS_METRICO = "EPSG:32718"
PERCENTIL_HOTSPOT = 0.90
TAMANO_GRILLA_M = 250
GRUPOS_EVALUACION = ["g1", "g2"]
NOMBRES_GRUPOS = {
    "g1": "Violencia contra la mujer y grupos vulnerables",
    "g2": "Patrimonio",
}
COLORES_GRUPOS = {"g1": "#C43C4E", "g2": "#2878A5"}
MESES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

salida_base = Path("outputs_etapa56")
salida_figuras = salida_base / "figuras"
salida_tablas = salida_base / "tablas"
salida_datasets = salida_base / "datasets"
for carpeta in [salida_figuras, salida_tablas, salida_datasets]:
    carpeta.mkdir(parents=True, exist_ok=True)

def guardar_figura(nombre, dpi=300):
    ruta = salida_figuras / nombre
    plt.tight_layout()
    plt.savefig(ruta, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.show()
    return ruta

def moda_segura(serie):
    valores = serie.dropna().astype(str)
    valores = valores[valores.str.strip() != ""]
    if valores.empty:
        return "No disponible"
    return valores.value_counts().index[0]

def expandir_limites(bounds, proporcion=0.12):
    minx, miny, maxx, maxy = bounds
    dx = max(maxx - minx, 1)
    dy = max(maxy - miny, 1)
    return (
        minx - dx * proporcion,
        miny - dy * proporcion,
        maxx + dx * proporcion,
        maxy + dy * proporcion,
    )

print("Contextily disponible:", CONTEXTILY_DISPONIBLE)
print("Folium disponible:", FOLIUM_DISPONIBLE)
print("Percentil hotspot:", PERCENTIL_HOTSPOT)

# %% [markdown]
# ## 5.6.1. Carga y reutilización de insumos finales
# 
# %%
ruta_puntos_54 = Path("outputs_etapa54/datasets/puntos_resultados_tecnicas_por_grupo.geojson")
ruta_dataset_analitico = Path("outputs_etapa3/datasets/dataset_modalidades_actuales_delimitado.csv")
ruta_grilla_resultados = Path("outputs_etapa55/datasets/grilla_hexagonal_250m_resultados_tecnicas.geojson")
ruta_grilla_seleccionada = Path("outputs_etapa55/datasets/grilla_hexagonal_250m_seleccionada.geojson")
ruta_distrito = Path("outputs_etapa3/datasets/distrito_huaral_gadm41.geojson")
ruta_conclusion_55 = Path("outputs_etapa55/tablas/conclusion_tecnica_recomendada_por_grupo.xlsx")

for ruta in [
    ruta_puntos_54,
    ruta_dataset_analitico,
    ruta_grilla_resultados,
    ruta_grilla_seleccionada,
    ruta_distrito,
]:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el insumo requerido: {ruta}")

gdf_puntos_54 = gpd.read_file(ruta_puntos_54)
df_analitico = pd.read_csv(ruta_dataset_analitico, low_memory=False)
gdf_grilla_resultados = gpd.read_file(ruta_grilla_resultados)
gdf_grilla_base = gpd.read_file(ruta_grilla_seleccionada)
gdf_distrito = gpd.read_file(ruta_distrito)
conclusion_55 = pd.read_excel(ruta_conclusion_55) if ruta_conclusion_55.exists() else pd.DataFrame()

df_analitico["lat_hecho"] = pd.to_numeric(df_analitico["lat_hecho"], errors="coerce")
df_analitico["long_hecho"] = pd.to_numeric(df_analitico["long_hecho"], errors="coerce")
df_analitico["año_hecho"] = pd.to_numeric(df_analitico["año_hecho"], errors="coerce")
df_analitico["mes_hecho"] = pd.to_numeric(df_analitico["mes_hecho"], errors="coerce")
df_analitico = df_analitico[
    df_analitico["grupo_id"].astype(str).isin(GRUPOS_EVALUACION)
    & df_analitico["lat_hecho"].notna()
    & df_analitico["long_hecho"].notna()
].copy()

gdf_delitos = gpd.GeoDataFrame(
    df_analitico,
    geometry=gpd.points_from_xy(df_analitico["long_hecho"], df_analitico["lat_hecho"]),
    crs="EPSG:4326",
)

gdf_distrito_m = gdf_distrito.to_crs(CRS_METRICO)
gdf_grilla_base_m = gdf_grilla_base.to_crs(CRS_METRICO)
gdf_grilla_resultados_m = gdf_grilla_resultados.to_crs(CRS_METRICO)
gdf_delitos_m = gdf_delitos.to_crs(CRS_METRICO)
limite_distrital = unary_union(gdf_distrito_m.geometry)
gdf_delitos_m = gdf_delitos_m[gdf_delitos_m.geometry.within(limite_distrital)].copy()

grilla_kde_m = gdf_grilla_resultados_m[
    (gdf_grilla_resultados_m["tecnica"] == "KDE")
    & (gdf_grilla_resultados_m["grupo_id"].astype(str).isin(GRUPOS_EVALUACION))
].copy()

resumen_insumos = pd.DataFrame({
    "Insumo": [
        "Delitos analíticos dentro de Huaral",
        "Hexágonos de la grilla seleccionada",
        "Registros de grilla KDE por grupo",
        "Tamaño de grilla (m)",
        "Técnica final",
    ],
    "Valor": [
        len(gdf_delitos_m),
        len(gdf_grilla_base_m),
        len(grilla_kde_m),
        TAMANO_GRILLA_M,
        "KDE",
    ],
})
resumen_insumos.to_excel(salida_tablas / "resumen_insumos_etapa56.xlsx", index=False)
resumen_insumos

# %% [markdown]
# ## 5.6.2. Identificación de celdas candidatas a zona caliente final
# 
# %%
filas_umbral = []
capas_candidatas = []

for grupo_id in GRUPOS_EVALUACION:
    capa = grilla_kde_m[grilla_kde_m["grupo_id"].astype(str) == grupo_id].copy()
    valores_relevantes = capa.loc[capa["conteo_delitos"] > 0, "valor_tecnica"]
    if valores_relevantes.empty:
        raise ValueError(f"No existen celdas ocupadas para {grupo_id}.")
    umbral = float(valores_relevantes.quantile(PERCENTIL_HOTSPOT))
    capa["umbral_kde_final"] = umbral
    capa["es_hotspot_final"] = (
        (capa["conteo_delitos"] > 0)
        & (capa["valor_tecnica"] >= umbral)
    )
    candidatas = capa[capa["es_hotspot_final"]].copy()
    candidatas["percentil_hotspot"] = PERCENTIL_HOTSPOT
    capas_candidatas.append(candidatas)
    filas_umbral.append({
        "grupo_id": grupo_id,
        "grupo": NOMBRES_GRUPOS[grupo_id],
        "percentil_hotspot": PERCENTIL_HOTSPOT,
        "umbral_kde": umbral,
        "celdas_ocupadas_evaluadas": int((capa["conteo_delitos"] > 0).sum()),
        "celdas_hotspot_final": len(candidatas),
        "delitos_en_celdas_hotspot": int(candidatas["conteo_delitos"].sum()),
        "pct_delitos_grupo_en_celdas_hotspot": round(
            candidatas["conteo_delitos"].sum() / capa["conteo_delitos"].sum() * 100,
            2,
        ),
    })

celdas_hotspot_finales_m = gpd.GeoDataFrame(
    pd.concat(capas_candidatas, ignore_index=True),
    geometry="geometry",
    crs=CRS_METRICO,
)
tabla_umbrales_hotspot = pd.DataFrame(filas_umbral)
tabla_umbrales_hotspot.to_excel(
    salida_tablas / "umbrales_y_celdas_hotspot_finales.xlsx",
    index=False,
)
celdas_hotspot_finales_m.to_crs(epsg=4326).to_file(
    salida_datasets / "celdas_hotspot_finales_kde.geojson",
    driver="GeoJSON",
)
tabla_umbrales_hotspot

# %% [markdown]
# ## 5.6.3. Agrupación de celdas contiguas en zonas calientes finales
# 
# %%
def componentes_conectados(capa):
    capa = capa.reset_index(drop=True).copy()
    pesos = Queen.from_dataframe(capa, ids=capa.index.tolist(), silence_warnings=True)
    visitados = set()
    componentes = []
    for nodo in capa.index:
        if nodo in visitados:
            continue
        pila = [nodo]
        visitados.add(nodo)
        componente = []
        while pila:
            actual = pila.pop()
            componente.append(actual)
            for vecino in pesos.neighbors.get(actual, []):
                if vecino not in visitados:
                    visitados.add(vecino)
                    pila.append(vecino)
        componentes.append(componente)
    return capa, componentes

zonas_por_grupo = {}
filas_zonas = []
for grupo_id in GRUPOS_EVALUACION:
    candidatas = celdas_hotspot_finales_m[
        celdas_hotspot_finales_m["grupo_id"].astype(str) == grupo_id
    ].copy()
    candidatas, componentes = componentes_conectados(candidatas)
    candidatos_zona = []
    for numero, indices in enumerate(componentes, start=1):
        sub = candidatas.loc[indices]
        candidatos_zona.append({
            "codigo_zona": f"ZC-{grupo_id.upper()}-{numero:02d}",
            "grupo_id": grupo_id,
            "grupo": NOMBRES_GRUPOS[grupo_id],
            "numero_celdas": len(sub),
            "densidad_kde_max": float(sub["valor_tecnica"].max()),
            "densidad_kde_media": float(sub["valor_tecnica"].mean()),
            "conteo_grilla": int(sub["conteo_delitos"].sum()),
            "geometry": unary_union(sub.geometry),
        })
    zona_gdf = gpd.GeoDataFrame(candidatos_zona, crs=CRS_METRICO)
    zona_gdf = zona_gdf.sort_values(
        ["conteo_grilla", "densidad_kde_max"],
        ascending=[False, False],
    ).reset_index(drop=True)
    zona_gdf["codigo_zona"] = [
        f"ZC-{grupo_id.upper()}-{i:02d}" for i in range(1, len(zona_gdf) + 1)
    ]
    zonas_por_grupo[grupo_id] = zona_gdf
    filas_zonas.append(zona_gdf)

zonas_g1 = zonas_por_grupo["g1"]
zonas_g2 = zonas_por_grupo["g2"]
zonas_finales_m = gpd.GeoDataFrame(
    pd.concat(filas_zonas, ignore_index=True),
    geometry="geometry",
    crs=CRS_METRICO,
)

# Se vincula cada hexágono hotspot con su zona disuelta y se calcula su centro.
celdas_hotspot_finales_m = celdas_hotspot_finales_m.copy()
codigos_zona = []
for _, celda in celdas_hotspot_finales_m.iterrows():
    zonas_grupo = zonas_finales_m[
        zonas_finales_m["grupo_id"].astype(str) == str(celda["grupo_id"])
    ]
    coincidencias = zonas_grupo[zonas_grupo.geometry.intersects(celda.geometry)]
    codigos_zona.append(
        coincidencias.iloc[0]["codigo_zona"] if not coincidencias.empty else pd.NA
    )
celdas_hotspot_finales_m["codigo_zona"] = codigos_zona

centros_hexagonos_m = celdas_hotspot_finales_m[
    [
        "hex_id",
        "codigo_zona",
        "grupo_id",
        "grupo",
        "conteo_delitos",
        "valor_tecnica",
        "umbral_kde_final",
        "percentil_hotspot",
        "geometry",
    ]
].copy()
centros_hexagonos_m["geometry"] = centros_hexagonos_m.geometry.centroid
centros_hexagonos_m = gpd.GeoDataFrame(
    centros_hexagonos_m,
    geometry="geometry",
    crs=CRS_METRICO,
)
centros_hexagonos_wgs84 = centros_hexagonos_m.to_crs(epsg=4326)

tabla_centros_hexagonos_hotspot = pd.DataFrame({
    "hex_id": centros_hexagonos_wgs84["hex_id"],
    "codigo_zona": centros_hexagonos_wgs84["codigo_zona"],
    "grupo_id": centros_hexagonos_wgs84["grupo_id"],
    "grupo": centros_hexagonos_wgs84["grupo"],
    "conteo_delitos": centros_hexagonos_wgs84["conteo_delitos"],
    "densidad_kde": centros_hexagonos_wgs84["valor_tecnica"],
    "centroide_longitud": centros_hexagonos_wgs84.geometry.x.round(6),
    "centroide_latitud": centros_hexagonos_wgs84.geometry.y.round(6),
})
tabla_centros_hexagonos_hotspot = tabla_centros_hexagonos_hotspot.sort_values(
    ["grupo_id", "codigo_zona", "hex_id"]
).reset_index(drop=True)
tabla_centros_hexagonos_hotspot.to_excel(
    salida_tablas / "coordenadas_centros_hexagonos_hotspot.xlsx",
    index=False,
)
tabla_centros_hexagonos_hotspot.to_csv(
    salida_tablas / "coordenadas_centros_hexagonos_hotspot.csv",
    index=False,
    encoding="utf-8-sig",
)
centros_hexagonos_wgs84.to_file(
    salida_datasets / "centros_hexagonos_hotspot.geojson",
    driver="GeoJSON",
)

resumen_componentes = zonas_finales_m[
    ["codigo_zona", "grupo_id", "grupo", "numero_celdas", "conteo_grilla"]
].copy()
resumen_componentes.to_excel(
    salida_tablas / "resumen_componentes_zonas_calientes.xlsx",
    index=False,
)
tabla_centros_hexagonos_hotspot

# %% [markdown]
# ## 5.6.4. Caracterización de zonas calientes finales
# 
# %%
filas_caracterizacion = []
asignaciones_delitos = []

for _, zona in zonas_finales_m.iterrows():
    delitos_grupo = gdf_delitos_m[
        gdf_delitos_m["grupo_id"].astype(str) == zona["grupo_id"]
    ].copy()
    dentro = delitos_grupo[delitos_grupo.geometry.within(zona.geometry)].copy()
    if dentro.empty:
        dentro = delitos_grupo[delitos_grupo.geometry.intersects(zona.geometry)].copy()
    dentro["codigo_zona"] = zona["codigo_zona"]
    asignaciones_delitos.append(dentro)

    area_km2 = zona.geometry.area / 1_000_000
    total_grupo = len(delitos_grupo)
    centroide_wgs84 = gpd.GeoSeries(
        [zona.geometry.centroid],
        crs=CRS_METRICO,
    ).to_crs(epsg=4326).iloc[0]
    mes_modal = moda_segura(dentro["mes_nombre"]) if "mes_nombre" in dentro.columns else (
        MESES.get(int(pd.to_numeric(dentro["mes_hecho"], errors="coerce").mode().iloc[0]), "No disponible")
        if not dentro.empty and pd.to_numeric(dentro["mes_hecho"], errors="coerce").notna().any()
        else "No disponible"
    )
    filas_caracterizacion.append({
        "codigo_zona": zona["codigo_zona"],
        "grupo_id": zona["grupo_id"],
        "grupo": zona["grupo"],
        "numero_celdas": int(zona["numero_celdas"]),
        "cantidad_registros": len(dentro),
        "area_km2": area_km2,
        "densidad_delitos_km2": len(dentro) / area_km2 if area_km2 else np.nan,
        "modalidad_predominante": moda_segura(dentro["modalidad_hecho"]),
        "tipo_hecho_predominante": moda_segura(dentro["tipo_hecho"]),
        "subtipo_predominante": moda_segura(dentro["subtipo_hecho"]),
        "turno_predominante": moda_segura(dentro["turno_hecho"]),
        "anio_mayor_registros": moda_segura(dentro["año_hecho"]),
        "mes_mayor_registros": mes_modal,
        "porcentaje_total_grupo": len(dentro) / total_grupo * 100 if total_grupo else np.nan,
        "densidad_kde_max": zona["densidad_kde_max"],
        "centroide_longitud": centroide_wgs84.x,
        "centroide_latitud": centroide_wgs84.y,
    })

tabla_zonas_calientes_finales = pd.DataFrame(filas_caracterizacion).sort_values(
    ["grupo_id", "cantidad_registros", "densidad_delitos_km2"],
    ascending=[True, False, False],
).reset_index(drop=True)
tabla_zonas_calientes_finales["area_km2"] = tabla_zonas_calientes_finales["area_km2"].round(4)
tabla_zonas_calientes_finales["densidad_delitos_km2"] = (
    tabla_zonas_calientes_finales["densidad_delitos_km2"].round(2)
)
tabla_zonas_calientes_finales["porcentaje_total_grupo"] = (
    tabla_zonas_calientes_finales["porcentaje_total_grupo"].round(2)
)
tabla_zonas_calientes_finales["centroide_longitud"] = (
    tabla_zonas_calientes_finales["centroide_longitud"].round(6)
)
tabla_zonas_calientes_finales["centroide_latitud"] = (
    tabla_zonas_calientes_finales["centroide_latitud"].round(6)
)

delitos_en_zonas_m = gpd.GeoDataFrame(
    pd.concat(asignaciones_delitos, ignore_index=True),
    geometry="geometry",
    crs=CRS_METRICO,
) if asignaciones_delitos else gpd.GeoDataFrame(columns=[], crs=CRS_METRICO)

tabla_zonas_calientes_finales.to_excel(
    salida_tablas / "tabla_zonas_calientes_finales.xlsx",
    index=False,
)
tabla_zonas_calientes_finales.to_csv(
    salida_tablas / "tabla_zonas_calientes_finales.csv",
    index=False,
    encoding="utf-8-sig",
)

tabla_zonas_tesis = tabla_zonas_calientes_finales[
    [
        "codigo_zona",
        "grupo_id",
        "cantidad_registros",
        "area_km2",
        "densidad_delitos_km2",
        "modalidad_predominante",
        "porcentaje_total_grupo",
    ]
].rename(columns={
    "codigo_zona": "Zona",
    "grupo_id": "Grupo",
    "cantidad_registros": "Registros",
    "area_km2": "Área (km²)",
    "densidad_delitos_km2": "Densidad (registros/km²)",
    "modalidad_predominante": "Modalidad predominante",
    "porcentaje_total_grupo": "Participación del grupo (%)",
})
tabla_zonas_tesis.to_excel(
    salida_tablas / "tabla_zonas_calientes_resumida_para_tesis.xlsx",
    index=False,
)
tabla_zonas_tesis

# %% [markdown]
# ## 5.6.5. Mapas finales de zonas calientes
# 
# %%
def obtener_proveedor(nombre):
    proveedores = {
        "CartoDB.Positron": cx.providers.CartoDB.Positron,
        "OpenStreetMap.Mapnik": cx.providers.OpenStreetMap.Mapnik,
        "Esri.WorldImagery": cx.providers.Esri.WorldImagery,
    }
    return proveedores[nombre]

def agregar_mapa_base_seguro(ax, proveedor, alternativas=None, zoom="auto", alpha=0.90):
    """Añade teselas solo como referencia cartográfica; no intervienen en el cálculo."""
    if not CONTEXTILY_DISPONIBLE:
        return False, "Sin mapa base"
    candidatos = [proveedor] + list(alternativas or [])
    for nombre in candidatos:
        try:
            cx.add_basemap(
                ax,
                source=obtener_proveedor(nombre),
                crs="EPSG:3857",
                zoom=zoom,
                alpha=alpha,
                attribution=False,
                reset_extent=False,
            )
            return True, nombre
        except Exception as exc:
            print(f"No se pudo cargar {nombre}: {exc}")
    return False, "Sin mapa base"

def texto_fuente_mapa(proveedor):
    if proveedor == "Sin mapa base":
        return "Fuente: Elaboración propia con registros 2020–2025; mapa alternativo sin teselas."
    nombres = {
        "CartoDB.Positron": "CartoDB Positron",
        "OpenStreetMap.Mapnik": "OpenStreetMap",
        "Esri.WorldImagery": "Esri World Imagery",
    }
    return (
        "Fuente: Elaboración propia con registros 2020–2025 y mapa base "
        f"{nombres.get(proveedor, proveedor)}."
    )

def etiquetar_zonas(ax, zonas, fontsize=8, desplazamiento=(0, 0)):
    for _, fila in zonas.iterrows():
        punto = fila.geometry.representative_point()
        ax.annotate(
            fila["codigo_zona"],
            (punto.x, punto.y),
            xytext=desplazamiento,
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#161616",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#555555", "linewidth": 0.3, "pad": 1.6},
            zorder=8,
        )

def preparar_capas_web():
    return (
        gdf_distrito_m.to_crs(epsg=3857),
        gdf_grilla_base_m.to_crs(epsg=3857),
        zonas_finales_m.to_crs(epsg=3857),
    )

def plot_mapa_final_zonas(grupo_id, nombre_archivo, titulo):
    distrito_web, grilla_web, zonas_web = preparar_capas_web()
    zonas = zonas_web[zonas_web["grupo_id"] == grupo_id].copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    distrito_web.plot(ax=ax, facecolor="none", edgecolor="#202020", linewidth=1.1, zorder=4)
    grilla_web.boundary.plot(ax=ax, color="#FFFFFF", linewidth=0.12, alpha=0.28, zorder=3)
    zonas.plot(
        ax=ax,
        color=COLORES_GRUPOS[grupo_id],
        edgecolor="#111111",
        linewidth=1.8,
        alpha=0.78,
        zorder=6,
    )
    puntos = zonas.geometry.representative_point()
    ax.scatter(
        puntos.x,
        puntos.y,
        s=24,
        color=COLORES_GRUPOS[grupo_id],
        edgecolor="white",
        linewidth=0.6,
        zorder=7,
    )
    etiquetar_zonas(ax, zonas)
    agregado, proveedor = agregar_mapa_base_seguro(
        ax,
        "CartoDB.Positron",
        alternativas=["OpenStreetMap.Mapnik"],
        zoom="auto",
        alpha=0.92,
    )
    if not agregado:
        distrito_web.plot(ax=ax, facecolor="#F2F2F2", edgecolor="#202020", linewidth=1.1, zorder=1)
        grilla_web.boundary.plot(ax=ax, color="#B5B5B5", linewidth=0.10, alpha=0.30, zorder=2)
        zonas.plot(ax=ax, color=COLORES_GRUPOS[grupo_id], edgecolor="#111111", alpha=0.78, zorder=6)
        etiquetar_zonas(ax, zonas)
    ax.set_title(titulo)
    ax.set_axis_off()
    ax.text(0.01, 0.01, texto_fuente_mapa(proveedor), transform=ax.transAxes, fontsize=8, zorder=10)
    guardar_figura(nombre_archivo)

plot_mapa_final_zonas(
    "g1",
    "figura_45_zonas_calientes_finales_g1.png",
    "Zonas calientes finales - G1: violencia contra la mujer y grupos vulnerables",
)
plot_mapa_final_zonas(
    "g2",
    "figura_46_zonas_calientes_finales_g2.png",
    "Zonas calientes finales - G2: delitos contra el patrimonio",
)

# %%
def plot_mapa_comparativo_zonas(nombre_archivo):
    distrito_web, grilla_web, zonas_web = preparar_capas_web()
    g1_web = zonas_web[zonas_web["grupo_id"] == "g1"].copy()
    g2_web = zonas_web[zonas_web["grupo_id"] == "g2"].copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    distrito_web.plot(ax=ax, facecolor="none", edgecolor="#202020", linewidth=1.1, zorder=4)
    grilla_web.boundary.plot(ax=ax, color="#FFFFFF", linewidth=0.12, alpha=0.25, zorder=3)
    g1_web.plot(
        ax=ax,
        facecolor=COLORES_GRUPOS["g1"],
        edgecolor="#7A1022",
        linewidth=1.7,
        alpha=0.50,
        zorder=6,
    )
    g2_web.plot(
        ax=ax,
        facecolor="none",
        edgecolor=COLORES_GRUPOS["g2"],
        linewidth=2.2,
        hatch="///",
        zorder=7,
    )
    etiquetar_zonas(ax, g1_web, desplazamiento=(-13, 8))
    etiquetar_zonas(ax, g2_web, desplazamiento=(13, -8))
    agregado, proveedor = agregar_mapa_base_seguro(
        ax,
        "CartoDB.Positron",
        alternativas=["OpenStreetMap.Mapnik"],
        zoom="auto",
        alpha=0.92,
    )
    if not agregado:
        distrito_web.plot(ax=ax, facecolor="#F2F2F2", edgecolor="#202020", linewidth=1.1, zorder=1)
    ax.legend(
        handles=[
            Patch(facecolor=COLORES_GRUPOS["g1"], edgecolor="#7A1022", alpha=0.55, label="G1"),
            Patch(facecolor="white", edgecolor=COLORES_GRUPOS["g2"], hatch="///", label="G2"),
        ],
        frameon=True,
        facecolor="white",
        framealpha=0.88,
        loc="lower right",
    )
    ax.set_title("Comparación de zonas calientes finales entre G1 y G2")
    ax.set_axis_off()
    ax.text(0.01, 0.01, texto_fuente_mapa(proveedor), transform=ax.transAxes, fontsize=8, zorder=10)
    guardar_figura(nombre_archivo)

def plot_zoom_urbano_por_grupo(grupo_id, nombre_archivo):
    """Genera paneles de acercamiento para los focos de un solo grupo."""
    distrito_web, grilla_web, zonas_web = preparar_capas_web()
    zonas_grupo = zonas_web[zonas_web["grupo_id"] == grupo_id].copy()
    centroides = zonas_grupo.geometry.centroid
    coords = np.column_stack([centroides.x, centroides.y])
    zonas_grupo["sector_zoom"] = DBSCAN(eps=3000, min_samples=1).fit_predict(coords)
    sectores = sorted(zonas_grupo["sector_zoom"].unique())
    fig, axes = plt.subplots(1, len(sectores), figsize=(5.2 * len(sectores), 4.2))
    if len(sectores) == 1:
        axes = [axes]
    proveedores_usados = []

    for numero, (ax, sector) in enumerate(zip(axes, sectores), start=1):
        sub = zonas_grupo[zonas_grupo["sector_zoom"] == sector].copy()
        minx, miny, maxx, maxy = expandir_limites(sub.total_bounds, 0.75)
        grilla_local = grilla_web.cx[minx:maxx, miny:maxy]
        distrito_local = distrito_web.cx[minx:maxx, miny:maxy]
        distrito_local.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#303030",
            linewidth=0.9,
            zorder=4,
        )
        grilla_local.boundary.plot(
            ax=ax,
            color="#FFFFFF",
            linewidth=0.35,
            alpha=0.48,
            zorder=3,
        )
        sub.plot(
            ax=ax,
            facecolor=COLORES_GRUPOS[grupo_id],
            edgecolor="#111111",
            linewidth=1.8,
            alpha=0.68,
            zorder=6,
        )
        etiquetar_zonas(ax, sub, fontsize=8)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        agregado, proveedor = agregar_mapa_base_seguro(
            ax,
            "OpenStreetMap.Mapnik",
            alternativas=["Esri.WorldImagery", "CartoDB.Positron"],
            zoom="auto",
            alpha=0.88,
        )
        proveedores_usados.append(proveedor)
        if not agregado:
            distrito_local.plot(
                ax=ax,
                facecolor="#F4F4F4",
                edgecolor="#303030",
                linewidth=0.9,
                zorder=1,
            )
            grilla_local.boundary.plot(
                ax=ax,
                color="#AAAAAA",
                linewidth=0.25,
                alpha=0.5,
                zorder=2,
            )
        ax.set_title(f"Sector de concentración {numero}")
        ax.set_axis_off()

    fig.legend(
        handles=[
            Patch(
                facecolor=COLORES_GRUPOS[grupo_id],
                edgecolor="#111111",
                alpha=0.68,
                label=grupo_id.upper(),
            ),
        ],
        frameon=True,
        facecolor="white",
        framealpha=0.90,
        loc="lower center",
    )
    fig.suptitle(
        f"Acercamiento por sectores - {grupo_id.upper()}: {NOMBRES_GRUPOS[grupo_id]}",
        fontsize=13,
        y=0.96,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.16, top=0.84, wspace=0.08)
    proveedores_validos = [p for p in proveedores_usados if p != "Sin mapa base"]
    proveedor_fuente = proveedores_validos[0] if proveedores_validos else "Sin mapa base"
    fig.text(0.01, 0.01, texto_fuente_mapa(proveedor_fuente), fontsize=8)
    guardar_figura(nombre_archivo)

def plot_zoom_urbano_zonas(nombre_archivo):
    distrito_web, grilla_web, zonas_web = preparar_capas_web()
    centroides = zonas_web.geometry.centroid
    coords = np.column_stack([centroides.x, centroides.y])
    zonas_zoom = zonas_web.copy()
    zonas_zoom["sector_zoom"] = DBSCAN(eps=3000, min_samples=1).fit_predict(coords)
    sectores = sorted(zonas_zoom["sector_zoom"].unique())
    fig, axes = plt.subplots(1, len(sectores), figsize=(5.5 * len(sectores), 6))
    if len(sectores) == 1:
        axes = [axes]
    proveedores_usados = []

    for numero, (ax, sector) in enumerate(zip(axes, sectores), start=1):
        sub = zonas_zoom[zonas_zoom["sector_zoom"] == sector].copy()
        minx, miny, maxx, maxy = expandir_limites(sub.total_bounds, 0.75)
        grilla_local = grilla_web.cx[minx:maxx, miny:maxy]
        distrito_local = distrito_web.cx[minx:maxx, miny:maxy]
        distrito_local.plot(ax=ax, facecolor="none", edgecolor="#303030", linewidth=0.9, zorder=4)
        grilla_local.boundary.plot(ax=ax, color="#FFFFFF", linewidth=0.35, alpha=0.48, zorder=3)
        for grupo_id, desplazamiento in [("g1", (-11, 8)), ("g2", (11, -8))]:
            zonas = sub[sub["grupo_id"] == grupo_id]
            if zonas.empty:
                continue
            if grupo_id == "g1":
                zonas.plot(
                    ax=ax,
                    facecolor=COLORES_GRUPOS[grupo_id],
                    edgecolor="#111111",
                    linewidth=1.5,
                    alpha=0.58,
                    zorder=6,
                )
            else:
                zonas.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor=COLORES_GRUPOS[grupo_id],
                    linewidth=2.2,
                    hatch="///",
                    zorder=7,
                )
            etiquetar_zonas(ax, zonas, fontsize=8, desplazamiento=desplazamiento)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        agregado, proveedor = agregar_mapa_base_seguro(
            ax,
            "OpenStreetMap.Mapnik",
            alternativas=["Esri.WorldImagery", "CartoDB.Positron"],
            zoom="auto",
            alpha=0.88,
        )
        proveedores_usados.append(proveedor)
        if not agregado:
            distrito_local.plot(ax=ax, facecolor="#F4F4F4", edgecolor="#303030", linewidth=0.9, zorder=1)
            grilla_local.boundary.plot(ax=ax, color="#AAAAAA", linewidth=0.25, alpha=0.5, zorder=2)
        ax.set_title(f"Sector de concentración {numero}")
        ax.set_axis_off()

    fig.legend(
        handles=[
            Patch(facecolor=COLORES_GRUPOS["g1"], edgecolor="#111111", alpha=0.58, label="G1"),
            Patch(facecolor="white", edgecolor=COLORES_GRUPOS["g2"], hatch="///", label="G2"),
        ],
        frameon=True,
        facecolor="white",
        framealpha=0.90,
        loc="lower center",
        ncol=2,
    )
    fig.suptitle("Acercamiento por sectores de concentración de zonas calientes finales", fontsize=13)
    proveedores_validos = [p for p in proveedores_usados if p != "Sin mapa base"]
    proveedor_fuente = proveedores_validos[0] if proveedores_validos else "Sin mapa base"
    fig.text(0.01, 0.01, texto_fuente_mapa(proveedor_fuente), fontsize=8)
    guardar_figura(nombre_archivo)

plot_mapa_comparativo_zonas("figura_47_comparacion_zonas_calientes_finales.png")
plot_zoom_urbano_por_grupo(
    "g1",
    "figura_48a_zoom_urbano_zonas_calientes_g1.png",
)
plot_zoom_urbano_por_grupo(
    "g2",
    "figura_48b_zoom_urbano_zonas_calientes_g2.png",
)
plot_zoom_urbano_zonas("figura_48_zoom_urbano_zonas_calientes_finales.png")

# %% [markdown]
# ## 5.6.6. Exportación de insumos finales
# 
# %%
zonas_finales_wgs84 = zonas_finales_m.to_crs(epsg=4326)
zonas_finales_wgs84.to_file(
    salida_datasets / "zonas_calientes_finales.geojson",
    driver="GeoJSON",
)

ruta_gpkg = salida_datasets / "zonas_calientes_finales.gpkg"
zonas_finales_wgs84.to_file(ruta_gpkg, layer="zonas_finales", driver="GPKG")
celdas_hotspot_finales_m.to_crs(epsg=4326).to_file(
    ruta_gpkg,
    layer="celdas_hotspot",
    driver="GPKG",
)
centros_hexagonos_wgs84.to_file(
    ruta_gpkg,
    layer="centros_hexagonos_hotspot",
    driver="GPKG",
)
if not delitos_en_zonas_m.empty:
    delitos_en_zonas_m.to_crs(epsg=4326).to_file(
        ruta_gpkg,
        layer="delitos_en_zonas",
        driver="GPKG",
    )

if FOLIUM_DISPONIBLE:
    centro = zonas_finales_wgs84.geometry.unary_union.centroid
    mapa = folium.Map(
        location=[centro.y, centro.x],
        zoom_start=13,
        tiles="CartoDB positron",
    )
    folium.GeoJson(
        gdf_distrito.to_crs(epsg=4326),
        name="Distrito de Huaral",
        style_function=lambda _: {
            "fillOpacity": 0,
            "color": "#555555",
            "weight": 1,
        },
    ).add_to(mapa)
    for grupo_id in GRUPOS_EVALUACION:
        capa = zonas_finales_wgs84[zonas_finales_wgs84["grupo_id"] == grupo_id]
        folium.GeoJson(
            capa,
            name=f"Zonas {grupo_id.upper()}",
            style_function=lambda _, color=COLORES_GRUPOS[grupo_id]: {
                "fillColor": color,
                "color": "#202020",
                "weight": 1.5,
                "fillOpacity": 0.65,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["codigo_zona", "grupo", "numero_celdas", "conteo_grilla"],
                aliases=["Zona", "Grupo", "Celdas", "Conteo en grilla"],
            ),
        ).add_to(mapa)
    folium.LayerControl().add_to(mapa)
    mapa.save(salida_datasets / "mapa_interactivo_zonas_calientes_finales.html")

archivos_exportados = pd.DataFrame({
    "producto": [
        "Zonas finales GeoJSON",
        "Zonas y capas GeoPackage",
        "Tabla completa Excel",
        "Tabla resumida para tesis",
        "Mapa interactivo",
        "Figuras estáticas",
    ],
    "ruta": [
        str(salida_datasets / "zonas_calientes_finales.geojson"),
        str(ruta_gpkg),
        str(salida_tablas / "tabla_zonas_calientes_finales.xlsx"),
        str(salida_tablas / "tabla_zonas_calientes_resumida_para_tesis.xlsx"),
        str(salida_datasets / "mapa_interactivo_zonas_calientes_finales.html"),
        str(salida_figuras),
    ],
})
archivos_exportados.to_excel(salida_tablas / "inventario_productos_etapa56.xlsx", index=False)
archivos_exportados

# %% [markdown]
# ## 5.6.7. Texto interpretativo automático para apoyar la redacción
# 
# %%
textos_interpretativos = []

for grupo_id in GRUPOS_EVALUACION:
    sub = tabla_zonas_calientes_finales[
        tabla_zonas_calientes_finales["grupo_id"] == grupo_id
    ].copy()
    total_zonas = len(sub)
    registros_zonas = int(sub["cantidad_registros"].sum())
    total_grupo = len(gdf_delitos_m[gdf_delitos_m["grupo_id"].astype(str) == grupo_id])
    pct = registros_zonas / total_grupo * 100 if total_grupo else np.nan
    prioritaria_registros = sub.sort_values("cantidad_registros", ascending=False).iloc[0]
    prioritaria_densidad = sub.sort_values("densidad_delitos_km2", ascending=False).iloc[0]
    modalidades = ", ".join(
        sub["modalidad_predominante"].dropna().astype(str).drop_duplicates().tolist()
    )
    textos_interpretativos.extend([
        {
            "tema": f"Número y cobertura de zonas {grupo_id.upper()}",
            "texto": (
                f"Para {NOMBRES_GRUPOS[grupo_id]} se identificaron {total_zonas} zonas calientes "
                f"finales, que contienen {registros_zonas} registros, equivalentes al {pct:.2f}% "
                "del total del grupo."
            ),
        },
        {
            "tema": f"Prioridad por registros {grupo_id.upper()}",
            "texto": (
                f"La zona {prioritaria_registros['codigo_zona']} presenta la mayor cantidad de "
                f"registros ({int(prioritaria_registros['cantidad_registros'])}) dentro de {grupo_id.upper()}."
            ),
        },
        {
            "tema": f"Prioridad por densidad {grupo_id.upper()}",
            "texto": (
                f"La zona {prioritaria_densidad['codigo_zona']} alcanza la mayor densidad, con "
                f"{prioritaria_densidad['densidad_delitos_km2']:.2f} registros por km²."
            ),
        },
        {
            "tema": f"Modalidades predominantes {grupo_id.upper()}",
            "texto": f"Las modalidades predominantes observadas en las zonas de {grupo_id.upper()} son: {modalidades}.",
        },
    ])

resumen_grupos = (
    tabla_zonas_calientes_finales.groupby(["grupo_id", "grupo"], as_index=False)
    .agg(
        numero_zonas=("codigo_zona", "count"),
        registros_en_zonas=("cantidad_registros", "sum"),
        area_total_km2=("area_km2", "sum"),
    )
)
grupo_mayor = resumen_grupos.sort_values("registros_en_zonas", ascending=False).iloc[0]
textos_interpretativos.append({
    "tema": "Comparación entre grupos",
    "texto": (
        f"El grupo {grupo_mayor['grupo_id'].upper()} concentra la mayor cantidad de registros "
        f"dentro de zonas calientes finales, con {int(grupo_mayor['registros_en_zonas'])} casos."
    ),
})
textos_interpretativos.append({
    "tema": "Localización territorial",
    "texto": (
        "Los mapas muestran que las zonas finales se concentran en un sector reducido del distrito. "
        "Esta concentración coincide visualmente con el núcleo de mayor presencia de registros, "
        "pero no se clasifica formalmente como zona urbana por no contar con una delimitación urbana oficial."
    ),
})

textos_interpretativos_df = pd.DataFrame(textos_interpretativos)
textos_interpretativos_df.to_excel(
    salida_tablas / "textos_interpretativos_etapa56.xlsx",
    index=False,
)
textos_interpretativos_df.to_csv(
    salida_tablas / "textos_interpretativos_etapa56.csv",
    index=False,
    encoding="utf-8-sig",
)
resumen_grupos.to_excel(salida_tablas / "resumen_zonas_por_grupo.xlsx", index=False)
textos_interpretativos_df
