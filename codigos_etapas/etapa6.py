from pathlib import Path
# %% [markdown]
# # 5.5. Evaluación y validación de las técnicas aplicadas
# 
# %%
from pathlib import Path
import os
import time
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
from shapely.geometry import Polygon
from shapely.ops import unary_union

from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import hdbscan
    HDBSCAN_LIB_DISPONIBLE = True
except Exception as exc:
    HDBSCAN_LIB_DISPONIBLE = False
    HDBSCAN_LIB_ERROR = exc

try:
    from libpysal.weights import Queen
    from esda.moran import Moran, Moran_Local
    from esda.getisord import G_Local
    PYSAL_DISPONIBLE = True
except Exception as exc:
    PYSAL_DISPONIBLE = False
    PYSAL_ERROR = exc

plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

salida_base = Path("outputs_etapa55")
salida_figuras = salida_base / "figuras"
salida_tablas = salida_base / "tablas"
salida_datasets = salida_base / "datasets"
for carpeta in [salida_figuras, salida_tablas, salida_datasets]:
    carpeta.mkdir(parents=True, exist_ok=True)

CRS_METRICO = "EPSG:32718"
TAMANO_HEXAGONO_M = None
TAMANO_HEXAGONO_SELECCIONADO_M = None
TAMANOS_SENSIBILIDAD_M = [100, 150, 200, 250, 500]
MAX_HEXAGONOS_ADMISIBLE = 10000
MIN_CELDAS_OCUPADAS_PROMEDIO = 50
CUOTA_AREA_SELECCION_GRILLA = 0.05
PESOS_SELECCION_GRILLA = {
    "predictivo": 0.40,
    "fragmentacion": 0.25,
    "conectividad": 0.15,
    "detalle": 0.20,
}
CUOTAS_AREA_HOTSPOT = [0.01, 0.05, 0.10]
ANIO_VALIDACION_TEMPORAL = 2025
ANIO_CORTE_ENTRENAMIENTO = 2024
ALPHA_SIGNIFICANCIA = 0.05
GRUPOS_EVALUACION = ["g1", "g2"]
NOMBRES_GRUPOS = {
    "g1": "Violencia contra la mujer y grupos vulnerables",
    "g2": "Patrimonio",
    "g3": "Otros hechos de baja frecuencia",
}
COLORES_TECNICAS = {"KDE": "#E15759", "DBSCAN": "#4C78A8", "HDBSCAN": "#59A14F"}
COLORES_LISA = {
    "Alto-Alto": "#B2182B",
    "Bajo-Bajo": "#2166AC",
    "Alto-Bajo": "#EF8A62",
    "Bajo-Alto": "#67A9CF",
    "No significativo": "#D9D9D9",
}

def guardar_figura(nombre, dpi=300):
    ruta = salida_figuras / nombre
    plt.tight_layout()
    plt.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.show()
    return ruta

print("PySAL disponible:", PYSAL_DISPONIBLE)
print("hdbscan disponible:", HDBSCAN_LIB_DISPONIBLE)
if not PYSAL_DISPONIBLE:
    print("Instalar dependencias requeridas para autocorrelación espacial: pip install libpysal esda")


# %% [markdown]
# ## 5.5.1. Carga de resultados de la etapa 5.4
# 
# %%
ruta_puntos_resultados = Path("outputs_etapa54/datasets/puntos_resultados_tecnicas_por_grupo.geojson")
ruta_distrito = Path("outputs_etapa3/datasets/distrito_huaral_gadm41.geojson")
ruta_kde_parametros = Path("outputs_etapa54/tablas/kde_resultados_bandwidth_por_grupo.xlsx")
ruta_dbscan_resumen = Path("outputs_etapa54/tablas/dbscan_resumen_modelos_por_grupo.xlsx")
ruta_hdbscan_resumen = Path("outputs_etapa54/tablas/hdbscan_resumen_modelos_por_grupo.xlsx")

if not ruta_puntos_resultados.exists():
    raise FileNotFoundError(f"No se encontró {ruta_puntos_resultados}. Ejecutar primero la etapa 5.4.")
if not ruta_distrito.exists():
    raise FileNotFoundError(f"No se encontró {ruta_distrito}. Ejecutar primero la etapa exploratoria espacial.")

gdf_puntos = gpd.read_file(ruta_puntos_resultados).copy()
gdf_distrito = gpd.read_file(ruta_distrito).copy()

if gdf_puntos.crs is None:
    gdf_puntos = gdf_puntos.set_crs("EPSG:4326")
if gdf_distrito.crs is None:
    gdf_distrito = gdf_distrito.set_crs("EPSG:4326")

gdf_puntos_m = gdf_puntos.to_crs(CRS_METRICO).copy()
gdf_distrito_m = gdf_distrito.to_crs(CRS_METRICO).copy()
gdf_puntos_eval_m = gdf_puntos_m[gdf_puntos_m["grupo_id"].astype(str).isin(GRUPOS_EVALUACION)].copy()

if "fecha_hora_hecho" in gdf_puntos_eval_m.columns:
    gdf_puntos_eval_m["fecha_hora_hecho"] = pd.to_datetime(gdf_puntos_eval_m["fecha_hora_hecho"], errors="coerce")
if "año_hecho" not in gdf_puntos_eval_m.columns and "fecha_hora_hecho" in gdf_puntos_eval_m.columns:
    gdf_puntos_eval_m["año_hecho"] = gdf_puntos_eval_m["fecha_hora_hecho"].dt.year

kde_parametros = pd.read_excel(ruta_kde_parametros) if ruta_kde_parametros.exists() else pd.DataFrame()
dbscan_resumen_previo = pd.read_excel(ruta_dbscan_resumen) if ruta_dbscan_resumen.exists() else pd.DataFrame()
hdbscan_resumen_previo = pd.read_excel(ruta_hdbscan_resumen) if ruta_hdbscan_resumen.exists() else pd.DataFrame()

resumen_carga = (
    gdf_puntos_eval_m.groupby(["grupo_id", "grupo"], dropna=False)
    .size()
    .reset_index(name="registros")
)
resumen_carga


# %% [markdown]
# ## 5.5.1.a. Diagnóstico de coordenadas repetidas y concentración artificial
# %%
def diagnosticar_coordenadas_repetidas(puntos_m, decimales=7):
    puntos = puntos_m.copy()
    puntos["lon_redondeada"] = puntos.to_crs(epsg=4326).geometry.x.round(decimales)
    puntos["lat_redondeada"] = puntos.to_crs(epsg=4326).geometry.y.round(decimales)
    filas_resumen = []
    filas_top = []
    for grupo_id, sub in puntos.groupby(puntos["grupo_id"].astype(str)):
        grupo_nombre = sub["grupo"].iloc[0] if "grupo" in sub.columns and not sub.empty else NOMBRES_GRUPOS.get(grupo_id, grupo_id)
        conteos = (
            sub.groupby(["grupo_id", "grupo", "lon_redondeada", "lat_redondeada"], dropna=False)
            .size()
            .reset_index(name="registros_en_coordenada")
            .sort_values("registros_en_coordenada", ascending=False)
        )
        total = len(sub)
        repetidos = int(conteos.loc[conteos["registros_en_coordenada"] > 1, "registros_en_coordenada"].sum())
        max_conc = int(conteos["registros_en_coordenada"].max()) if not conteos.empty else 0
        filas_resumen.append({
            "grupo_id": grupo_id,
            "grupo": grupo_nombre,
            "registros": total,
            "coordenadas_unicas": len(conteos),
            "registros_en_coordenadas_repetidas": repetidos,
            "pct_registros_en_coordenadas_repetidas": round(repetidos / total * 100, 2) if total else np.nan,
            "max_registros_en_una_coordenada": max_conc,
            "pct_max_coordenada": round(max_conc / total * 100, 2) if total else np.nan,
            "nota": "Riesgo alto de concentración artificial" if total and max_conc / total >= 0.10 else "Revisar como control de calidad espacial",
        })
        filas_top.append(conteos.head(20))
    resumen = pd.DataFrame(filas_resumen)
    top = pd.concat(filas_top, ignore_index=True) if filas_top else pd.DataFrame()
    return resumen, top

resumen_coordenadas_repetidas, top_coordenadas_repetidas = diagnosticar_coordenadas_repetidas(gdf_puntos_eval_m)
resumen_coordenadas_repetidas.to_excel(salida_tablas / "diagnostico_coordenadas_repetidas_por_grupo.xlsx", index=False)
resumen_coordenadas_repetidas.to_csv(salida_tablas / "diagnostico_coordenadas_repetidas_por_grupo.csv", index=False, encoding="utf-8-sig")
top_coordenadas_repetidas.to_excel(salida_tablas / "top_coordenadas_repetidas_por_grupo.xlsx", index=False)
resumen_coordenadas_repetidas

# %% [markdown]
# ## 5.5.2. Selección previa de la grilla hexagonal
# 
# 

# %%
def crear_hexagono(cx, cy, lado_m):
    angulos = np.deg2rad(np.arange(0, 360, 60) + 30)
    return Polygon([(cx + lado_m * np.cos(a), cy + lado_m * np.sin(a)) for a in angulos])


def crear_grilla_hexagonal(poligono_gdf, lado_m=100):
    limite = unary_union(poligono_gdf.geometry)
    minx, miny, maxx, maxy = limite.bounds
    dx = np.sqrt(3) * lado_m
    dy = 1.5 * lado_m
    filas = []
    y = miny - dy
    fila = 0
    while y <= maxy + dy:
        x_offset = 0 if fila % 2 == 0 else dx / 2
        x = minx - dx + x_offset
        while x <= maxx + dx:
            hexagono = crear_hexagono(x, y, lado_m)
            if hexagono.intersects(limite):
                geom = hexagono.intersection(limite)
                if not geom.is_empty and geom.area > 0:
                    filas.append({"hex_id": len(filas) + 1, "geometry": geom})
            x += dx
        y += dy
        fila += 1
    return gpd.GeoDataFrame(filas, crs=poligono_gdf.crs)


def agregar_conteos_por_grupo(hex_base, puntos_m, grupos=GRUPOS_EVALUACION):
    grilla = hex_base.copy()
    for grupo_id in grupos:
        puntos_grupo = puntos_m[puntos_m["grupo_id"].astype(str) == grupo_id].copy()
        col = f"delitos_{grupo_id}"
        if puntos_grupo.empty:
            grilla[col] = 0
            continue
        join = gpd.sjoin(
            puntos_grupo[["grupo_id", "geometry"]],
            grilla[["hex_id", "geometry"]],
            how="left",
            predicate="within",
        )
        conteos = join.groupby("hex_id").size()
        grilla[col] = grilla["hex_id"].map(conteos).fillna(0).astype(int)
    grilla["delitos_total_eval"] = grilla[[f"delitos_{g}" for g in grupos]].sum(axis=1)
    return grilla


def normalizar_minmax(serie, mayor_es_mejor=True):
    serie = pd.to_numeric(serie, errors="coerce")
    if serie.notna().sum() == 0:
        return pd.Series(0.5, index=serie.index)
    minimo, maximo = serie.min(), serie.max()
    if np.isclose(minimo, maximo):
        base = pd.Series(0.5, index=serie.index)
    else:
        base = (serie - minimo) / (maximo - minimo)
    return base if mayor_es_mejor else 1 - base


def seleccionar_por_cuota_area(capa, valor_col, cuota):
    orden = capa.copy()
    orden["_valor_orden"] = pd.to_numeric(orden[valor_col], errors="coerce").fillna(0)
    orden["_area_m2"] = orden.geometry.area
    orden = orden.sort_values(["_valor_orden", "hex_id"], ascending=[False, True]).copy()
    objetivo = orden["_area_m2"].sum() * cuota
    acumulada_previa = orden["_area_m2"].cumsum() - orden["_area_m2"]
    orden["_seleccion"] = acumulada_previa < objetivo
    ids = set(orden.loc[orden["_seleccion"], "hex_id"])
    seleccion = capa["hex_id"].isin(ids)
    area_real = capa.loc[seleccion].geometry.area.sum() / capa.geometry.area.sum()
    return seleccion, float(area_real)


# %% [markdown]
# ### Evaluación multiescala y selección reproducible
# 
# %%
def evaluar_grilla_previa(grilla, puntos_m, tamano_m):
    fila = {
        "tamano_lado_m": tamano_m,
        "numero_hexagonos": len(grilla),
        "hexagonos_vacios_total_eval": int((grilla["delitos_total_eval"] == 0).sum()),
        "pct_hexagonos_vacios_total_eval": round((grilla["delitos_total_eval"] == 0).mean() * 100, 2),
    }
    if PYSAL_DISPONIBLE:
        w = Queen.from_dataframe(grilla, ids=grilla.index.tolist(), silence_warnings=True)
        fila["celdas_aisladas"] = len(w.islands)
        fila["pct_celdas_aisladas"] = len(w.islands) / len(grilla) * 100
    else:
        fila["celdas_aisladas"] = np.nan
        fila["pct_celdas_aisladas"] = np.nan

    puntos = puntos_m.copy()
    puntos["año_hecho"] = pd.to_numeric(puntos["año_hecho"], errors="coerce")
    for grupo_id in GRUPOS_EVALUACION:
        col = f"delitos_{grupo_id}"
        valores = grilla[col].astype(float)
        ocupadas = valores[valores > 0]
        total_grupo = valores.sum()
        fila[f"hexagonos_con_delitos_{grupo_id}"] = int((valores > 0).sum())
        fila[f"pct_celdas_unitarias_{grupo_id}"] = float((ocupadas == 1).mean() * 100) if len(ocupadas) else np.nan
        fila[f"mediana_delitos_celda_ocupada_{grupo_id}"] = float(ocupadas.median()) if len(ocupadas) else np.nan
        fila[f"max_delitos_hexagono_{grupo_id}"] = int(valores.max())
        fila[f"pct_delitos_celda_dominante_{grupo_id}"] = float(valores.max() / total_grupo * 100) if total_grupo else np.nan

        if PYSAL_DISPONIBLE and valores.std() > 0:
            w_moran = Queen.from_dataframe(grilla, ids=grilla.index.tolist(), silence_warnings=True)
            w_moran.transform = "r"
            moran = Moran(valores.to_numpy(), w_moran, permutations=199)
            fila[f"moran_conteos_{grupo_id}"] = float(moran.I)
            fila[f"moran_p_{grupo_id}"] = float(moran.p_sim)
        else:
            fila[f"moran_conteos_{grupo_id}"] = np.nan
            fila[f"moran_p_{grupo_id}"] = np.nan

        train = puntos[
            (puntos["grupo_id"].astype(str) == grupo_id)
            & (puntos["año_hecho"] <= ANIO_CORTE_ENTRENAMIENTO)
        ].copy()
        test = puntos[
            (puntos["grupo_id"].astype(str) == grupo_id)
            & (puntos["año_hecho"] == ANIO_VALIDACION_TEMPORAL)
        ].copy()
        join_train = gpd.sjoin(
            train[["geometry"]],
            grilla[["hex_id", "geometry"]],
            how="left",
            predicate="within",
        )
        conteo_train = join_train.groupby("hex_id").size()
        capa_train = grilla[["hex_id", "geometry"]].copy()
        capa_train["conteo_train"] = capa_train["hex_id"].map(conteo_train).fillna(0)
        seleccion, cuota_real = seleccionar_por_cuota_area(
            capa_train,
            "conteo_train",
            CUOTA_AREA_SELECCION_GRILLA,
        )
        join_test = gpd.sjoin(
            test[["geometry"]],
            grilla[["hex_id", "geometry"]],
            how="left",
            predicate="within",
        )
        conteo_test = join_test.groupby("hex_id").size()
        test_por_hex = grilla["hex_id"].map(conteo_test).fillna(0)
        capturados = float(test_por_hex[seleccion].sum())
        hit_rate = capturados / len(test) if len(test) else np.nan
        pai = hit_rate / cuota_real if cuota_real > 0 and np.isfinite(hit_rate) else np.nan
        fila[f"cuota_area_real_{grupo_id}"] = cuota_real
        fila[f"hit_rate_2025_{grupo_id}"] = hit_rate
        fila[f"pai_2025_{grupo_id}"] = pai
    return fila


resumen_sensibilidad_grilla = []
grillas_sensibilidad = {}
for tamano_m in TAMANOS_SENSIBILIDAD_M:
    grilla_tmp = crear_grilla_hexagonal(gdf_distrito_m, lado_m=tamano_m)
    grilla_tmp = agregar_conteos_por_grupo(grilla_tmp, gdf_puntos_eval_m)
    grillas_sensibilidad[tamano_m] = grilla_tmp
    grilla_tmp.to_crs(epsg=4326).to_file(
        salida_datasets / f"grilla_hexagonal_{tamano_m}m_conteo_delitos.geojson",
        driver="GeoJSON",
    )
    resumen_sensibilidad_grilla.append(
        evaluar_grilla_previa(grilla_tmp, gdf_puntos_eval_m, tamano_m)
    )

resumen_sensibilidad_grilla = pd.DataFrame(resumen_sensibilidad_grilla)
resumen_sensibilidad_grilla["celdas_ocupadas_promedio"] = resumen_sensibilidad_grilla[
    [f"hexagonos_con_delitos_{g}" for g in GRUPOS_EVALUACION]
].mean(axis=1)
resumen_sensibilidad_grilla["pct_celdas_unitarias_promedio"] = resumen_sensibilidad_grilla[
    [f"pct_celdas_unitarias_{g}" for g in GRUPOS_EVALUACION]
].mean(axis=1)
resumen_sensibilidad_grilla["pai_2025_promedio"] = resumen_sensibilidad_grilla[
    [f"pai_2025_{g}" for g in GRUPOS_EVALUACION]
].mean(axis=1)
resumen_sensibilidad_grilla["cumple_costo"] = (
    resumen_sensibilidad_grilla["numero_hexagonos"] <= MAX_HEXAGONOS_ADMISIBLE
)
resumen_sensibilidad_grilla["cumple_detalle_minimo"] = (
    resumen_sensibilidad_grilla["celdas_ocupadas_promedio"] >= MIN_CELDAS_OCUPADAS_PROMEDIO
)
resumen_sensibilidad_grilla["admisible"] = (
    resumen_sensibilidad_grilla["cumple_costo"]
    & resumen_sensibilidad_grilla["cumple_detalle_minimo"]
)

admisibles = resumen_sensibilidad_grilla["admisible"]
if not admisibles.any():
    raise ValueError("Ninguna grilla cumple los criterios mínimos de admisibilidad.")

resumen_sensibilidad_grilla["score_predictivo"] = np.nan
resumen_sensibilidad_grilla["score_fragmentacion"] = np.nan
resumen_sensibilidad_grilla["score_conectividad"] = np.nan
resumen_sensibilidad_grilla["score_detalle"] = np.nan
resumen_sensibilidad_grilla.loc[admisibles, "score_predictivo"] = normalizar_minmax(
    resumen_sensibilidad_grilla.loc[admisibles, "pai_2025_promedio"], True
)
resumen_sensibilidad_grilla.loc[admisibles, "score_fragmentacion"] = normalizar_minmax(
    resumen_sensibilidad_grilla.loc[admisibles, "pct_celdas_unitarias_promedio"], False
)
resumen_sensibilidad_grilla.loc[admisibles, "score_conectividad"] = normalizar_minmax(
    resumen_sensibilidad_grilla.loc[admisibles, "pct_celdas_aisladas"], False
)
resumen_sensibilidad_grilla.loc[admisibles, "score_detalle"] = normalizar_minmax(
    resumen_sensibilidad_grilla.loc[admisibles, "celdas_ocupadas_promedio"], True
)
resumen_sensibilidad_grilla["score_seleccion_grilla"] = (
    resumen_sensibilidad_grilla["score_predictivo"] * PESOS_SELECCION_GRILLA["predictivo"]
    + resumen_sensibilidad_grilla["score_fragmentacion"] * PESOS_SELECCION_GRILLA["fragmentacion"]
    + resumen_sensibilidad_grilla["score_conectividad"] * PESOS_SELECCION_GRILLA["conectividad"]
    + resumen_sensibilidad_grilla["score_detalle"] * PESOS_SELECCION_GRILLA["detalle"]
)
resumen_sensibilidad_grilla["ranking_grilla"] = np.nan
resumen_sensibilidad_grilla.loc[admisibles, "ranking_grilla"] = (
    resumen_sensibilidad_grilla.loc[admisibles, "score_seleccion_grilla"]
    .rank(ascending=False, method="dense")
)

fila_seleccionada = (
    resumen_sensibilidad_grilla[admisibles]
    .sort_values(["score_seleccion_grilla", "tamano_lado_m"], ascending=[False, True])
    .iloc[0]
)
TAMANO_HEXAGONO_SELECCIONADO_M = int(fila_seleccionada["tamano_lado_m"])
TAMANO_HEXAGONO_M = TAMANO_HEXAGONO_SELECCIONADO_M
hexagonos_conteo_m = grillas_sensibilidad[TAMANO_HEXAGONO_SELECCIONADO_M].copy()
hexagonos_base_m = hexagonos_conteo_m.copy()

resumen_sensibilidad_grilla["seleccionada"] = (
    resumen_sensibilidad_grilla["tamano_lado_m"] == TAMANO_HEXAGONO_SELECCIONADO_M
)
resumen_sensibilidad_grilla["justificacion_seleccion"] = np.where(
    resumen_sensibilidad_grilla["seleccionada"],
    (
        "Mayor puntaje entre escalas admisibles, combinando predicción temporal, "
        "fragmentación, conectividad y detalle espacial para todo el distrito."
    ),
    "",
)
resumen_sensibilidad_grilla.to_excel(
    salida_tablas / "seleccion_tamano_grilla_hexagonal.xlsx",
    index=False,
)
resumen_sensibilidad_grilla.to_csv(
    salida_tablas / "seleccion_tamano_grilla_hexagonal.csv",
    index=False,
    encoding="utf-8-sig",
)

criterios_seleccion_grilla = pd.DataFrame({
    "criterio": [
        "Área de estudio",
        "Máximo de hexágonos admisible",
        "Mínimo promedio de celdas ocupadas",
        "Peso predictivo temporal",
        "Peso fragmentación",
        "Peso conectividad",
        "Peso detalle espacial",
        "Tamaño seleccionado (m)",
    ],
    "valor": [
        "Distrito completo de Huaral",
        MAX_HEXAGONOS_ADMISIBLE,
        MIN_CELDAS_OCUPADAS_PROMEDIO,
        PESOS_SELECCION_GRILLA["predictivo"],
        PESOS_SELECCION_GRILLA["fragmentacion"],
        PESOS_SELECCION_GRILLA["conectividad"],
        PESOS_SELECCION_GRILLA["detalle"],
        TAMANO_HEXAGONO_SELECCIONADO_M,
    ],
})
criterios_seleccion_grilla.to_excel(
    salida_tablas / "criterios_seleccion_grilla_hexagonal.xlsx",
    index=False,
)

resumen_grilla = pd.DataFrame({
    "indicador": [
        "Tamaño seleccionado del lado del hexágono (m)",
        "Número de hexágonos",
        "Hexágonos con delitos g1",
        "Hexágonos con delitos g2",
        "Hexágonos vacíos en grupos evaluados",
        "Porcentaje de hexágonos vacíos",
        "Puntaje de selección",
    ],
    "valor": [
        TAMANO_HEXAGONO_SELECCIONADO_M,
        len(hexagonos_conteo_m),
        int((hexagonos_conteo_m["delitos_g1"] > 0).sum()),
        int((hexagonos_conteo_m["delitos_g2"] > 0).sum()),
        int((hexagonos_conteo_m["delitos_total_eval"] == 0).sum()),
        round((hexagonos_conteo_m["delitos_total_eval"] == 0).mean() * 100, 2),
        round(float(fila_seleccionada["score_seleccion_grilla"]), 4),
    ],
})
resumen_grilla.to_excel(salida_tablas / "resumen_grilla_seleccionada.xlsx", index=False)
hexagonos_conteo_m.to_crs(epsg=4326).to_file(
    salida_datasets / f"grilla_hexagonal_{TAMANO_HEXAGONO_SELECCIONADO_M}m_seleccionada.geojson",
    driver="GeoJSON",
)

print(f"Grilla seleccionada: {TAMANO_HEXAGONO_SELECCIONADO_M} m")
resumen_sensibilidad_grilla


# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, grupo_id in zip(axes, GRUPOS_EVALUACION):
    col = f"delitos_{grupo_id}"
    gdf_distrito_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#404040", linewidth=0.8)
    hexagonos_conteo_m.plot(
        ax=ax,
        column=col,
        cmap="YlOrRd",
        linewidth=0.15,
        edgecolor="#777777",
        legend=True,
        legend_kwds={"label": "Cantidad de delitos", "shrink": 0.65},
    )
    ax.set_title(
        f"Grilla seleccionada {TAMANO_HEXAGONO_SELECCIONADO_M} m - "
        f"{grupo_id}: {NOMBRES_GRUPOS[grupo_id]}"
    )
    ax.set_axis_off()
guardar_figura(
    f"fig_551_grilla_hexagonal_{TAMANO_HEXAGONO_SELECCIONADO_M}m_conteo_delitos_por_grupo.png"
)

fig, ax = plt.subplots(figsize=(9, 5))
plot_grilla = resumen_sensibilidad_grilla.copy()
ax.plot(
    plot_grilla["tamano_lado_m"],
    plot_grilla["score_seleccion_grilla"],
    marker="o",
    color="#4C78A8",
)
ax.scatter(
    [TAMANO_HEXAGONO_SELECCIONADO_M],
    [fila_seleccionada["score_seleccion_grilla"]],
    s=120,
    color="#E15759",
    zorder=3,
    label="Escala seleccionada",
)
ax.set_xlabel("Lado del hexágono (m)")
ax.set_ylabel("Puntaje de selección")
ax.set_title("Evaluación previa del tamaño de grilla")
ax.legend(frameon=False)
guardar_figura("fig_551b_seleccion_tamano_grilla_hexagonal.png")


# %% [markdown]
# ## 5.5.3. Asociación de KDE, DBSCAN y HDBSCAN a hexágonos
# %%
def coordenadas(gdf):
    return np.column_stack([gdf.geometry.x, gdf.geometry.y])


def obtener_bandwidth(grupo_id, default=200):
    if kde_parametros.empty or "grupo_id" not in kde_parametros.columns:
        return default
    fila = kde_parametros[kde_parametros["grupo_id"].astype(str) == grupo_id]
    if fila.empty or pd.isna(fila.iloc[0].get("bandwidth_m", np.nan)):
        return default
    return int(fila.iloc[0]["bandwidth_m"])


def obtener_dbscan_parametros(grupo_id, eps_default=75, min_samples_default=15):
    if dbscan_resumen_previo.empty or "grupo_id" not in dbscan_resumen_previo.columns:
        return eps_default, min_samples_default
    fila = dbscan_resumen_previo[dbscan_resumen_previo["grupo_id"].astype(str) == grupo_id]
    if fila.empty:
        return eps_default, min_samples_default
    return float(fila.iloc[0].get("eps_usado_m", eps_default)), int(
        fila.iloc[0].get("min_samples_usado", min_samples_default)
    )


def obtener_hdbscan_parametros(grupo_id, min_cluster_default=50, min_samples_default=20):
    if hdbscan_resumen_previo.empty or "grupo_id" not in hdbscan_resumen_previo.columns:
        return min_cluster_default, min_samples_default
    fila = hdbscan_resumen_previo[hdbscan_resumen_previo["grupo_id"].astype(str) == grupo_id]
    if fila.empty:
        return min_cluster_default, min_samples_default
    return int(fila.iloc[0].get("min_cluster_size_usado", min_cluster_default)), int(
        fila.iloc[0].get("min_samples_usado", min_samples_default)
    )


def asociar_tecnicas_hexagonos(hex_base, puntos_m, grupos=GRUPOS_EVALUACION):
    centroides = hex_base.geometry.centroid
    coords_centroides = np.column_stack([centroides.x, centroides.y])
    capas = []
    tiempos = []
    for grupo_id in grupos:
        puntos_grupo = puntos_m[puntos_m["grupo_id"].astype(str) == grupo_id].copy()
        grupo_nombre = puntos_grupo["grupo"].iloc[0] if not puntos_grupo.empty else NOMBRES_GRUPOS.get(grupo_id, grupo_id)
        if puntos_grupo.empty:
            continue
        coords_grupo = coordenadas(puntos_grupo)
        for tecnica in ["KDE", "DBSCAN", "HDBSCAN"]:
            inicio = time.perf_counter()
            capa = hex_base[["hex_id", "geometry"]].copy()
            capa["grupo_id"] = grupo_id
            capa["grupo"] = grupo_nombre
            capa["tecnica"] = tecnica
            capa["conteo_delitos"] = hex_base[f"delitos_{grupo_id}"].values
            if tecnica == "KDE":
                bandwidth = obtener_bandwidth(grupo_id)
                modelo = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
                modelo.fit(coords_grupo)
                densidad = np.exp(modelo.score_samples(coords_centroides))
                capa["valor_tecnica"] = densidad
                capa["parametros"] = f"bandwidth={bandwidth} m"
                capa["umbral_hotspot_tecnica"] = np.nanpercentile(densidad, 95) if np.isfinite(densidad).any() else np.nan
                capa["hotspot_tecnica"] = capa["valor_tecnica"] >= capa["umbral_hotspot_tecnica"]
            else:
                col_label = "cluster_dbscan" if tecnica == "DBSCAN" else "cluster_hdbscan"
                puntos_cluster = puntos_grupo[puntos_grupo[col_label].fillna(-1).astype(int) != -1].copy()
                if puntos_cluster.empty:
                    capa["valor_tecnica"] = 0
                else:
                    join = gpd.sjoin(
                        puntos_cluster[[col_label, "geometry"]],
                        hex_base[["hex_id", "geometry"]],
                        how="left",
                        predicate="within",
                    )
                    conteos_cluster = join.groupby("hex_id").size()
                    capa["valor_tecnica"] = capa["hex_id"].map(conteos_cluster).fillna(0).astype(int)
                capa["parametros"] = "puntos asignados a clusters; ruido=-1 excluido"
                capa["umbral_hotspot_tecnica"] = 0
                capa["hotspot_tecnica"] = capa["valor_tecnica"] > 0
            duracion = time.perf_counter() - inicio
            tiempos.append({
                "grupo_id": grupo_id,
                "grupo": grupo_nombre,
                "tecnica": tecnica,
                "tiempo_asociacion_hexagonos_s": duracion,
                "registros_grupo": len(puntos_grupo),
                "tiempo_por_registro_s": duracion / len(puntos_grupo) if len(puntos_grupo) else np.nan,
                "memoria_puntos_grupo_mb": puntos_grupo.memory_usage(deep=True).sum() / (1024 ** 2),
            })
            capas.append(capa)
    return pd.concat(capas, ignore_index=True), pd.DataFrame(tiempos)

hex_tecnicas_m, tiempos_asociacion = asociar_tecnicas_hexagonos(hexagonos_base_m, gdf_puntos_eval_m)
hex_tecnicas_m.to_crs(epsg=4326).to_file(
    salida_datasets / f"grilla_hexagonal_{TAMANO_HEXAGONO_SELECCIONADO_M}m_resultados_tecnicas.geojson",
    driver="GeoJSON",
)
tiempos_asociacion.head()


# %% [markdown]
# ### Hotspots comparables por cuota de área
# 
# %%
def evaluar_hotspots_por_cuota(hex_tecnicas, cuotas=CUOTAS_AREA_HOTSPOT):
    filas = []
    capas = []
    for (grupo_id, grupo, tecnica), capa in hex_tecnicas.groupby(["grupo_id", "grupo", "tecnica"]):
        capa = gpd.GeoDataFrame(capa.copy(), geometry="geometry", crs=hex_tecnicas.crs)
        capa["valor_tecnica_num"] = pd.to_numeric(capa["valor_tecnica"], errors="coerce").fillna(0)
        capa = capa.sort_values(["valor_tecnica_num", "hex_id"], ascending=[False, True]).reset_index(drop=True)
        delitos_total = float(capa["conteo_delitos"].sum())
        for cuota in cuotas:
            seleccion, cuota_real = seleccionar_por_cuota_area(capa, "valor_tecnica_num", cuota)
            n_hotspots = int(seleccion.sum())
            delitos_capturados = float(capa.loc[seleccion, "conteo_delitos"].sum())
            hit_rate = delitos_capturados / delitos_total if delitos_total else np.nan
            pai = hit_rate / cuota_real if cuota_real > 0 and np.isfinite(hit_rate) else np.nan
            filas.append({
                "grupo_id": grupo_id,
                "grupo": grupo,
                "tecnica": tecnica,
                "cuota_area": cuota,
                "cuota_area_pct": round(cuota * 100, 2),
                "cuota_area_real": cuota_real,
                "hexagonos_hotspot": n_hotspots,
                "delitos_total": delitos_total,
                "delitos_capturados": delitos_capturados,
                "hit_rate": hit_rate,
                "pai": pai,
            })
            capa_cuota = capa[["hex_id", "grupo_id", "grupo", "tecnica", "valor_tecnica", "conteo_delitos", "geometry"]].copy()
            capa_cuota["cuota_area"] = cuota
            capa_cuota["hotspot_cuota"] = seleccion
            capas.append(capa_cuota)
    tabla = pd.DataFrame(filas)
    capa_hotspots = pd.concat(capas, ignore_index=True) if capas else gpd.GeoDataFrame(columns=[])
    if not capa_hotspots.empty:
        capa_hotspots = gpd.GeoDataFrame(capa_hotspots, geometry="geometry", crs=hex_tecnicas.crs)
    return tabla, capa_hotspots

evaluacion_hotspots_cuotas, hotspots_cuotas_m = evaluar_hotspots_por_cuota(hex_tecnicas_m)
evaluacion_hotspots_cuotas.to_excel(salida_tablas / "evaluacion_hotspots_cuotas_area.xlsx", index=False)
evaluacion_hotspots_cuotas.to_csv(salida_tablas / "evaluacion_hotspots_cuotas_area.csv", index=False, encoding="utf-8-sig")
if not hotspots_cuotas_m.empty:
    hotspots_cuotas_m.to_crs(epsg=4326).to_file(salida_datasets / "hotspots_cuotas_area_tecnicas.geojson", driver="GeoJSON")


hotspots_operativos_resumidos = (
    evaluacion_hotspots_cuotas[
        np.isclose(evaluacion_hotspots_cuotas["cuota_area"].astype(float), 0.05)
    ][
        [
            "grupo_id",
            "tecnica",
            "cuota_area_real",
            "delitos_capturados",
            "hit_rate",
            "pai",
        ]
    ]
    .copy()
    .rename(
        columns={
            "grupo_id": "Grupo",
            "tecnica": "Técnica",
            "cuota_area_real": "Área real seleccionada (%)",
            "delitos_capturados": "Delitos capturados",
            "hit_rate": "Delitos capturados (%)",
            "pai": "PAI",
        }
    )
)
hotspots_operativos_resumidos["Área real seleccionada (%)"] = (
    hotspots_operativos_resumidos["Área real seleccionada (%)"] * 100
).round(2)
hotspots_operativos_resumidos["Delitos capturados (%)"] = (
    hotspots_operativos_resumidos["Delitos capturados (%)"] * 100
).round(2)
hotspots_operativos_resumidos["PAI"] = hotspots_operativos_resumidos["PAI"].round(2)
hotspots_operativos_resumidos.to_excel(
    salida_tablas / "hotspots_operativos_resumidos_para_tesis.xlsx",
    index=False,
)
hotspots_operativos_resumidos.to_csv(
    salida_tablas / "hotspots_operativos_resumidos_para_tesis.csv",
    index=False,
    encoding="utf-8-sig",
)
hotspots_operativos_resumidos


# %% [markdown]
# ### Validación predictiva temporal
# 
# %%
def construir_valores_tecnica_entrenamiento(hex_base, puntos_train, grupo_id, tecnica):
    capa = hex_base[["hex_id", "geometry"]].copy()
    puntos_grupo = puntos_train[puntos_train["grupo_id"].astype(str) == grupo_id].copy()
    if puntos_grupo.empty:
        capa["valor_tecnica"] = 0.0
        return capa
    if tecnica == "KDE":
        centroides = hex_base.geometry.centroid
        coords_centroides = np.column_stack([centroides.x, centroides.y])
        modelo = KernelDensity(kernel="gaussian", bandwidth=obtener_bandwidth(grupo_id))
        modelo.fit(coordenadas(puntos_grupo))
        capa["valor_tecnica"] = np.exp(modelo.score_samples(coords_centroides))
        return capa

    coords = coordenadas(puntos_grupo)
    if tecnica == "DBSCAN":
        eps, min_samples = obtener_dbscan_parametros(grupo_id)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    else:
        if not HDBSCAN_LIB_DISPONIBLE:
            raise ImportError(f"Falta hdbscan para validación temporal: {HDBSCAN_LIB_ERROR}")
        min_cluster_size, min_samples = obtener_hdbscan_parametros(grupo_id)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        ).fit_predict(coords)
    puntos_cluster = puntos_grupo.loc[labels != -1].copy()
    if puntos_cluster.empty:
        capa["valor_tecnica"] = 0.0
        return capa
    join = gpd.sjoin(
        puntos_cluster[["geometry"]],
        hex_base[["hex_id", "geometry"]],
        how="left",
        predicate="within",
    )
    conteos_cluster = join.groupby("hex_id").size()
    capa["valor_tecnica"] = capa["hex_id"].map(conteos_cluster).fillna(0).astype(float)
    return capa

def validar_temporalmente_hotspots(hex_base, puntos_m, cuotas=CUOTAS_AREA_HOTSPOT):
    if "año_hecho" not in puntos_m.columns:
        return pd.DataFrame(), gpd.GeoDataFrame(columns=[])
    puntos = puntos_m.copy()
    puntos["año_hecho"] = pd.to_numeric(puntos["año_hecho"], errors="coerce")
    puntos_train = puntos[puntos["año_hecho"] <= ANIO_CORTE_ENTRENAMIENTO].copy()
    puntos_test = puntos[puntos["año_hecho"] == ANIO_VALIDACION_TEMPORAL].copy()
    if puntos_train.empty or puntos_test.empty:
        return pd.DataFrame(), gpd.GeoDataFrame(columns=[])
    filas = []
    capas = []
    for grupo_id in GRUPOS_EVALUACION:
        test_grupo = puntos_test[puntos_test["grupo_id"].astype(str) == grupo_id].copy()
        train_grupo = puntos_train[puntos_train["grupo_id"].astype(str) == grupo_id].copy()
        if test_grupo.empty or train_grupo.empty:
            continue
        grupo_nombre = test_grupo["grupo"].iloc[0]
        join_test = gpd.sjoin(
            test_grupo[["geometry"]],
            hex_base[["hex_id", "geometry"]],
            how="left",
            predicate="within",
        )
        conteos_test = join_test.groupby("hex_id").size()
        delitos_test_total = float(len(test_grupo))
        for tecnica in ["KDE", "DBSCAN", "HDBSCAN"]:
            capa_valores = construir_valores_tecnica_entrenamiento(hex_base, puntos_train, grupo_id, tecnica)
            capa_valores["conteo_test"] = capa_valores["hex_id"].map(conteos_test).fillna(0).astype(int)
            capa_valores = capa_valores.sort_values(["valor_tecnica", "hex_id"], ascending=[False, True]).reset_index(drop=True)
            for cuota in cuotas:
                seleccion, cuota_real = seleccionar_por_cuota_area(
                    capa_valores,
                    "valor_tecnica",
                    cuota,
                )
                delitos_capturados = float(capa_valores.loc[seleccion, "conteo_test"].sum())
                hit_rate = delitos_capturados / delitos_test_total if delitos_test_total else np.nan
                pai = hit_rate / cuota_real if cuota_real > 0 and np.isfinite(hit_rate) else np.nan
                filas.append({
                    "grupo_id": grupo_id,
                    "grupo": grupo_nombre,
                    "tecnica": tecnica,
                    "anio_entrenamiento_hasta": ANIO_CORTE_ENTRENAMIENTO,
                    "anio_validacion": ANIO_VALIDACION_TEMPORAL,
                    "cuota_area": cuota,
                    "cuota_area_pct": round(cuota * 100, 2),
                    "cuota_area_real": cuota_real,
                    "delitos_test_total": delitos_test_total,
                    "delitos_test_capturados": delitos_capturados,
                    "hit_rate_2025": hit_rate,
                    "pai_2025": pai,
                    "nota": "Modelos reentrenados con datos <= 2024 y evaluados contra 2025 sin usar el test para desempatar",
                })
                capa_tmp = capa_valores[["hex_id", "valor_tecnica", "conteo_test", "geometry"]].copy()
                capa_tmp["grupo_id"] = grupo_id
                capa_tmp["grupo"] = grupo_nombre
                capa_tmp["tecnica"] = tecnica
                capa_tmp["cuota_area"] = cuota
                capa_tmp["hotspot_temporal"] = seleccion
                capas.append(capa_tmp)
    tabla = pd.DataFrame(filas)
    capa_hotspots = pd.concat(capas, ignore_index=True) if capas else gpd.GeoDataFrame(columns=[])
    if not capa_hotspots.empty:
        capa_hotspots = gpd.GeoDataFrame(capa_hotspots, geometry="geometry", crs=hex_base.crs)
    return tabla, capa_hotspots

validacion_temporal, hotspots_temporales_m = validar_temporalmente_hotspots(hexagonos_base_m, gdf_puntos_eval_m)
validacion_temporal.to_excel(salida_tablas / "validacion_predictiva_temporal_2025.xlsx", index=False)
validacion_temporal.to_csv(salida_tablas / "validacion_predictiva_temporal_2025.csv", index=False, encoding="utf-8-sig")
if not hotspots_temporales_m.empty:
    hotspots_temporales_m.to_crs(epsg=4326).to_file(salida_datasets / "hotspots_validacion_temporal_2025.geojson", driver="GeoJSON")


validacion_temporal_resumida = (
    validacion_temporal[
        np.isclose(validacion_temporal["cuota_area"].astype(float), 0.05)
    ][
        [
            "grupo_id",
            "tecnica",
            "cuota_area_real",
            "delitos_test_total",
            "hit_rate_2025",
            "pai_2025",
        ]
    ]
    .copy()
    .rename(
        columns={
            "grupo_id": "Grupo",
            "tecnica": "Técnica",
            "cuota_area_real": "Área real seleccionada (%)",
            "delitos_test_total": "Delitos evaluados en 2025",
            "hit_rate_2025": "Delitos 2025 capturados (%)",
            "pai_2025": "PAI 2025",
        }
    )
)
validacion_temporal_resumida["Área real seleccionada (%)"] = (
    validacion_temporal_resumida["Área real seleccionada (%)"] * 100
).round(2)
validacion_temporal_resumida["Delitos 2025 capturados (%)"] = (
    validacion_temporal_resumida["Delitos 2025 capturados (%)"] * 100
).round(2)
validacion_temporal_resumida["PAI 2025"] = validacion_temporal_resumida["PAI 2025"].round(2)
validacion_temporal_resumida.to_excel(
    salida_tablas / "validacion_temporal_resumida_para_tesis.xlsx",
    index=False,
)
validacion_temporal_resumida.to_csv(
    salida_tablas / "validacion_temporal_resumida_para_tesis.csv",
    index=False,
    encoding="utf-8-sig",
)
validacion_temporal_resumida


# %% [markdown]
# ## 5.5.4. Validación espacial: Moran Global, LISA y Getis-Ord Gi\*
# %%
def benjamini_hochberg(pvalues, alpha=0.05):
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p)
    rejected = np.zeros(len(p), dtype=bool)
    q = np.full(len(p), np.nan, dtype=float)
    if valid.sum() == 0:
        return rejected, q
    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    m = len(ranked)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresholds
    if passed.any():
        cutoff = ranked[np.where(passed)[0].max()]
        rejected[valid] = p_valid <= cutoff
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_valid = np.empty_like(q_ranked)
    q_valid[order] = np.minimum(q_ranked, 1)
    q[valid] = q_valid
    return rejected, q


def _pvalor(obj):
    for nombre in ["p_sim", "p_norm", "p_z_sim"]:
        if hasattr(obj, nombre):
            return getattr(obj, nombre)
    return np.nan


def _zvalor_gi(obj):
    for nombre in ["Zs", "z_sim", "z_norm"]:
        if hasattr(obj, nombre):
            return getattr(obj, nombre)
    return np.full(len(obj.p_sim), np.nan) if hasattr(obj, "p_sim") else np.array([])


def clasificar_lisa(moran_local, alpha=0.05):
    mapa = {1: "Alto-Alto", 2: "Bajo-Alto", 3: "Bajo-Bajo", 4: "Alto-Bajo"}
    categorias = []
    for q, p in zip(moran_local.q, moran_local.p_sim):
        categorias.append(mapa.get(int(q), "No significativo") if p < alpha else "No significativo")
    return categorias


def calcular_validacion_espacial(hex_tecnicas, alpha=ALPHA_SIGNIFICANCIA):
    if not PYSAL_DISPONIBLE:
        raise ImportError(f"Faltan librerías PySAL: {PYSAL_ERROR}")
    resultados = []
    capas_lisa = []
    capas_gi = []
    for (grupo_id, grupo, tecnica), capa in hex_tecnicas.groupby(["grupo_id", "grupo", "tecnica"]):
        capa = gpd.GeoDataFrame(capa.copy(), geometry="geometry", crs=hex_tecnicas.crs).reset_index(drop=True)
        valores = pd.to_numeric(capa["valor_tecnica"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if len(capa) < 3 or np.nanstd(valores) == 0:
            resultados.append({
                "grupo_id": grupo_id, "grupo": grupo, "tecnica": tecnica,
                "moran_i": np.nan, "moran_p": np.nan,
                "lisa_alto_alto_sig": 0, "lisa_bajo_bajo_sig": 0,
                "gi_hotspots_sig": 0, "gi_hotspots_sig_pct": 0,
                "gi_p_min": np.nan,
                "nota": "Sin variabilidad suficiente para autocorrelación espacial",
            })
            continue
        w = Queen.from_dataframe(capa, ids=capa.index.tolist(), silence_warnings=True)
        w.transform = "r"
        moran = Moran(valores, w, permutations=999)
        lisa = Moran_Local(valores, w, permutations=999, n_jobs=1, seed=42)
        gi = G_Local(valores, w, star=True, permutations=999, n_jobs=1, seed=42)
        lisa_cat = clasificar_lisa(lisa, alpha=alpha)
        lisa_p = np.asarray(lisa.p_sim, dtype=float)
        lisa_fdr_sig, lisa_q = benjamini_hochberg(lisa_p, alpha=alpha)
        lisa_cat_fdr = [cat if sig else "No significativo" for cat, sig in zip(lisa_cat, lisa_fdr_sig)]
        gi_z = _zvalor_gi(gi)
        gi_p = np.asarray(_pvalor(gi), dtype=float)
        gi_fdr_sig, gi_q = benjamini_hochberg(gi_p, alpha=alpha)
        gi_hotspot = (gi_p < alpha) & (gi_z > 0)
        gi_hotspot_fdr = gi_fdr_sig & (gi_z > 0)
        capa_lisa = capa[["hex_id", "grupo_id", "grupo", "tecnica", "valor_tecnica", "geometry"]].copy()
        capa_lisa["lisa_categoria"] = lisa_cat
        capa_lisa["lisa_categoria_fdr"] = lisa_cat_fdr
        capa_lisa["lisa_p"] = lisa_p
        capa_lisa["lisa_q_fdr"] = lisa_q
        capa_lisa["lisa_i"] = lisa.Is
        capa_gi = capa[["hex_id", "grupo_id", "grupo", "tecnica", "valor_tecnica", "geometry"]].copy()
        capa_gi["gi_z"] = gi_z
        capa_gi["gi_p"] = gi_p
        capa_gi["gi_q_fdr"] = gi_q
        capa_gi["gi_hotspot_sig"] = gi_hotspot
        capa_gi["gi_hotspot_fdr_sig"] = gi_hotspot_fdr
        capas_lisa.append(capa_lisa)
        capas_gi.append(capa_gi)
        resultados.append({
            "grupo_id": grupo_id,
            "grupo": grupo,
            "tecnica": tecnica,
            "moran_i": moran.I,
            "moran_p": moran.p_sim,
            "lisa_alto_alto_sig": int((np.array(lisa_cat) == "Alto-Alto").sum()),
            "lisa_bajo_bajo_sig": int((np.array(lisa_cat) == "Bajo-Bajo").sum()),
            "lisa_alto_alto_fdr_sig": int((np.array(lisa_cat_fdr) == "Alto-Alto").sum()),
            "lisa_bajo_bajo_fdr_sig": int((np.array(lisa_cat_fdr) == "Bajo-Bajo").sum()),
            "gi_hotspots_sig": int(gi_hotspot.sum()),
            "gi_hotspots_sig_pct": round(gi_hotspot.mean() * 100, 2),
            "gi_hotspots_fdr_sig": int(gi_hotspot_fdr.sum()),
            "gi_hotspots_fdr_sig_pct": round(gi_hotspot_fdr.mean() * 100, 2),
            "gi_p_min": float(np.nanmin(gi_p)) if len(gi_p) else np.nan,
            "gi_q_fdr_min": float(np.nanmin(gi_q)) if len(gi_q) else np.nan,
            "tamano_grilla_m": TAMANO_HEXAGONO_SELECCIONADO_M,
            "nota": "Calculado con pesos Queen sobre la grilla seleccionada",
        })
    lisa_gdf = pd.concat(capas_lisa, ignore_index=True) if capas_lisa else gpd.GeoDataFrame(columns=[])
    gi_gdf = pd.concat(capas_gi, ignore_index=True) if capas_gi else gpd.GeoDataFrame(columns=[])
    if not lisa_gdf.empty:
        lisa_gdf = gpd.GeoDataFrame(lisa_gdf, geometry="geometry", crs=hex_tecnicas.crs)
    if not gi_gdf.empty:
        gi_gdf = gpd.GeoDataFrame(gi_gdf, geometry="geometry", crs=hex_tecnicas.crs)
    return pd.DataFrame(resultados), lisa_gdf, gi_gdf

validacion_espacial, lisa_hex_m, gi_hex_m = calcular_validacion_espacial(hex_tecnicas_m)
validacion_espacial.to_excel(salida_tablas / "validacion_espacial_tecnicas_por_grupo.xlsx", index=False)
validacion_espacial.to_csv(salida_tablas / "validacion_espacial_tecnicas_por_grupo.csv", index=False, encoding="utf-8-sig")

validacion_espacial_resumida = (
    validacion_espacial[
        [
            "grupo_id",
            "tecnica",
            "moran_i",
            "moran_p",
            "lisa_alto_alto_fdr_sig",
            "gi_hotspots_fdr_sig_pct",
        ]
    ]
    .copy()
    .rename(
        columns={
            "grupo_id": "Grupo",
            "tecnica": "Técnica",
            "moran_i": "Moran I",
            "moran_p": "p-valor Moran",
            "lisa_alto_alto_fdr_sig": "Celdas Alto-Alto LISA (FDR)",
            "gi_hotspots_fdr_sig_pct": "Hotspots Gi* significativos (%)",
        }
    )
)
validacion_espacial_resumida["Moran I"] = validacion_espacial_resumida["Moran I"].round(4)
validacion_espacial_resumida["p-valor Moran"] = validacion_espacial_resumida["p-valor Moran"].round(4)
validacion_espacial_resumida["Hotspots Gi* significativos (%)"] = (
    validacion_espacial_resumida["Hotspots Gi* significativos (%)"].round(2)
)
validacion_espacial_resumida.to_excel(
    salida_tablas / "validacion_espacial_resumida_para_tesis.xlsx",
    index=False,
)
validacion_espacial_resumida.to_csv(
    salida_tablas / "validacion_espacial_resumida_para_tesis.csv",
    index=False,
    encoding="utf-8-sig",
)
if not lisa_hex_m.empty:
    lisa_hex_m.to_crs(epsg=4326).to_file(salida_datasets / f"lisa_hexagonal_{TAMANO_HEXAGONO_SELECCIONADO_M}m_tecnicas.geojson", driver="GeoJSON")
if not gi_hex_m.empty:
    gi_hex_m.to_crs(epsg=4326).to_file(salida_datasets / f"getis_ord_gi_hexagonal_{TAMANO_HEXAGONO_SELECCIONADO_M}m_tecnicas.geojson", driver="GeoJSON")
validacion_espacial_resumida


# %%
if not gi_hex_m.empty:
    fig, axes = plt.subplots(len(GRUPOS_EVALUACION), 3, figsize=(15, 10))
    for i, grupo_id in enumerate(GRUPOS_EVALUACION):
        for j, tecnica in enumerate(["KDE", "DBSCAN", "HDBSCAN"]):
            ax = axes[i, j]
            capa = gi_hex_m[(gi_hex_m["grupo_id"] == grupo_id) & (gi_hex_m["tecnica"] == tecnica)].copy()
            gdf_distrito_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#404040", linewidth=0.6)
            capa.plot(ax=ax, color="#EFEFEF", edgecolor="#FFFFFF", linewidth=0.05)
            capa[capa["gi_hotspot_sig"]].plot(ax=ax, column="gi_z", cmap="Reds", edgecolor="#7F0000", linewidth=0.12, legend=False)
            ax.set_title(f"Gi* significativo - {grupo_id} - {tecnica} - {TAMANO_HEXAGONO_SELECCIONADO_M} m")
            ax.set_axis_off()
    guardar_figura("fig_552_hotspots_getis_ord_gi_tecnicas_por_grupo.png")


# %%
if not lisa_hex_m.empty:
    fig, axes = plt.subplots(len(GRUPOS_EVALUACION), 3, figsize=(15, 10))
    for i, grupo_id in enumerate(GRUPOS_EVALUACION):
        for j, tecnica in enumerate(["KDE", "DBSCAN", "HDBSCAN"]):
            ax = axes[i, j]
            capa = lisa_hex_m[(lisa_hex_m["grupo_id"] == grupo_id) & (lisa_hex_m["tecnica"] == tecnica)].copy()
            gdf_distrito_m.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#404040", linewidth=0.6)
            for categoria, color in COLORES_LISA.items():
                sub = capa[capa["lisa_categoria"] == categoria]
                if not sub.empty:
                    sub.plot(ax=ax, color=color, edgecolor="#FFFFFF", linewidth=0.04, label=categoria)
            ax.set_title(f"LISA - {grupo_id} - {tecnica} - {TAMANO_HEXAGONO_SELECCIONADO_M} m")
            ax.set_axis_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    guardar_figura("fig_553_lisa_tecnicas_por_grupo.png")


# %% [markdown]
# ## 5.5.5. Evaluación de clustering
# 
# %%
def metricas_clustering_para_labels(puntos_grupo, columna_label, tecnica):
    inicio = time.perf_counter()
    labels = puntos_grupo[columna_label].fillna(-1).astype(int).to_numpy()
    coords = coordenadas(puntos_grupo)
    mascara_cluster = labels != -1
    labels_validos = labels[mascara_cluster]
    coords_validos = coords[mascara_cluster]
    clusters = sorted(set(labels_validos.tolist()))
    n_clusters = len(clusters)
    ruido = int((labels == -1).sum())
    pct_ruido = ruido / len(labels) * 100 if len(labels) else np.nan
    tamanos = pd.Series(labels_validos).value_counts() if len(labels_validos) else pd.Series(dtype=int)
    puede_calcular = n_clusters >= 2 and len(coords_validos) > n_clusters
    if puede_calcular:
        silhouette = silhouette_score(coords_validos, labels_validos)
        davies = davies_bouldin_score(coords_validos, labels_validos)
        calinski = calinski_harabasz_score(coords_validos, labels_validos)
    else:
        silhouette = np.nan
        davies = np.nan
        calinski = np.nan
    duracion = time.perf_counter() - inicio
    return {
        "tecnica": tecnica,
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies,
        "calinski_harabasz_index": calinski,
        "numero_clusters": n_clusters,
        "porcentaje_ruido": pct_ruido,
        "tamano_cluster_mas_grande": int(tamanos.max()) if not tamanos.empty else 0,
        "tamano_promedio_cluster": float(tamanos.mean()) if not tamanos.empty else np.nan,
        "tiempo_metricas_clustering_s": duracion,
        "nota": "Calculado sin puntos de ruido" if puede_calcular else "No aplicable por cantidad insuficiente de clusters válidos",
    }

filas_clustering = []
for grupo_id in GRUPOS_EVALUACION:
    puntos_grupo = gdf_puntos_eval_m[gdf_puntos_eval_m["grupo_id"].astype(str) == grupo_id].copy()
    grupo_nombre = puntos_grupo["grupo"].iloc[0] if not puntos_grupo.empty else NOMBRES_GRUPOS[grupo_id]
    filas_clustering.append({
        "grupo_id": grupo_id,
        "grupo": grupo_nombre,
        "tecnica": "KDE",
        "silhouette_score": "No aplica",
        "davies_bouldin_index": "No aplica",
        "calinski_harabasz_index": "No aplica",
        "numero_clusters": "No aplica",
        "porcentaje_ruido": "No aplica",
        "tamano_cluster_mas_grande": "No aplica",
        "tamano_promedio_cluster": "No aplica",
        "tiempo_metricas_clustering_s": np.nan,
        "nota": "KDE no asigna etiquetas de cluster",
    })
    for tecnica, columna in [("DBSCAN", "cluster_dbscan"), ("HDBSCAN", "cluster_hdbscan")]:
        fila = metricas_clustering_para_labels(puntos_grupo, columna, tecnica)
        fila.update({"grupo_id": grupo_id, "grupo": grupo_nombre})
        filas_clustering.append(fila)

metricas_clustering = pd.DataFrame(filas_clustering)
metricas_clustering.to_excel(salida_tablas / "metricas_clustering_tecnicas_por_grupo.xlsx", index=False)
metricas_clustering.to_csv(salida_tablas / "metricas_clustering_tecnicas_por_grupo.csv", index=False, encoding="utf-8-sig")


clustering_resumido = (
    metricas_clustering[
        metricas_clustering["tecnica"].isin(["DBSCAN", "HDBSCAN"])
    ][
        [
            "grupo_id",
            "tecnica",
            "silhouette_score",
            "davies_bouldin_index",
            "numero_clusters",
            "porcentaje_ruido",
        ]
    ]
    .copy()
    .rename(
        columns={
            "grupo_id": "Grupo",
            "tecnica": "Técnica",
            "silhouette_score": "Silhouette",
            "davies_bouldin_index": "Davies-Bouldin",
            "numero_clusters": "Número de clusters",
            "porcentaje_ruido": "Ruido (%)",
        }
    )
)
clustering_resumido["Silhouette"] = pd.to_numeric(
    clustering_resumido["Silhouette"], errors="coerce"
).round(4)
clustering_resumido["Davies-Bouldin"] = pd.to_numeric(
    clustering_resumido["Davies-Bouldin"], errors="coerce"
).round(4)
clustering_resumido["Ruido (%)"] = pd.to_numeric(
    clustering_resumido["Ruido (%)"], errors="coerce"
).round(2)
clustering_resumido.to_excel(
    salida_tablas / "clustering_resumido_para_tesis.xlsx",
    index=False,
)
clustering_resumido.to_csv(
    salida_tablas / "clustering_resumido_para_tesis.csv",
    index=False,
    encoding="utf-8-sig",
)
clustering_resumido


# %% [markdown]
# ## 5.5.6. Evaluación computacional
# 
# %%
tiempos_clustering = metricas_clustering[["grupo_id", "grupo", "tecnica", "tiempo_metricas_clustering_s"]].copy()
desempeno_computacional = tiempos_asociacion.merge(
    tiempos_clustering,
    on=["grupo_id", "grupo", "tecnica"],
    how="left",
)
desempeno_computacional["tiempo_total_validacion_s"] = desempeno_computacional[
    ["tiempo_asociacion_hexagonos_s", "tiempo_metricas_clustering_s"]
].fillna(0).sum(axis=1)
desempeno_computacional["tiempo_total_por_registro_s"] = desempeno_computacional["tiempo_total_validacion_s"] / desempeno_computacional["registros_grupo"]

desempeno_computacional.to_excel(salida_tablas / "desempeno_computacional_tecnicas_por_grupo.xlsx", index=False)
desempeno_computacional.to_csv(salida_tablas / "desempeno_computacional_tecnicas_por_grupo.csv", index=False, encoding="utf-8-sig")


desempeno_computacional_resumido = (
    desempeno_computacional[
        [
            "grupo_id",
            "tecnica",
            "registros_grupo",
            "tiempo_total_validacion_s",
            "tiempo_total_por_registro_s",
            "memoria_puntos_grupo_mb",
        ]
    ]
    .copy()
    .rename(
        columns={
            "grupo_id": "Grupo",
            "tecnica": "Técnica",
            "registros_grupo": "Registros",
            "tiempo_total_validacion_s": "Tiempo total (s)",
            "tiempo_total_por_registro_s": "Tiempo por registro (ms)",
            "memoria_puntos_grupo_mb": "Memoria del grupo (MB)",
        }
    )
)
desempeno_computacional_resumido["Tiempo total (s)"] = (
    desempeno_computacional_resumido["Tiempo total (s)"].round(4)
)
desempeno_computacional_resumido["Tiempo por registro (ms)"] = (
    desempeno_computacional_resumido["Tiempo por registro (ms)"] * 1000
).round(4)
desempeno_computacional_resumido["Memoria del grupo (MB)"] = (
    desempeno_computacional_resumido["Memoria del grupo (MB)"].round(3)
)
desempeno_computacional_resumido.to_excel(
    salida_tablas / "desempeno_computacional_resumido_para_tesis.xlsx",
    index=False,
)
desempeno_computacional_resumido.to_csv(
    salida_tablas / "desempeno_computacional_resumido_para_tesis.csv",
    index=False,
    encoding="utf-8-sig",
)
desempeno_computacional_resumido


# %% [markdown]
# ## 5.5.7. Ranking comparativo final
# 
# %%
def normalizar_serie(s, mayor_es_mejor=True):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    minimo, maximo = s.min(), s.max()
    if np.isclose(maximo, minimo):
        base = pd.Series(0.5, index=s.index)
    else:
        base = (s - minimo) / (maximo - minimo)
    return base if mayor_es_mejor else 1 - base


def extraer_cuota(tabla, cuota=0.05, prefijo=""):
    if tabla is None or tabla.empty:
        return pd.DataFrame(columns=["grupo_id", "grupo", "tecnica"])
    sub = tabla[np.isclose(tabla["cuota_area"].astype(float), cuota)].copy()
    cols_id = ["grupo_id", "grupo", "tecnica"]
    cols_metricas = [c for c in sub.columns if c not in cols_id + ["cuota_area", "cuota_area_pct", "nota"]]
    sub = sub[cols_id + cols_metricas]
    if prefijo:
        sub = sub.rename(columns={c: f"{prefijo}{c}" for c in cols_metricas})
    return sub

cuota_principal = 0.05
hotspots_5pct = extraer_cuota(evaluacion_hotspots_cuotas, cuota_principal, "cuota_")
temporal_5pct = extraer_cuota(validacion_temporal, cuota_principal, "temporal_")

ranking_base = validacion_espacial.merge(
    metricas_clustering,
    on=["grupo_id", "grupo", "tecnica"],
    how="left",
).merge(
    desempeno_computacional[["grupo_id", "grupo", "tecnica", "tiempo_total_validacion_s", "memoria_puntos_grupo_mb"]],
    on=["grupo_id", "grupo", "tecnica"],
    how="left",
).merge(
    hotspots_5pct,
    on=["grupo_id", "grupo", "tecnica"],
    how="left",
).merge(
    temporal_5pct,
    on=["grupo_id", "grupo", "tecnica"],
    how="left",
)

ranking_base["silhouette_num"] = pd.to_numeric(ranking_base["silhouette_score"], errors="coerce")
ranking_base["davies_num"] = pd.to_numeric(ranking_base["davies_bouldin_index"], errors="coerce")
ranking_base["calinski_num"] = pd.to_numeric(ranking_base["calinski_harabasz_index"], errors="coerce")
ranking_base["ruido_num"] = pd.to_numeric(ranking_base["porcentaje_ruido"], errors="coerce")

filas_final = []
filas_hotspots = []
filas_clustering_rank = []
for grupo_id, sub in ranking_base.groupby("grupo_id"):
    sub = sub.copy()
    sub["score_moran"] = normalizar_serie(sub["moran_i"], True)
    sub["score_gi_fdr"] = normalizar_serie(sub.get("gi_hotspots_fdr_sig_pct", sub["gi_hotspots_sig_pct"]), True)
    sub["score_lisa_fdr"] = normalizar_serie(sub.get("lisa_alto_alto_fdr_sig", sub["lisa_alto_alto_sig"]), True)
    sub["score_espacial"] = sub[["score_moran", "score_gi_fdr", "score_lisa_fdr"]].mean(axis=1, skipna=True)
    sub["score_hit_rate"] = normalizar_serie(sub.get("cuota_hit_rate", np.nan), True)
    sub["score_pai"] = normalizar_serie(sub.get("cuota_pai", np.nan), True)
    sub["score_hotspot_operativo"] = sub[["score_hit_rate", "score_pai"]].mean(axis=1, skipna=True)
    sub["score_temporal_hit"] = normalizar_serie(sub.get("temporal_hit_rate_2025", np.nan), True)
    sub["score_temporal_pai"] = normalizar_serie(sub.get("temporal_pai_2025", np.nan), True)
    sub["score_temporal"] = sub[["score_temporal_hit", "score_temporal_pai"]].mean(axis=1, skipna=True)
    sub["score_silhouette"] = normalizar_serie(sub["silhouette_num"], True)
    sub["score_davies"] = normalizar_serie(sub["davies_num"], False)
    sub["score_calinski"] = normalizar_serie(sub["calinski_num"], True)
    sub["score_clustering"] = sub[["score_silhouette", "score_davies", "score_calinski"]].mean(axis=1, skipna=True)
    sub.loc[sub["tecnica"] == "KDE", "score_clustering"] = np.nan
    sub["score_ruido"] = normalizar_serie(sub["ruido_num"], False)
    sub.loc[sub["tecnica"] == "KDE", "score_ruido"] = np.nan
    sub["score_computacional"] = normalizar_serie(sub["tiempo_total_validacion_s"], False)
    sub["score_interpretabilidad"] = sub["tecnica"].map({"KDE": 0.75, "DBSCAN": 0.90, "HDBSCAN": 0.85})

    sub["score_hotspots_predictivo"] = (
        sub["score_temporal"].fillna(0.50) * 0.55 +
        sub["score_hotspot_operativo"].fillna(0.50) * 0.25 +
        sub["score_espacial"].fillna(0.50) * 0.20
    )
    sub["ranking_hotspots"] = sub["score_hotspots_predictivo"].rank(ascending=False, method="dense").astype(int)

    sub["score_total"] = (
        sub["score_temporal"].fillna(0.50) * 0.40 +
        sub["score_espacial"].fillna(0.50) * 0.25 +
        sub["score_hotspot_operativo"].fillna(0.50) * 0.15 +
        sub["score_clustering"].fillna(0.50) * 0.10 +
        sub["score_ruido"].fillna(0.50) * 0.05 +
        sub["score_interpretabilidad"].fillna(0.50) * 0.05
    )
    sub["ranking_grupo"] = sub["score_total"].rank(ascending=False, method="dense").astype(int)

    rank_cluster = sub[sub["tecnica"].isin(["DBSCAN", "HDBSCAN"])].copy()
    rank_cluster["score_clustering_total"] = (
        rank_cluster["score_clustering"].fillna(0.50) * 0.70 +
        rank_cluster["score_ruido"].fillna(0.50) * 0.20 +
        rank_cluster["score_interpretabilidad"].fillna(0.50) * 0.10
    )
    rank_cluster["ranking_clustering"] = rank_cluster["score_clustering_total"].rank(ascending=False, method="dense").astype(int)

    filas_final.append(sub)
    filas_hotspots.append(sub.copy())
    filas_clustering_rank.append(rank_cluster)

ranking_final = pd.concat(filas_final, ignore_index=True).sort_values(["grupo_id", "ranking_grupo", "tecnica"])
ranking_hotspots_predictivo = pd.concat(filas_hotspots, ignore_index=True).sort_values(["grupo_id", "ranking_hotspots", "tecnica"])
ranking_clustering = pd.concat(filas_clustering_rank, ignore_index=True).sort_values(["grupo_id", "ranking_clustering", "tecnica"])

columnas_ranking = [
    "grupo_id", "grupo", "tecnica", "ranking_grupo", "score_total",
    "score_temporal", "score_espacial", "score_hotspot_operativo", "score_clustering",
    "score_ruido", "score_interpretabilidad",
    "moran_i", "moran_p", "gi_hotspots_fdr_sig", "gi_hotspots_fdr_sig_pct", "lisa_alto_alto_fdr_sig",
    "cuota_hit_rate", "cuota_pai", "temporal_hit_rate_2025", "temporal_pai_2025",
    "silhouette_score", "davies_bouldin_index", "calinski_harabasz_index", "porcentaje_ruido",
    "tiempo_total_validacion_s",
]
columnas_ranking = [c for c in columnas_ranking if c in ranking_final.columns]
ranking_final[columnas_ranking].to_excel(salida_tablas / "ranking_final_tecnicas_geoespaciales.xlsx", index=False)
ranking_final[columnas_ranking].to_csv(salida_tablas / "ranking_final_tecnicas_geoespaciales.csv", index=False, encoding="utf-8-sig")

columnas_hotspots = [
    "grupo_id", "grupo", "tecnica", "ranking_hotspots", "score_hotspots_predictivo",
    "score_temporal", "score_hotspot_operativo", "score_espacial",
    "cuota_hit_rate", "cuota_pai", "temporal_hit_rate_2025", "temporal_pai_2025",
    "gi_hotspots_fdr_sig_pct", "lisa_alto_alto_fdr_sig", "moran_i",
]
columnas_hotspots = [c for c in columnas_hotspots if c in ranking_hotspots_predictivo.columns]
ranking_hotspots_predictivo[columnas_hotspots].to_excel(salida_tablas / "ranking_hotspots_predictivo_tecnicas.xlsx", index=False)
ranking_hotspots_predictivo[columnas_hotspots].to_csv(salida_tablas / "ranking_hotspots_predictivo_tecnicas.csv", index=False, encoding="utf-8-sig")

columnas_clustering = [
    "grupo_id", "grupo", "tecnica", "ranking_clustering", "score_clustering_total",
    "score_clustering", "score_ruido", "silhouette_score", "davies_bouldin_index",
    "calinski_harabasz_index", "numero_clusters", "porcentaje_ruido",
]
columnas_clustering = [c for c in columnas_clustering if c in ranking_clustering.columns]
ranking_clustering[columnas_clustering].to_excel(salida_tablas / "ranking_clustering_dbscan_hdbscan.xlsx", index=False)
ranking_clustering[columnas_clustering].to_csv(salida_tablas / "ranking_clustering_dbscan_hdbscan.csv", index=False, encoding="utf-8-sig")


ranking_final_resumido = (
    ranking_final[
        [
            "grupo_id",
            "tecnica",
            "ranking_grupo",
            "score_total",
            "score_temporal",
            "score_espacial",
        ]
    ]
    .copy()
    .rename(
        columns={
            "grupo_id": "Grupo",
            "tecnica": "Técnica",
            "ranking_grupo": "Posición",
            "score_total": "Puntaje total",
            "score_temporal": "Puntaje temporal",
            "score_espacial": "Puntaje espacial",
        }
    )
)
for columna in ["Puntaje total", "Puntaje temporal", "Puntaje espacial"]:
    ranking_final_resumido[columna] = ranking_final_resumido[columna].round(4)
ranking_final_resumido.to_excel(
    salida_tablas / "ranking_final_resumido_para_tesis.xlsx",
    index=False,
)
ranking_final_resumido.to_csv(
    salida_tablas / "ranking_final_resumido_para_tesis.csv",
    index=False,
    encoding="utf-8-sig",
)

ranking_final_resumido


# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, grupo_id in zip(axes, GRUPOS_EVALUACION):
    sub = ranking_final[ranking_final["grupo_id"] == grupo_id].sort_values("score_total", ascending=True)
    ax.barh(sub["tecnica"], sub["score_total"], color=[COLORES_TECNICAS.get(t, "#777777") for t in sub["tecnica"]])
    ax.set_xlim(0, 1)
    ax.set_title(f"Ranking final - {grupo_id}: {NOMBRES_GRUPOS[grupo_id]}")
    ax.set_xlabel("Puntaje compuesto")
    for y, (_, fila) in enumerate(sub.iterrows()):
        ax.text(fila["score_total"] + 0.01, y, f"#{int(fila['ranking_grupo'])}", va="center", fontsize=10)
guardar_figura("fig_554_ranking_final_tecnicas_por_grupo.png")

metricas_plot = validacion_espacial.copy()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, grupo_id in zip(axes, GRUPOS_EVALUACION):
    sub = metricas_plot[metricas_plot["grupo_id"] == grupo_id]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["moran_i"], width=0.4, label="Moran I", color="#4C78A8")
    ax.bar(x + 0.2, sub["gi_hotspots_sig_pct"], width=0.4, label="Gi* hotspots (%)", color="#E15759")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["tecnica"])
    ax.set_title(f"Métricas espaciales principales - {grupo_id}")
    ax.legend(frameon=False)
guardar_figura("fig_555_metricas_espaciales_comparativas.png")
