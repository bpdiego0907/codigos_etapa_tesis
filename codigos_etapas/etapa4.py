from pathlib import Path
import os

base_dir = Path.cwd()
matplotlib_cache = base_dir / ".matplotlib_cache"
matplotlib_cache.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ruta = base_dir / "data"

if not ruta.exists():
    ruta = base_dir.parent / "data"

df = pd.read_csv(ruta / "denuncias_total_huaral.csv")
df_url2 = pd.read_csv(ruta / "denuncias_huaral_url2.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df.head()

# %% [markdown]
# ## Etapa 3: Análisis exploratorio y delimitación del conjunto analítico
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import unicodedata
import re
import warnings
plt.style.use("default")
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.max_open_warning": 0,
})
salida_base = Path("outputs_etapa3")
salida_figuras = salida_base / "figuras"
salida_tablas = salida_base / "tablas"
salida_datasets = salida_base / "datasets"
for carpeta in [salida_figuras, salida_tablas, salida_datasets]:
    carpeta.mkdir(parents=True, exist_ok=True)
columnas_categoricas = [
    "departamento_hecho", "provincia_hecho", "distrito_hecho",
    "materia_hecho", "tipo_hecho", "subtipo_hecho", "modalidad_hecho",
    "turno_hecho", "fuente"
]
columnas_numericas = ["año_hecho", "mes_hecho", "dia_hecho", "hora_inicio_hecho"]
def advertir(mensaje):
    warnings.warn(mensaje, stacklevel=2)
    print(f"ADVERTENCIA: {mensaje}")
def normalizar_texto(x):
    if pd.isna(x):
        return np.nan
    texto = str(x).strip().upper()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto if texto else np.nan
def limpiar_nombre_archivo(texto):
    texto = normalizar_texto(texto)
    if pd.isna(texto):
        texto = "SIN_NOMBRE"
    texto = re.sub(r"[^A-Z0-9]+", "_", str(texto)).strip("_").lower()
    return texto[:120] or "sin_nombre"
def guardar_figura(nombre):
    ruta_figura = salida_figuras / nombre
    plt.tight_layout()
    plt.savefig(ruta_figura, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    return ruta_figura
def tabla_frecuencia(dataframe, columna):
    if columna not in dataframe.columns:
        advertir(f"No existe la columna {columna}.")
        return pd.DataFrame(columns=[columna, "frecuencia", "porcentaje", "porcentaje_acumulado"])
    tabla = (
        dataframe[columna]
        .fillna("SIN DATO")
        .value_counts(dropna=False)
        .rename_axis(columna)
        .reset_index(name="frecuencia")
    )
    total = tabla["frecuencia"].sum()
    tabla["porcentaje"] = np.where(total > 0, tabla["frecuencia"] / total * 100, 0).round(2)
    tabla["porcentaje_acumulado"] = tabla["porcentaje"].cumsum().round(2)
    return tabla
def exportar_excel(tabla, nombre):
    ruta_archivo = salida_tablas / nombre
    tabla.to_excel(ruta_archivo, index=False)
    return ruta_archivo
def exportar_csv(dataframe, nombre):
    ruta_archivo = salida_datasets / nombre
    dataframe.to_csv(ruta_archivo, index=False, encoding="utf-8-sig")
    return ruta_archivo
def barras_horizontales(tabla, categoria, valor, titulo, nombre, top=20, color="#4C78A8", mostrar_porcentaje=False):
    if tabla.empty or categoria not in tabla.columns or valor not in tabla.columns:
        advertir(f"No hay datos suficientes para graficar {titulo}.")
        return None
    datos = tabla.head(top).sort_values(valor)
    plt.figure(figsize=(10, max(4, 0.35 * len(datos))))
    barras = plt.barh(datos[categoria].astype(str), datos[valor], color=color)
    max_valor = datos[valor].max()
    desplazamiento = max_valor * 0.015 if max_valor > 0 else 1
    for idx, barra in enumerate(barras):
        ancho = barra.get_width()
        if mostrar_porcentaje and "porcentaje" in datos.columns:
            etiqueta = f"{int(ancho):,} ({datos.iloc[idx]['porcentaje']:.1f}%)"
        else:
            etiqueta = f"{int(ancho):,}"
        plt.text(
            ancho + desplazamiento,
            barra.get_y() + barra.get_height() / 2,
            etiqueta,
            va="center",
            ha="left",
            fontsize=9
        )
    plt.xlim(0, max_valor * 1.22 if max_valor > 0 and mostrar_porcentaje else max_valor * 1.15 if max_valor > 0 else 1)
    plt.title(titulo)
    plt.xlabel("Registros")
    plt.ylabel("")
    return guardar_figura(nombre)
def tabla_cruzada_conteo(dataframe, columnas):
    faltantes = [c for c in columnas if c not in dataframe.columns]
    if faltantes:
        advertir(f"Faltan columnas para tabla cruzada: {faltantes}.")
        return pd.DataFrame(columns=columnas + ["cantidad"])
    return dataframe.groupby(columnas, dropna=False).size().reset_index(name="cantidad").sort_values("cantidad", ascending=False)
def pivot_conteo(dataframe, index, columns):
    if index not in dataframe.columns or columns not in dataframe.columns:
        advertir(f"No se puede crear tabla cruzada {index} vs {columns}.")
        return pd.DataFrame()
    return pd.pivot_table(dataframe, index=index, columns=columns, values="fecha_hora_hecho", aggfunc="count", fill_value=0)

# %% [markdown]
# ## 1. Análisis temporal general del dataset completo
# 

# %%
df_eda = df.copy()
if "fecha_hora_hecho" in df_eda.columns:
    df_eda["fecha_hora_hecho"] = pd.to_datetime(df_eda["fecha_hora_hecho"], errors="coerce")
else:
    advertir("No existe fecha_hora_hecho; se crea columna vacía.")
    df_eda["fecha_hora_hecho"] = pd.NaT
for columna in columnas_numericas:
    if columna in df_eda.columns:
        df_eda[columna] = pd.to_numeric(df_eda[columna], errors="coerce")
    else:
        advertir(f"No existe {columna}; se crea columna vacía.")
        df_eda[columna] = np.nan
for columna in columnas_categoricas:
    if columna in df_eda.columns:
        df_eda[columna] = df_eda[columna].apply(normalizar_texto)
    else:
        advertir(f"No existe {columna}; se crea columna vacía.")
        df_eda[columna] = np.nan
df_eda["modalidad_hecho"] = df_eda["modalidad_hecho"].fillna("SIN MODALIDAD ESPECIFICADA")
df_eda["periodo_mensual"] = df_eda["fecha_hora_hecho"].dt.to_period("M").astype(str)
df_eda.loc[df_eda["fecha_hora_hecho"].isna(), "periodo_mensual"] = np.nan
df_eda["fecha_mes"] = df_eda["fecha_hora_hecho"].dt.to_period("M").dt.to_timestamp()
df_eda.head()

# %%
registros_por_anio = df_eda.groupby("año_hecho", dropna=False).size().reset_index(name="cantidad").sort_values("año_hecho")
exportar_excel(registros_por_anio, "dataset_completo_registros_por_anio.xlsx")
registros_por_anio

# %%
registros_por_mes = df_eda.groupby(["fecha_mes", "periodo_mensual"], dropna=False).size().reset_index(name="cantidad").sort_values("fecha_mes")
exportar_excel(registros_por_mes, "dataset_completo_registros_por_mes.xlsx")
registros_por_mes

# %%
registros_por_turno = tabla_frecuencia(df_eda, "turno_hecho")
exportar_excel(registros_por_turno, "dataset_completo_registros_por_turno.xlsx")
registros_por_turno

# %%
registros_por_fuente = tabla_frecuencia(df_eda, "fuente")
exportar_excel(registros_por_fuente, "dataset_completo_registros_por_fuente.xlsx")
registros_por_fuente

# %%
plt.figure(figsize=(9, 5))
plt.bar(registros_por_anio["año_hecho"].astype("Int64").astype(str), registros_por_anio["cantidad"], color="#4C78A8")
plt.title("Registros por año - dataset completo")
plt.xlabel("Año")
plt.ylabel("Registros")
guardar_figura("fig_01_dataset_completo_registros_por_anio.png")

# %%
datos_mes = registros_por_mes.dropna(subset=["fecha_mes"])
plt.figure(figsize=(12, 5))
plt.plot(datos_mes["fecha_mes"], datos_mes["cantidad"], marker="o", linewidth=1.8, color="#4C78A8")
plt.title("Evolución mensual de registros - dataset completo")
plt.xlabel("Mes")
plt.ylabel("Registros")
guardar_figura("fig_02_dataset_completo_evolucion_mensual.png")

# %%
plt.figure(figsize=(9, 5))
plt.bar(registros_por_turno["turno_hecho"].astype(str), registros_por_turno["frecuencia"], color="#59A14F")
plt.title("Registros por turno - dataset completo")
plt.xlabel("Turno")
plt.ylabel("Registros")
plt.xticks(rotation=30, ha="right")
guardar_figura("fig_03_dataset_completo_registros_por_turno.png")

# %% [markdown]
# ## 2. Análisis categórico del dataset completo
# 

# %%
freq_tipo_hecho = tabla_frecuencia(df_eda, "tipo_hecho")
exportar_excel(freq_tipo_hecho, "dataset_completo_frecuencia_tipo_hecho.xlsx")
freq_tipo_hecho

# %%
freq_subtipo_hecho = tabla_frecuencia(df_eda, "subtipo_hecho")
exportar_excel(freq_subtipo_hecho, "dataset_completo_frecuencia_subtipo_hecho.xlsx")
freq_subtipo_hecho

# %%
freq_modalidad_hecho = tabla_frecuencia(df_eda, "modalidad_hecho")
exportar_excel(freq_modalidad_hecho, "dataset_completo_frecuencia_modalidad_hecho.xlsx")
freq_modalidad_hecho

# %%
freq_materia_hecho = tabla_frecuencia(df_eda, "materia_hecho")
exportar_excel(freq_materia_hecho, "dataset_completo_frecuencia_materia_hecho.xlsx")
freq_materia_hecho

# %%
# Frecuencia de categorías de materia_hecho
barras_horizontales(
    freq_materia_hecho,
    "materia_hecho",
    "frecuencia",
    "Categorías de materia de hecho - dataset completo",
    "fig_06b_dataset_completo_materia_hecho.png",
    top=len(freq_materia_hecho),
    color="#76B7B2"
)

# %%
freq_fuente = tabla_frecuencia(df_eda, "fuente")
exportar_excel(freq_fuente, "dataset_completo_frecuencia_fuente.xlsx")
freq_fuente

# %%
barras_horizontales(freq_tipo_hecho, "tipo_hecho", "frecuencia", "Principales tipos de hecho - dataset completo", "fig_04_dataset_completo_tipo_hecho.png", top=20, color="#4C78A8")

# %%
barras_horizontales(freq_subtipo_hecho, "subtipo_hecho", "frecuencia", "Principales subtipos de hecho - dataset completo", "fig_05_dataset_completo_subtipo_hecho.png", top=20, color="#59A14F")

# %%
barras_horizontales(freq_modalidad_hecho, "modalidad_hecho", "frecuencia", "Principales modalidades de hecho - dataset completo", "fig_06_dataset_completo_modalidad_hecho.png", top=20, color="#F28E2B")

# %%
tipo_subtipo_modalidad = tabla_cruzada_conteo(df_eda, ["tipo_hecho", "subtipo_hecho", "modalidad_hecho"])
exportar_excel(tipo_subtipo_modalidad, "dataset_completo_tipo_subtipo_modalidad.xlsx")
tipo_subtipo_modalidad

# %% [markdown]
# ## 3. Extracción de modalidades actuales desde df_url2
# 

# %%
df_url2_ref = df_url2.copy()
if "modalidad_hecho" not in df_url2_ref.columns:
    advertir("df_url2 no contiene modalidad_hecho; se crea columna vacía.")
    df_url2_ref["modalidad_hecho"] = np.nan
df_url2_ref["modalidad_hecho"] = df_url2_ref["modalidad_hecho"].apply(normalizar_texto).fillna("SIN MODALIDAD ESPECIFICADA")
modalidades_url2 = pd.DataFrame({
    "modalidad_hecho_url2": sorted(df_url2_ref["modalidad_hecho"].dropna().unique())
})
exportar_excel(modalidades_url2, "modalidades_actuales_url2.xlsx")
modalidades_url2

# %%
frecuencia_modalidades_url2 = tabla_frecuencia(df_url2_ref, "modalidad_hecho").rename(columns={
    "frecuencia": "cantidad_url2",
    "porcentaje": "porcentaje_url2"
}).drop(columns=["porcentaje_acumulado"], errors="ignore")
exportar_excel(frecuencia_modalidades_url2, "frecuencia_modalidades_url2.xlsx")
frecuencia_modalidades_url2

# %%
print(f"Cantidad de modalidades únicas en df_url2: {modalidades_url2.shape[0]}")
print("Modalidades únicas:")
for modalidad in modalidades_url2["modalidad_hecho_url2"]:
    print(f"- {modalidad}")

# %% [markdown]
# ## 4. Filtrado del dataset principal según modalidades actuales de df_url2
# 

# %%
modalidades_actuales = set(df_url2_ref["modalidad_hecho"].dropna().unique())
condiciones = [
    df_eda["modalidad_hecho"].isin(modalidades_actuales),
    df_eda["modalidad_hecho"].eq("SIN MODALIDAD ESPECIFICADA")
]
opciones = ["MODALIDAD ACTUAL", "SIN MODALIDAD ESPECIFICADA"]
df_eda["estado_modalidad_url2"] = np.select(condiciones, opciones, default="MODALIDAD NO ACTUAL")
df_modalidades_actuales = df_eda[df_eda["estado_modalidad_url2"] == "MODALIDAD ACTUAL"].copy()
df_modalidades_no_actuales = df_eda[df_eda["estado_modalidad_url2"] == "MODALIDAD NO ACTUAL"].copy()
exportar_csv(df_modalidades_actuales, "dataset_modalidades_actuales.csv")
exportar_csv(df_modalidades_no_actuales, "dataset_modalidades_no_actuales.csv")
df_modalidades_actuales.head()

# %%
resumen_filtrado_modalidades_actuales = pd.DataFrame({
    "categoria": [
        "Dataset completo",
        "Registros con modalidad actual",
        "Registros con modalidad no actual",
        "Registros sin modalidad especificada",
        "Total"
    ],
    "cantidad": [
        len(df_eda),
        (df_eda["estado_modalidad_url2"] == "MODALIDAD ACTUAL").sum(),
        (df_eda["estado_modalidad_url2"] == "MODALIDAD NO ACTUAL").sum(),
        (df_eda["estado_modalidad_url2"] == "SIN MODALIDAD ESPECIFICADA").sum(),
        len(df_eda)
    ]
})
exportar_excel(resumen_filtrado_modalidades_actuales, "resumen_filtrado_modalidades_actuales.xlsx")
resumen_filtrado_modalidades_actuales

# %%
frecuencia_estado_modalidad_url2 = tabla_frecuencia(df_eda, "estado_modalidad_url2")
exportar_excel(frecuencia_estado_modalidad_url2, "frecuencia_estado_modalidad_url2.xlsx")
frecuencia_estado_modalidad_url2

# %%
modalidades_no_actuales_en_dataset_principal = tabla_cruzada_conteo(df_modalidades_no_actuales, ["tipo_hecho", "subtipo_hecho", "modalidad_hecho"])
total_no_actuales = modalidades_no_actuales_en_dataset_principal["cantidad"].sum()
modalidades_no_actuales_en_dataset_principal["porcentaje"] = np.where(total_no_actuales > 0, modalidades_no_actuales_en_dataset_principal["cantidad"] / total_no_actuales * 100, 0).round(2)
exportar_excel(modalidades_no_actuales_en_dataset_principal, "modalidades_no_actuales_en_dataset_principal.xlsx")
modalidades_no_actuales_en_dataset_principal

# %%
modalidades_actuales_en_dataset_principal = tabla_cruzada_conteo(df_modalidades_actuales, ["tipo_hecho", "subtipo_hecho", "modalidad_hecho"])
total_actuales = modalidades_actuales_en_dataset_principal["cantidad"].sum()
modalidades_actuales_en_dataset_principal["porcentaje"] = np.where(total_actuales > 0, modalidades_actuales_en_dataset_principal["cantidad"] / total_actuales * 100, 0).round(2)
exportar_excel(modalidades_actuales_en_dataset_principal, "modalidades_actuales_en_dataset_principal.xlsx")
modalidades_actuales_en_dataset_principal

# %%
estados_orden = ["MODALIDAD ACTUAL", "MODALIDAD NO ACTUAL", "SIN MODALIDAD ESPECIFICADA"]
datos_estado = frecuencia_estado_modalidad_url2.set_index("estado_modalidad_url2").reindex(estados_orden).reset_index().fillna({"frecuencia": 0})
plt.figure(figsize=(9, 5))
barras = plt.barh(
    datos_estado["estado_modalidad_url2"],
    datos_estado["frecuencia"],
    color=["#59A14F", "#E15759", "#9D9D9D"]
)
max_frecuencia = datos_estado["frecuencia"].max()
desplazamiento = max_frecuencia * 0.015 if max_frecuencia > 0 else 1
for barra in barras:
    ancho = barra.get_width()
    plt.text(
        ancho + desplazamiento,
        barra.get_y() + barra.get_height() / 2,
        f"{int(ancho):,}",
        va="center",
        ha="left",
        fontsize=9
    )
plt.xlim(0, max_frecuencia * 1.15 if max_frecuencia > 0 else 1)
plt.title("Filtrado del dataset según modalidades actuales de la URL vigente")
plt.xlabel("Registros")
plt.ylabel("")
guardar_figura("fig_07_filtrado_modalidades_actuales.png")

# %% [markdown]
# ## 5. Análisis temporal del dataset con modalidades actuales
# 

# %%
modalidades_actuales_registros_por_mes = (
    df_modalidades_actuales.groupby(["fecha_mes", "periodo_mensual"], dropna=False)
    .size().reset_index(name="cantidad").sort_values("fecha_mes")
)
exportar_excel(modalidades_actuales_registros_por_mes, "modalidades_actuales_registros_por_mes.xlsx")
modalidades_actuales_registros_por_mes

# %%
modalidades_actuales_registros_por_anio = (
    df_modalidades_actuales
    .groupby("año_hecho", dropna=False)
    .size()
    .reset_index(name="cantidad")
    .dropna(subset=["año_hecho"])
    .sort_values("año_hecho")
)
modalidades_actuales_registros_por_anio["año_hecho"] = modalidades_actuales_registros_por_anio["año_hecho"].astype(int)
exportar_excel(modalidades_actuales_registros_por_anio, "modalidades_actuales_registros_por_anio.xlsx")
plt.figure(figsize=(9, 5))
plt.plot(
    modalidades_actuales_registros_por_anio["año_hecho"],
    modalidades_actuales_registros_por_anio["cantidad"],
    marker="o",
    linewidth=1.8,
    color="#4C78A8"
)
for _, fila in modalidades_actuales_registros_por_anio.iterrows():
    plt.text(
        fila["año_hecho"],
        fila["cantidad"],
        f"{int(fila['cantidad']):,}",
        ha="center",
        va="bottom",
        fontsize=9
    )
plt.title("Evolución anual de registros con modalidades actuales")
plt.xlabel("Año")
plt.ylabel("Registros")
plt.xticks(modalidades_actuales_registros_por_anio["año_hecho"])
guardar_figura("fig_08a_modalidades_actuales_evolucion_anual.png")
modalidades_actuales_registros_por_anio

# %%
datos_mes_actuales = modalidades_actuales_registros_por_mes.dropna(subset=["fecha_mes"])
plt.figure(figsize=(12, 5))
plt.plot(datos_mes_actuales["fecha_mes"], datos_mes_actuales["cantidad"], marker="o", linewidth=1.8, color="#4C78A8")
plt.title("Evolución mensual de registros con modalidades actuales")
plt.xlabel("Mes")
plt.ylabel("Registros")
guardar_figura("fig_08_modalidades_actuales_evolucion_mensual.png")

# %%
df_actuales_2025 = df_modalidades_actuales[df_modalidades_actuales["año_hecho"] == 2025].copy()
conteo_mensual_2025 = (
    df_actuales_2025.groupby("mes_hecho", dropna=False)
    .size()
    .reset_index(name="cantidad")
    .query("mes_hecho == mes_hecho")
)
meses_2025 = pd.DataFrame({"mes_hecho": range(1, 13)})
conteo_mensual_2025 = meses_2025.merge(conteo_mensual_2025, on="mes_hecho", how="left").fillna({"cantidad": 0})
conteo_mensual_2025["cantidad"] = conteo_mensual_2025["cantidad"].astype(int)
conteo_mensual_2025["mes_nombre"] = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
exportar_excel(conteo_mensual_2025, "registros_mensuales_2025_modalidades_actuales.xlsx")
conteo_mensual_2025

# %%
# Promedio mensual de registros por año con modalidades actuales
registros_mensuales_por_anio = (
    df_modalidades_actuales
    .dropna(subset=["fecha_mes", "año_hecho"])
    .groupby(["año_hecho", "periodo_mensual"], dropna=False)
    .size()
    .reset_index(name="cantidad_mensual")
)

promedio_mensual_por_anio_modalidades_actuales = (
    registros_mensuales_por_anio
    .groupby("año_hecho", as_index=False)
    .agg(
        registros_totales=("cantidad_mensual", "sum"),
        meses_observados=("periodo_mensual", "nunique"),
        promedio_mensual=("cantidad_mensual", "mean")
    )
    .sort_values("año_hecho")
)
promedio_mensual_por_anio_modalidades_actuales["año_hecho"] = promedio_mensual_por_anio_modalidades_actuales["año_hecho"].astype(int)
promedio_mensual_por_anio_modalidades_actuales["promedio_mensual"] = promedio_mensual_por_anio_modalidades_actuales["promedio_mensual"].round(2)
promedio_global_promedios_anuales = promedio_mensual_por_anio_modalidades_actuales["promedio_mensual"].mean()

exportar_excel(promedio_mensual_por_anio_modalidades_actuales, "promedio_mensual_por_anio_modalidades_actuales.xlsx")

plt.figure(figsize=(10, 5))
barras = plt.bar(
    promedio_mensual_por_anio_modalidades_actuales["año_hecho"].astype(str),
    promedio_mensual_por_anio_modalidades_actuales["promedio_mensual"],
    color="#4C78A8"
)
plt.axhline(
    promedio_global_promedios_anuales,
    color="#E15759",
    linestyle="--",
    linewidth=1.8,
    label=f"Promedio general: {promedio_global_promedios_anuales:,.1f}"
)

max_promedio_anual = promedio_mensual_por_anio_modalidades_actuales["promedio_mensual"].max()
desplazamiento = max_promedio_anual * 0.015 if max_promedio_anual > 0 else 1
for barra in barras:
    alto = barra.get_height()
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        alto + desplazamiento,
        f"{alto:,.1f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.ylim(0, max(max_promedio_anual, promedio_global_promedios_anuales) * 1.18 if max_promedio_anual > 0 else 1)
plt.title("Promedio mensual de registros por año - modalidades actuales")
plt.xlabel("Año")
plt.ylabel("Promedio mensual de registros")
plt.legend()
guardar_figura("fig_09_registros_mensuales_2025_modalidades_actuales.png")

promedio_mensual_por_anio_modalidades_actuales

# %%
cobertura_2025_por_modalidad = pivot_conteo(df_actuales_2025, "modalidad_hecho", "periodo_mensual")
cobertura_2025_por_modalidad.to_excel(salida_tablas / "cobertura_2025_modalidades_actuales_por_modalidad.xlsx")
cobertura_2025_por_modalidad

# %%
cobertura_2025_por_tipo = pivot_conteo(df_actuales_2025, "tipo_hecho", "periodo_mensual")
cobertura_2025_por_tipo.to_excel(salida_tablas / "cobertura_2025_modalidades_actuales_por_tipo.xlsx")
cobertura_2025_por_tipo

# %%
# Evolución mensual 2025 por tipo de hecho: tipos principales
meses_2025_periodos = [f"2025-{mes:02d}" for mes in range(1, 13)]
etiquetas_meses_2025 = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
datos_tipo_2025 = cobertura_2025_por_tipo.reindex(columns=meses_2025_periodos, fill_value=0)
datos_tipo_2025 = datos_tipo_2025.loc[datos_tipo_2025.sum(axis=1).sort_values(ascending=False).index]
umbral_baja_frecuencia = 10
tipos_baja_frecuencia = datos_tipo_2025.index[datos_tipo_2025.max(axis=1) <= umbral_baja_frecuencia]
tipos_principales = datos_tipo_2025.index[datos_tipo_2025.max(axis=1) > umbral_baja_frecuencia]
datos_tipo_2025_principales = datos_tipo_2025.loc[tipos_principales]
datos_tipo_2025_baja_frecuencia = datos_tipo_2025.loc[tipos_baja_frecuencia]
plt.figure(figsize=(12, 6))
if datos_tipo_2025_principales.empty:
    plt.text(0.5, 0.5, "No hay tipos de hecho con más de 10 registros mensuales", ha="center", va="center", transform=plt.gca().transAxes)
else:
    for tipo_hecho, fila in datos_tipo_2025_principales.iterrows():
        plt.plot(range(1, 13), fila.values, marker="o", linewidth=1.8, label=tipo_hecho)
    plt.legend(title="Tipo de hecho", fontsize=8, title_fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title("Evolución mensual de registros por tipo de hecho - 2025")
plt.xlabel("Mes")
plt.ylabel("Registros")
plt.xticks(range(1, 13), etiquetas_meses_2025)
guardar_figura("fig_09b_evolucion_mensual_2025_por_tipo_hecho.png")

# %%
# Evolución mensual 2025 por tipo de hecho: baja frecuencia
plt.figure(figsize=(12, 6))
if datos_tipo_2025_baja_frecuencia.empty:
    plt.text(0.5, 0.5, "No hay tipos de hecho con escala mensual de 0 a 10", ha="center", va="center", transform=plt.gca().transAxes)
else:
    for tipo_hecho, fila in datos_tipo_2025_baja_frecuencia.iterrows():
        plt.plot(range(1, 13), fila.values, marker="o", linewidth=1.8, label=tipo_hecho)
    plt.legend(title="Tipo de hecho", fontsize=8, title_fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title("Evolución mensual de tipos de hecho de baja frecuencia - 2025")
plt.xlabel("Mes")
plt.ylabel("Registros")
plt.xticks(range(1, 13), etiquetas_meses_2025)
plt.ylim(0, 10)
plt.yticks(range(0, 11, 1))
guardar_figura("fig_09c_evolucion_mensual_2025_por_tipo_hecho_baja_frecuencia.png")

# %%
cobertura_2025_por_subtipo = pivot_conteo(df_actuales_2025, "subtipo_hecho", "periodo_mensual")
cobertura_2025_por_subtipo.to_excel(salida_tablas / "cobertura_2025_modalidades_actuales_por_subtipo.xlsx")
cobertura_2025_por_subtipo

# %%
# Evolución mensual 2025 por subtipo de hecho: subtipos principales
meses_2025_periodos = [f"2025-{mes:02d}" for mes in range(1, 13)]
etiquetas_meses_2025 = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
datos_subtipo_2025 = cobertura_2025_por_subtipo.reindex(columns=meses_2025_periodos, fill_value=0)
datos_subtipo_2025 = datos_subtipo_2025.loc[datos_subtipo_2025.sum(axis=1).sort_values(ascending=False).index]
subtipos_principales_2025 = datos_subtipo_2025.head(10)
plt.figure(figsize=(12, 6))
if subtipos_principales_2025.empty:
    plt.text(0.5, 0.5, "No hay subtipos de hecho para graficar", ha="center", va="center", transform=plt.gca().transAxes)
else:
    for subtipo_hecho, fila in subtipos_principales_2025.iterrows():
        plt.plot(range(1, 13), fila.values, marker="o", linewidth=1.8, label=subtipo_hecho)
    plt.legend(title="Subtipo de hecho", fontsize=8, title_fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title("Evolución mensual de registros por subtipo de hecho principal - 2025")
plt.xlabel("Mes")
plt.ylabel("Registros")
plt.xticks(range(1, 13), etiquetas_meses_2025)
guardar_figura("fig_09d_evolucion_mensual_2025_por_subtipo_hecho_principales.png")

# %%
# Dataset delimitado para análisis: modalidades actuales, sin exclusión temporal.
df_modalidades_actuales_delimitado = df_modalidades_actuales.copy()
# Alias para mantener compatibilidad con las celdas posteriores del notebook.
df_modalidades_actuales_periodo_efectivo = df_modalidades_actuales_delimitado.copy()
exportar_csv(df_modalidades_actuales_delimitado, "dataset_modalidades_actuales_delimitado.csv")
df_modalidades_actuales_delimitado.head()

# %%
df_modalidades_actuales_periodo_efectivo.shape

# %% [markdown]
# ## 6. Análisis categórico del dataset filtrado por modalidades actuales
# 

# %%
actuales_frecuencia_tipo_hecho = tabla_frecuencia(df_modalidades_actuales_periodo_efectivo, "tipo_hecho")
exportar_excel(actuales_frecuencia_tipo_hecho, "actuales_frecuencia_tipo_hecho.xlsx")
actuales_frecuencia_tipo_hecho

# %%
actuales_frecuencia_subtipo_hecho = tabla_frecuencia(df_modalidades_actuales_periodo_efectivo, "subtipo_hecho")
exportar_excel(actuales_frecuencia_subtipo_hecho, "actuales_frecuencia_subtipo_hecho.xlsx")
actuales_frecuencia_subtipo_hecho

# %%
actuales_frecuencia_modalidad_hecho = tabla_frecuencia(df_modalidades_actuales_periodo_efectivo, "modalidad_hecho")
exportar_excel(actuales_frecuencia_modalidad_hecho, "actuales_frecuencia_modalidad_hecho.xlsx")
actuales_frecuencia_modalidad_hecho

# %%
actuales_frecuencia_distrito = tabla_frecuencia(df_modalidades_actuales_periodo_efectivo, "distrito_hecho")
exportar_excel(actuales_frecuencia_distrito, "actuales_frecuencia_distrito.xlsx")
actuales_frecuencia_distrito

# %%
actuales_frecuencia_turno = tabla_frecuencia(df_modalidades_actuales_periodo_efectivo, "turno_hecho")
exportar_excel(actuales_frecuencia_turno, "actuales_frecuencia_turno.xlsx")
actuales_frecuencia_turno

# %%
actuales_frecuencia_fuente = tabla_frecuencia(df_modalidades_actuales_periodo_efectivo, "fuente")
exportar_excel(actuales_frecuencia_fuente, "actuales_frecuencia_fuente.xlsx")
actuales_frecuencia_fuente

# %%
barras_horizontales(actuales_frecuencia_tipo_hecho, "tipo_hecho", "frecuencia", "Tipos de hecho en dataset delimitado - modalidades actuales", "fig_10_actuales_tipo_hecho.png", top=20, color="#4C78A8", mostrar_porcentaje=True)

# %%
barras_horizontales(actuales_frecuencia_subtipo_hecho, "subtipo_hecho", "frecuencia", "Subtipos de hecho en dataset delimitado - modalidades actuales", "fig_11_actuales_subtipo_hecho.png", top=20, color="#59A14F", mostrar_porcentaje=True)

# %%
barras_horizontales(actuales_frecuencia_modalidad_hecho, "modalidad_hecho", "frecuencia", "Modalidades actuales más frecuentes - dataset delimitado", "fig_12_actuales_modalidad_hecho.png", top=20, color="#F28E2B", mostrar_porcentaje=True)

# %%
# Pareto de modalidades actuales más críticas
modalidades_pareto_actuales = actuales_frecuencia_modalidad_hecho.head(10).copy()

fig, ax1 = plt.subplots(figsize=(12, 5))
barras = ax1.bar(modalidades_pareto_actuales["modalidad_hecho"], modalidades_pareto_actuales["frecuencia"], color="#F28E2B")
ax1.set_xlabel("Modalidad de hecho")
ax1.set_ylabel("Registros")
ax1.tick_params(axis="x", rotation=45)
for etiqueta in ax1.get_xticklabels():
    etiqueta.set_ha("right")

max_frecuencia_modalidad = modalidades_pareto_actuales["frecuencia"].max()
desplazamiento_modalidad = max_frecuencia_modalidad * 0.015 if max_frecuencia_modalidad > 0 else 1
for idx, barra in enumerate(barras):
    alto = barra.get_height()
    porcentaje = modalidades_pareto_actuales.iloc[idx]["porcentaje"]
    ax1.text(
        barra.get_x() + barra.get_width() / 2,
        alto + desplazamiento_modalidad,
        f"{int(alto):,} ({porcentaje:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=8
    )
ax1.set_ylim(0, max_frecuencia_modalidad * 1.22 if max_frecuencia_modalidad > 0 else 1)

ax2 = ax1.twinx()
ax2.plot(
    modalidades_pareto_actuales["modalidad_hecho"],
    modalidades_pareto_actuales["porcentaje_acumulado"],
    color="#E15759",
    marker="o",
    linewidth=1.8
)
for x, porcentaje_acumulado in zip(modalidades_pareto_actuales["modalidad_hecho"], modalidades_pareto_actuales["porcentaje_acumulado"]):
    ax2.text(x, porcentaje_acumulado + 2, f"{porcentaje_acumulado:.1f}%", ha="center", va="bottom", fontsize=8, color="#A23A3A")
ax2.set_ylabel("Porcentaje acumulado")
ax2.set_ylim(0, 110)

plt.title("Pareto de modalidades actuales más frecuentes")
guardar_figura("fig_12b_actuales_modalidad_hecho_pareto.png")

# %%
barras_horizontales(actuales_frecuencia_distrito, "distrito_hecho", "frecuencia", "Registros por distrito - modalidades actuales dataset delimitado", "fig_13_actuales_distrito.png", top=20, color="#76B7B2", mostrar_porcentaje=True)

# %%
plt.figure(figsize=(9, 5))
barras = plt.bar(actuales_frecuencia_turno["turno_hecho"].astype(str), actuales_frecuencia_turno["frecuencia"], color="#59A14F")
max_turno = actuales_frecuencia_turno["frecuencia"].max()
desplazamiento_turno = max_turno * 0.015 if max_turno > 0 else 1
for idx, barra in enumerate(barras):
    alto = barra.get_height()
    porcentaje = actuales_frecuencia_turno.iloc[idx]["porcentaje"] if "porcentaje" in actuales_frecuencia_turno.columns else np.nan
    etiqueta = f"{int(alto):,} ({porcentaje:.1f}%)" if pd.notna(porcentaje) else f"{int(alto):,}"
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        alto + desplazamiento_turno,
        etiqueta,
        ha="center",
        va="bottom",
        fontsize=9
    )
plt.ylim(0, max_turno * 1.2 if max_turno > 0 else 1)
plt.title("Registros por turno - modalidades actuales dataset delimitado")
plt.xlabel("Turno")
plt.ylabel("Registros")
plt.xticks(rotation=30, ha="right")
guardar_figura("fig_14_actuales_turno.png")

# %% [markdown]
# ## 7. Agrupación analítica del dataset

def asignar_grupo_analitico(tipo_hecho):
    tipo = normalizar_texto(tipo_hecho)
    if tipo == "LEY DE VIOLENCIA CONTRA LA MUJER Y GRUPOS VULNERABLES":
        return pd.Series({
            "grupo_id": "g1",
            "grupo": "Violencia contra la mujer y grupos vulnerables"
        })
    if tipo == "PATRIMONIO (DELITO)":
        return pd.Series({
            "grupo_id": "g2",
            "grupo": "Patrimonio"
        })
    return pd.Series({
        "grupo_id": "g3",
        "grupo": "Otros hechos de baja frecuencia"
    })

columnas_grupo = df_modalidades_actuales_periodo_efectivo["tipo_hecho"].apply(asignar_grupo_analitico)
df_modalidades_actuales_periodo_efectivo = pd.concat(
    [df_modalidades_actuales_periodo_efectivo.drop(columns=["grupo_id", "grupo"], errors="ignore"), columnas_grupo],
    axis=1
)
df_modalidades_actuales_delimitado = df_modalidades_actuales_periodo_efectivo.copy()

frecuencia_grupos_analiticos = tabla_frecuencia(df_modalidades_actuales_periodo_efectivo, "grupo")
frecuencia_grupos_analiticos.insert(
    0,
    "grupo_id",
    frecuencia_grupos_analiticos["grupo"].map({
        "Violencia contra la mujer y grupos vulnerables": "g1",
        "Patrimonio": "g2",
        "Otros hechos de baja frecuencia": "g3",
    })
)
frecuencia_grupos_analiticos = frecuencia_grupos_analiticos[["grupo_id", "grupo", "frecuencia", "porcentaje", "porcentaje_acumulado"]]

exportar_csv(df_modalidades_actuales_periodo_efectivo, "dataset_modalidades_actuales_delimitado.csv")
exportar_excel(frecuencia_grupos_analiticos, "dataset_delimitado_frecuencia_grupos_analiticos.xlsx")
frecuencia_grupos_analiticos

# %% [markdown]
# ## 8. Exploración espacial preliminar del distrito de Huaral
# 
import geopandas as gpd
from shapely.geometry import Point
try:
    import pyogrio
    capas_gadm = pyogrio.list_layers("gadm41_PER.gpkg")
    capas_gadm = pd.DataFrame(capas_gadm, columns=["capa", "tipo_geometria"])
except Exception as e:
    capas_gadm = pd.DataFrame({"capa": ["ADM_ADM_3"], "tipo_geometria": ["MultiPolygon"]})
    advertir(f"No se pudieron listar capas con pyogrio; se usará ADM_ADM_3. Detalle: {e}")
exportar_excel(capas_gadm, "espacial_distrito_huaral_capas_gadm41_PER.xlsx")
capas_gadm

# %%
# Carga del límite del distrito de Huaral
ruta_gadm = Path("gadm41_PER.gpkg")
if not ruta_gadm.exists():
    raise FileNotFoundError("No se encontró gadm41_PER.gpkg en la carpeta del proyecto.")
capas_disponibles = capas_gadm["capa"].astype(str).tolist()
capa_distrital = "ADM_ADM_3" if "ADM_ADM_3" in capas_disponibles else next((c for c in capas_disponibles if c.endswith("3") or "ADM_3" in c), capas_disponibles[-1])
gdf_distritos_peru = gpd.read_file(ruta_gadm, layer=capa_distrital)
columnas_distritos_gadm = pd.DataFrame({"columna": gdf_distritos_peru.columns.astype(str)})
exportar_excel(columnas_distritos_gadm, "espacial_distrito_huaral_columnas_gadm.xlsx")
gdf_distritos_peru["pais_norm"] = gdf_distritos_peru["COUNTRY"].apply(normalizar_texto)
gdf_distritos_peru["departamento_norm"] = gdf_distritos_peru["NAME_1"].apply(normalizar_texto)
gdf_distritos_peru["provincia_norm"] = gdf_distritos_peru["NAME_2"].apply(normalizar_texto)
gdf_distritos_peru["distrito_norm"] = gdf_distritos_peru["NAME_3"].apply(normalizar_texto)
gdf_distrito_huaral = gdf_distritos_peru[
    (gdf_distritos_peru["pais_norm"] == "PERU")
    & (gdf_distritos_peru["departamento_norm"] == "LIMA")
    & (gdf_distritos_peru["provincia_norm"] == "HUARAL")
    & (gdf_distritos_peru["distrito_norm"] == "HUARAL")
].copy()
if gdf_distrito_huaral.empty:
    raise ValueError("No se encontró el distrito de Huaral en la capa distrital de GADM.")
gdf_distrito_huaral = gdf_distrito_huaral.to_crs(epsg=4326)
gdf_distrito_huaral.to_file(salida_datasets / "distrito_huaral_gadm41.geojson", driver="GeoJSON")
gdf_distrito_huaral[["COUNTRY", "NAME_1", "NAME_2", "NAME_3", "geometry"]]

# %%
# 
df_espacial = df_modalidades_actuales_periodo_efectivo.copy()
for columna in ["lat_hecho", "long_hecho"]:
    if columna not in df_espacial.columns:
        raise KeyError(f"No existe la columna requerida para análisis espacial: {columna}")
    df_espacial[columna] = pd.to_numeric(df_espacial[columna], errors="coerce")
df_espacial_valido = df_espacial.dropna(subset=["lat_hecho", "long_hecho"]).copy()
df_espacial_valido = df_espacial_valido[
    df_espacial_valido["lat_hecho"].between(-90, 90)
    & df_espacial_valido["long_hecho"].between(-180, 180)
].copy()
gdf_puntos_validos = gpd.GeoDataFrame(
    df_espacial_valido,
    geometry=gpd.points_from_xy(df_espacial_valido["long_hecho"], df_espacial_valido["lat_hecho"]),
    crs="EPSG:4326"
)
gdf_puntos_validos.to_file(salida_datasets / "puntos_validos_dataset_delimitado.geojson", driver="GeoJSON")
print(f"Registros del dataset delimitado: {len(df_modalidades_actuales_periodo_efectivo):,}")
print(f"Registros con coordenadas válidas: {len(gdf_puntos_validos):,}")
gdf_puntos_validos.head()

# %%
try:
    gdf_puntos_huaral_distrito_join = gpd.sjoin(
        gdf_puntos_validos,
        gdf_distrito_huaral[["NAME_3", "distrito_norm", "geometry"]],
        how="left",
        predicate="within"
    )
except TypeError:
    gdf_puntos_huaral_distrito_join = gpd.sjoin(
        gdf_puntos_validos,
        gdf_distrito_huaral[["NAME_3", "distrito_norm", "geometry"]],
        how="left",
        op="within"
    )
registros_dentro_distrito_huaral = gdf_puntos_huaral_distrito_join["index_right"].notna().sum()
registros_fuera_distrito_huaral = gdf_puntos_huaral_distrito_join["index_right"].isna().sum()
gdf_puntos_distrito_huaral = gdf_puntos_huaral_distrito_join[
    gdf_puntos_huaral_distrito_join["index_right"].notna()
].copy()
gdf_puntos_distrito_huaral = gdf_puntos_distrito_huaral.drop(columns=["index_right"], errors="ignore")
gdf_puntos_distrito_huaral.to_file(salida_datasets / "puntos_dataset_delimitado_distrito_huaral.geojson", driver="GeoJSON")
resumen_validacion_espacial_distrito_huaral = pd.DataFrame({
    "indicador": [
        "Registros totales del dataset delimitado",
        "Registros con coordenadas válidas",
        "Registros dentro del distrito de Huaral",
        "Registros fuera del distrito de Huaral"
    ],
    "cantidad": [
        len(df_modalidades_actuales_periodo_efectivo),
        len(gdf_puntos_validos),
        int(registros_dentro_distrito_huaral),
        int(registros_fuera_distrito_huaral)
    ]
})
exportar_excel(resumen_validacion_espacial_distrito_huaral, "espacial_resumen_validacion_distrito_huaral.xlsx")
resumen_validacion_espacial_distrito_huaral

# %%
fig, ax = plt.subplots(figsize=(9, 9))
gdf_distrito_huaral.plot(ax=ax, facecolor="#F2F2F2", edgecolor="#333333", linewidth=1.0)
gdf_puntos_distrito_huaral.plot(ax=ax, color="#E15759", markersize=6, alpha=0.35)
ax.set_title("Distribución espacial preliminar de registros en el distrito de Huaral")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_aspect("equal", adjustable="box")
guardar_figura("fig_21_espacial_distrito_huaral_mapa_general_puntos.png")

# %%
# Mapa de densidad visual de registros en el distrito de Huaral.
# Esta figura muestra concentración relativa de puntos; no identifica zonas calientes formalmente.
# Se usa escala logarítmica porque varias denuncias comparten exactamente la misma coordenada.
from matplotlib.colors import LogNorm
coordenadas_repetidas_huaral = (
    gdf_puntos_distrito_huaral
    .assign(
        long_hecho_redondeada=lambda x: x.geometry.x.round(7),
        lat_hecho_redondeada=lambda x: x.geometry.y.round(7)
    )
    .groupby(["long_hecho_redondeada", "lat_hecho_redondeada"], as_index=False)
    .size()
    .rename(columns={"size": "cantidad_registros"})
    .sort_values("cantidad_registros", ascending=False)
)
exportar_excel(coordenadas_repetidas_huaral, "espacial_distrito_huaral_coordenadas_repetidas.xlsx")
fig, ax = plt.subplots(figsize=(9, 9))
gdf_distrito_huaral.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#333333", linewidth=1.0)
if gdf_puntos_distrito_huaral.empty:
    ax.text(0.5, 0.5, "No hay puntos válidos para graficar", ha="center", va="center", transform=ax.transAxes)
else:
    x = gdf_puntos_distrito_huaral.geometry.x
    y = gdf_puntos_distrito_huaral.geometry.y
    minx, miny, maxx, maxy = gdf_distrito_huaral.total_bounds
    densidad = ax.hexbin(
        x,
        y,
        gridsize=45,
        cmap="YlOrRd",
        mincnt=1,
        alpha=0.80,
        extent=(minx, maxx, miny, maxy),
        norm=LogNorm()
    )
    gdf_distrito_huaral.boundary.plot(ax=ax, color="#333333", linewidth=1.0)
    cbar = fig.colorbar(densidad, ax=ax, shrink=0.72, fraction=0.035, pad=0.02, aspect=25)
    cbar.set_label("Registros por celda hexagonal (escala logarítmica)")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_aspect("equal", adjustable="box")
guardar_figura("fig_21b_espacial_distrito_huaral_mapa_densidad.png")
coordenadas_repetidas_huaral.head(10)

# %%
# Tabla de frecuencia por grupo analítico dentro del distrito de Huaral
espacial_distrito_huaral_grupo = tabla_frecuencia(gdf_puntos_distrito_huaral, "grupo")
espacial_distrito_huaral_grupo.insert(
    0,
    "grupo_id",
    espacial_distrito_huaral_grupo["grupo"].map({
        "Violencia contra la mujer y grupos vulnerables": "g1",
        "Patrimonio": "g2",
        "Otros hechos de baja frecuencia": "g3",
    })
)
espacial_distrito_huaral_grupo = espacial_distrito_huaral_grupo[["grupo_id", "grupo", "frecuencia", "porcentaje", "porcentaje_acumulado"]]
exportar_excel(espacial_distrito_huaral_grupo, "espacial_distrito_huaral_frecuencia_grupo_analitico.xlsx")
espacial_distrito_huaral_grupo

# %%
# Figura 2: registros por grupo analítico dentro del distrito de Huaral
barras_horizontales(
    espacial_distrito_huaral_grupo,
    "grupo",
    "frecuencia",
    "Registros por grupo analítico en el distrito de Huaral",
    "fig_22_espacial_distrito_huaral_grupo_analitico.png",
    top=min(10, len(espacial_distrito_huaral_grupo)),
    color="#4C78A8",
    mostrar_porcentaje=True
)

# %%
# Figura 3: mapa por grupo analítico dentro del distrito de Huaral
orden_grupos_espacial = [
    "Violencia contra la mujer y grupos vulnerables",
    "Patrimonio",
    "Otros hechos de baja frecuencia",
]
colores_grupo_espacial = {
    "Violencia contra la mujer y grupos vulnerables": "#E15759",
    "Patrimonio": "#4C78A8",
    "Otros hechos de baja frecuencia": "#59A14F",
}
fig, ax = plt.subplots(figsize=(9, 9))
gdf_distrito_huaral.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#333333", linewidth=1.0)
for grupo in orden_grupos_espacial:
    datos = gdf_puntos_distrito_huaral[gdf_puntos_distrito_huaral["grupo"] == grupo]
    if datos.empty:
        continue
    datos.plot(
        ax=ax,
        markersize=7,
        alpha=0.45,
        label=grupo,
        color=colores_grupo_espacial.get(grupo, "#9D9D9D")
    )
ax.set_title("Distribución espacial preliminar por grupo analítico - distrito de Huaral")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_aspect("equal", adjustable="box")
ax.legend(title="Grupo analítico", fontsize=8, title_fontsize=9, loc="best")
guardar_figura("fig_23_espacial_distrito_huaral_puntos_por_tipo_hecho.png")

# %%
# Mapas separados por grupo analítico dentro del distrito de Huaral
for idx, grupo_objetivo in enumerate(orden_grupos_espacial, start=1):
    datos_grupo = gdf_puntos_distrito_huaral[gdf_puntos_distrito_huaral["grupo"] == grupo_objetivo]
    if datos_grupo.empty:
        print(f"Sin registros con coordenadas válidas en el distrito de Huaral para: {grupo_objetivo}")
        continue
    fig, ax = plt.subplots(figsize=(8, 8))
    gdf_distrito_huaral.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#333333", linewidth=1.0)
    datos_grupo.plot(
        ax=ax,
        markersize=7,
        alpha=0.45,
        color=colores_grupo_espacial.get(grupo_objetivo, "#9D9D9D")
    )
    ax.set_title(f"Distribución espacial preliminar en Huaral - {grupo_objetivo}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal", adjustable="box")
    guardar_figura(f"fig_23{chr(96 + idx)}_espacial_distrito_huaral_grupo_{limpiar_nombre_archivo(grupo_objetivo)}.png")

# %%
minx, miny, maxx, maxy = gdf_distrito_huaral.total_bounds
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
axes = np.ravel(axes)
for ax, grupo in zip(axes, orden_grupos_espacial):
    gdf_distrito_huaral.plot(ax=ax, facecolor="#F7F7F7", edgecolor="#333333", linewidth=0.8)
    datos_grupo = gdf_puntos_distrito_huaral[gdf_puntos_distrito_huaral["grupo"] == grupo]
    if not datos_grupo.empty:
        datos_grupo.plot(
            ax=ax,
            markersize=6,
            alpha=0.45,
            color=colores_grupo_espacial.get(grupo, "#9D9D9D")
        )
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(grupo, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
fig.suptitle("Distribución espacial preliminar por grupo analítico - distrito de Huaral", y=1.02)
guardar_figura("fig_24_espacial_distrito_huaral_grupos_analiticos_grid.png")

# %%
# Mapa interactivo opcional con Folium 
try:
    import folium
    from folium.plugins import MarkerCluster
    folium_disponible = True
except Exception as e:
    folium_disponible = False
    advertir(f"Folium no disponible: {e}")
ruta_mapa_folium = None
if folium_disponible and not gdf_puntos_distrito_huaral.empty:
    centro_geom = gdf_distrito_huaral.geometry.union_all().centroid if hasattr(gdf_distrito_huaral.geometry, "union_all") else gdf_distrito_huaral.geometry.unary_union.centroid
    mapa_folium = folium.Map(location=[centro_geom.y, centro_geom.x], zoom_start=12, tiles="CartoDB positron")
    folium.GeoJson(
        gdf_distrito_huaral[["NAME_3", "geometry"]].to_json(),
        name="Límite del distrito de Huaral",
        style_function=lambda feature: {"fillColor": "#FFFFFF", "color": "#333333", "weight": 2, "fillOpacity": 0.05},
        tooltip=folium.GeoJsonTooltip(fields=["NAME_3"], aliases=["Distrito"])
    ).add_to(mapa_folium)
    for grupo in orden_grupos_espacial:
        datos_grupo = gdf_puntos_distrito_huaral[gdf_puntos_distrito_huaral["grupo"] == grupo]
        if datos_grupo.empty:
            continue
        capa_grupo = folium.FeatureGroup(name=f"Grupo: {grupo}", show=True)
        cluster_grupo = MarkerCluster(name=f"Cluster {grupo}").add_to(capa_grupo)
        color_grupo = colores_grupo_espacial.get(grupo, "#9D9D9D")
        for _, fila in datos_grupo.iterrows():
            folium.CircleMarker(
                location=[fila["lat_hecho"], fila["long_hecho"]],
                radius=3,
                color=color_grupo,
                fill=True,
                fill_opacity=0.6,
                popup=(
                    f"Grupo: {fila.get('grupo', '')}<br>"
                    f"Grupo ID: {fila.get('grupo_id', '')}<br>"
                    f"Tipo: {fila.get('tipo_hecho', '')}<br>"
                    f"Modalidad: {fila.get('modalidad_hecho', '')}<br>"
                    f"Distrito: {fila.get('distrito_hecho', '')}"
                )
            ).add_to(cluster_grupo)
        capa_grupo.add_to(mapa_folium)
    folium.LayerControl(collapsed=False).add_to(mapa_folium)
    ruta_mapa_folium = salida_datasets / "mapa_interactivo_exploracion_espacial_distrito_huaral_grupos.html"
    mapa_folium.save(ruta_mapa_folium)
    print(f"Mapa Folium guardado en: {ruta_mapa_folium}")
else:
    print("No se generó mapa Folium del distrito de Huaral.")

# %% [markdown]
# ## 10. Exportación de subconjuntos por modalidad actual
# 

# %%
carpeta_modalidades = salida_datasets / "subconjuntos_por_modalidad_actual"
carpeta_tipos = salida_datasets / "subconjuntos_por_tipo_hecho_actual"
carpeta_modalidades.mkdir(parents=True, exist_ok=True)
carpeta_tipos.mkdir(parents=True, exist_ok=True)
archivos_modalidad = []
for modalidad, datos in df_modalidades_actuales_periodo_efectivo.groupby("modalidad_hecho", dropna=False):
    nombre = limpiar_nombre_archivo(modalidad) + ".csv"
    ruta_archivo = carpeta_modalidades / nombre
    datos.to_csv(ruta_archivo, index=False, encoding="utf-8-sig")
    archivos_modalidad.append({"modalidad_hecho": modalidad, "archivo": str(ruta_archivo), "registros": len(datos)})
archivos_tipo = []
for tipo, datos in df_modalidades_actuales_periodo_efectivo.groupby("tipo_hecho", dropna=False):
    nombre = limpiar_nombre_archivo(tipo) + ".csv"
    ruta_archivo = carpeta_tipos / nombre
    datos.to_csv(ruta_archivo, index=False, encoding="utf-8-sig")
    archivos_tipo.append({"tipo_hecho": tipo, "archivo": str(ruta_archivo), "registros": len(datos)})
subconjuntos_exportados_modalidad = pd.DataFrame(archivos_modalidad).sort_values("registros", ascending=False)
subconjuntos_exportados_tipo = pd.DataFrame(archivos_tipo).sort_values("registros", ascending=False)
subconjuntos_exportados_modalidad

# %%
subconjuntos_exportados_tipo
# %%
def rango_temporal(dataframe):
    if dataframe.empty or "fecha_hora_hecho" not in dataframe.columns:
        return "Sin datos"
    inicio = dataframe["fecha_hora_hecho"].min()
    fin = dataframe["fecha_hora_hecho"].max()
    return f"{inicio:%Y-%m-%d} a {fin:%Y-%m-%d}" if pd.notna(inicio) and pd.notna(fin) else "Sin fecha válida"
modalidades_df = set(df_eda["modalidad_hecho"].dropna().unique())
modalidades_actuales_en_df = modalidades_df.intersection(modalidades_actuales)
modalidades_df_no_url2 = modalidades_df.difference(modalidades_actuales)
principales_modalidades_actuales = df_modalidades_actuales["modalidad_hecho"].value_counts().head(15)
print("RESUMEN FINAL ETAPA 3")
print(f"Total de registros del dataset completo: {len(df_eda):,}")
print(f"Total de registros con modalidad actual: {(df_eda['estado_modalidad_url2'] == 'MODALIDAD ACTUAL').sum():,}")
print(f"Total de registros con modalidad no actual: {(df_eda['estado_modalidad_url2'] == 'MODALIDAD NO ACTUAL').sum():,}")
print(f"Total de registros sin modalidad especificada: {(df_eda['estado_modalidad_url2'] == 'SIN MODALIDAD ESPECIFICADA').sum():,}")
print(f"Total de registros del dataset delimitado con modalidades actuales: {len(df_modalidades_actuales_periodo_efectivo):,}")
print(f"Cantidad de modalidades únicas en df: {len(modalidades_df):,}")
print(f"Cantidad de modalidades únicas en df_url2: {len(modalidades_actuales):,}")
print(f"Cantidad de modalidades actuales encontradas en df: {len(modalidades_actuales_en_df):,}")
print(f"Cantidad de modalidades de df que no existen en df_url2: {len(modalidades_df_no_url2):,}")
print(f"Rango temporal del dataset completo: {rango_temporal(df_eda)}")
print(f"Rango temporal del dataset filtrado por modalidades actuales: {rango_temporal(df_modalidades_actuales)}")
print(f"Rango temporal del dataset delimitado: {rango_temporal(df_modalidades_actuales_periodo_efectivo)}")
print("\nPrincipales modalidades actuales y conteos:")
for modalidad, conteo in principales_modalidades_actuales.items():
    print(f"- {modalidad}: {conteo:,}")
print(f"\nRuta de figuras: {salida_figuras.resolve()}")
print(f"Ruta de tablas: {salida_tablas.resolve()}")
print(f"Ruta de datasets: {salida_datasets.resolve()}")
