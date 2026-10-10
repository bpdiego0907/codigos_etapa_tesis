import pandas as pd
from pathlib import Path

base_dir = Path.cwd()
ruta = base_dir / "data"

if not ruta.exists():
    ruta = base_dir.parent / "data"

df1_huaralprov = pd.read_csv(ruta / "denuncias_huaral_url1.csv", sep = ",")
df2_huaralprov = pd.read_csv(ruta / "denuncias_huaral_url2.csv", sep = ",")

df1 = df1_huaralprov[df1_huaralprov["distrito_hecho"] == "HUARAL"]
df1["fuente"] = "url1"
df2 = df2_huaralprov[df2_huaralprov["distrito_hecho"] == "HUARAL"]
df2["fuente"] = "url2"

# %%
df1.shape

# %%
df2.shape

# %%
df1.columns

# %%
df1.head()

# %%
df1.info()

# %%
df2.columns

# %%
df2.head()

# %%
df2.info()

# %%
df2.columns = df2.columns.str.lower().str.strip()
df2.rename(columns={"fecha_hora_hecho": "fecha_hora_hecho_string"}, inplace=True)
df2.rename(columns={"fecha_hora_hecho_dt": "fecha_hora_hecho"}, inplace=True)
df2.columns

# %%
df1["tipo_hecho"].unique()

# %%
df2["tipo_hecho"].unique()

# %%
columnas_base = [
    "fecha_hora_hecho",
    "departamento_hecho",
    "provincia_hecho",
    "distrito_hecho",
    "materia_hecho",
    "tipo_hecho",
    "subtipo_hecho",
    "modalidad_hecho",
    "lat_hecho",
    "long_hecho",
    "direccion_hecho",
    "turno_hecho",
    "fuente"
]

schema = {
    "fecha_hora_hecho": "datetime",
    "departamento_hecho": "string",
    "provincia_hecho": "string",
    "distrito_hecho": "string",
    "materia_hecho": "string",
    "tipo_hecho": "string",
    "subtipo_hecho": "string",
    "modalidad_hecho": "string",
    "lat_hecho": "float",
    "long_hecho": "float",
    "direccion_hecho": "string",
    "turno_hecho": "string",
    "fuente": "string"
}

# %%
faltantes_df1 = set(columnas_base) - set(df1.columns)
faltantes_df2 = set(columnas_base) - set(df2.columns)

# %%
denuncias_huaral1 = df1[columnas_base].copy()
denuncias_huaral2 = df2[columnas_base].copy()

# %%
for col, tipo in schema.items():
    if tipo == "datetime":
        denuncias_huaral1[col] = pd.to_datetime(denuncias_huaral1[col], errors="coerce")
        denuncias_huaral2[col] = pd.to_datetime(denuncias_huaral2[col], errors="coerce")

    elif tipo == "float":
        denuncias_huaral1[col] = pd.to_numeric(denuncias_huaral1[col], errors="coerce")
        denuncias_huaral2[col] = pd.to_numeric(denuncias_huaral2[col], errors="coerce")

    else:
        denuncias_huaral1[col] = denuncias_huaral1[col].astype("string").str.strip()
        denuncias_huaral2[col] = denuncias_huaral2[col].astype("string").str.strip()

# %%
import unicodedata

cols_texto = [
    "departamento_hecho", "provincia_hecho", "distrito_hecho",
    "materia_hecho", "tipo_hecho", "subtipo_hecho",
    "modalidad_hecho", "direccion_hecho", "turno_hecho"
]

for col in cols_texto:
    denuncias_huaral1[col] = denuncias_huaral1[col].str.upper()
    denuncias_huaral2[col] = denuncias_huaral2[col].str.upper()

def quitar_tildes(texto):
    if pd.isna(texto):
        return texto
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

for col in cols_texto:
    denuncias_huaral1[col] = denuncias_huaral1[col].apply(quitar_tildes)
    denuncias_huaral2[col] = denuncias_huaral2[col].apply(quitar_tildes)

# Eliminar sin coordenadas
denuncias_huaral1 = denuncias_huaral1.dropna(subset=["lat_hecho", "long_hecho"])
denuncias_huaral2 = denuncias_huaral2.dropna(subset=["lat_hecho", "long_hecho"])

# %%
df_total = pd.concat([denuncias_huaral1, denuncias_huaral2], ignore_index=True)

# %%
cols_duplicado = [
    "fecha_hora_hecho",
    "lat_hecho",
    "long_hecho",
    "tipo_hecho",
    "subtipo_hecho",
    "modalidad_hecho"
]

duplicados = df_total.duplicated(subset=cols_duplicado, keep="first")

df_total = df_total.drop_duplicates(subset=cols_duplicado, keep="first")

print(f"Duplicados encontrados: {duplicados.sum()}")

# %%
df_total["año_hecho"] = df_total["fecha_hora_hecho"].dt.year
df_total["mes_hecho"] = df_total["fecha_hora_hecho"].dt.month
df_total["mes_nombre"] = df_total["fecha_hora_hecho"].dt.month_name(locale="es_ES")
df_total["dia_hecho"] = df_total["fecha_hora_hecho"].dt.day
df_total["hora_inicio_hecho"] = df_total["fecha_hora_hecho"].dt.hour

# %%
df_total.año_hecho.unique()

# %%
df_total.head(10)

# %%
df_total.shape

# %%
df_total.columns

# %%
df_total.info()

# %%
df_total.to_csv(ruta / "denuncias_total_huaral.csv", sep = ",")
