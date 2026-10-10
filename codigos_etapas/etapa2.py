import os
import time
import requests
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIGURACIÓN
# =========================================================
BASE = "https://services6.arcgis.com/lMIZrqiJkpM748BR/arcgis/rest/services"

CAPAS = {
    "ESTAFA": ("ESTAFA_IDSUBTIPO_10506", 4),
    "EXTORSION": ("EXTORSION_IDSUBTIPO_10508", 1),
    "HOMICIDIO": ("HOMICIDIO_IDMODALIDAD_1010102_AL_1010130", 2),
    "HURTO": ("HURTO_IDSUBTIPO_10501", 2),
    "MICROCOMERCIALIZACION": ("MICROCOMERCIALIZACION_IDMODALIDAD_1120341_5070112_5070122", 1),
    "PANDILLAJE": ("PANDILLAJE_IDSUBTIPO_40101", 6),
    "ROBO": ("ROBO_IDSUBTIPO_10502", 3),
    "VCM": ("VCM_IDTIPO_504", 0),
}

WHERE = "(año_hecho = 2025 AND provincia_hecho = 'HUARAL')"

OUT_DIR = Path.home() / "OneDrive" / "10MO CICLO" / "TALLER DE INVESTIGACIÓN" / "data"
OUT_CSV_TOTAL = os.path.join(OUT_DIR, "denuncias_huaral_2024.csv")

PAGE_SIZE = 2000
TIMEOUT = 60
PAUSA = 0.2

# =========================================================
# FUNCIONES
# =========================================================
def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    else:
        df["lat"] = pd.NA

    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    else:
        df["lon"] = pd.NA

    if "lat_hecho" in df.columns:
        df["lat_hecho_num"] = pd.to_numeric(df["lat_hecho"], errors="coerce")
        mask = df["lat"].isna() & df["lat_hecho_num"].notna()
        df.loc[mask, "lat"] = df.loc[mask, "lat_hecho_num"]

    if "long_hecho" in df.columns:
        df["long_hecho_num"] = pd.to_numeric(df["long_hecho"], errors="coerce")
        mask = df["lon"].isna() & df["long_hecho_num"].notna()
        df.loc[mask, "lon"] = df.loc[mask, "long_hecho_num"]

    if "fecha_hora_hecho" in df.columns:
        serie = df["fecha_hora_hecho"]

        if pd.api.types.is_numeric_dtype(serie):
            df["fecha_hora_hecho_dt"] = pd.to_datetime(serie, unit="ms", errors="coerce")
        else:
            num = pd.to_numeric(serie, errors="coerce")
            if num.notna().sum() > 0:
                df["fecha_hora_hecho_dt"] = pd.to_datetime(num, unit="ms", errors="coerce")
            else:
                df["fecha_hora_hecho_dt"] = pd.to_datetime(serie, errors="coerce")

        df["fecha_hecho_fmt"] = df["fecha_hora_hecho_dt"].dt.strftime("%d/%m/%Y")
        df["hora_hecho_fmt"] = df["fecha_hora_hecho_dt"].dt.strftime("%H:%M:%S")

    return df

def get_query_url(servicio: str, layer_id: int) -> str:
    return f"{BASE}/{servicio}/FeatureServer/{layer_id}/query"

def get_total_count(query_url: str, where: str) -> int:
    data = {
        "where": where,
        "returnCountOnly": "true",
        "f": "json"
    }

    r = requests.post(query_url, data=data, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()

    if "error" in j:
        raise RuntimeError(f"Error al obtener conteo: {j['error']}")

    return j.get("count", 0)

def fetch_page(query_url: str, where: str, offset: int, page_size: int) -> list[dict]:
    data = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": page_size,
        "orderByFields": "OBJECTID ASC"
    }

    r = requests.post(query_url, data=data, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()

    if "error" in j:
        raise RuntimeError(f"Error en offset {offset}: {j['error']}")

    rows = []
    for feat in j.get("features", []):
        attrs = feat.get("attributes", {}) or {}
        geom = feat.get("geometry", {}) or {}

        attrs["lon"] = geom.get("x")
        attrs["lat"] = geom.get("y")

        rows.append(attrs)

    return rows

def descargar_capa(nombre_categoria: str, servicio: str, layer_id: int, where: str) -> tuple[pd.DataFrame, dict]:
    query_url = get_query_url(servicio, layer_id)

    print(f"\nDescargando: {nombre_categoria}")
    print(f"Servicio: {servicio} | Layer: {layer_id}")

    total = get_total_count(query_url, where)
    print(f"Total esperado: {total}")

    if total == 0:
        return pd.DataFrame(), {
            "categoria": nombre_categoria,
            "servicio": servicio,
            "layer_id": layer_id,
            "total_esperado": 0,
            "total_descargado": 0,
            "estado": "sin registros"
        }

    all_rows = []
    offset = 0
    pagina = 1

    while True:
        rows = fetch_page(query_url, where, offset, PAGE_SIZE)

        if not rows:
            print("No hay más registros.")
            break

        all_rows.extend(rows)
        print(f"Página {pagina}: +{len(rows)} | acumulado={len(all_rows)}")

        if len(rows) < PAGE_SIZE:
            print("Última página detectada.")
            break

        offset += PAGE_SIZE
        pagina += 1
        time.sleep(PAUSA)

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df["categoria_fuente"] = nombre_categoria
        df["servicio_fuente"] = servicio
        df["layer_id_fuente"] = layer_id

    return df, {
        "categoria": nombre_categoria,
        "servicio": servicio,
        "layer_id": layer_id,
        "total_esperado": total,
        "total_descargado": len(df),
        "estado": "ok" if len(df) > 0 else "vacio"
    }

# =========================================================
# MAIN
# =========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dfs = []
    resumenes = []

    for categoria, (servicio, layer_id) in CAPAS.items():
        try:
            df_cat, resumen = descargar_capa(categoria, servicio, layer_id, WHERE)
            resumenes.append(resumen)

            if not df_cat.empty:
                df_cat = normalizar_columnas(df_cat)
                dfs.append(df_cat)

        except Exception as e:
            print(f"Error en {categoria}: {e}")
            resumenes.append({
                "categoria": categoria,
                "servicio": servicio,
                "layer_id": layer_id,
                "total_esperado": None,
                "total_descargado": 0,
                "estado": f"error: {e}"
            })

    if not dfs:
        print("\nNo se descargó ninguna capa con datos.")
        return

    print("\nConsolidando todo...")
    df_total = pd.concat(dfs, ignore_index=True)

    # Deduplicación tentativa
    if "ID_DGC_03" in df_total.columns:
        df_total = df_total.drop_duplicates(subset=["ID_DGC_03", "categoria_fuente"])
    elif "OBJECTID" in df_total.columns:
        df_total = df_total.drop_duplicates(subset=["OBJECTID", "categoria_fuente"])

    # Reordenar columnas
    preferidas = [
        "categoria_fuente",
        "ID_DGC_03",
        "OBJECTID",
        "fecha_hora_hecho",
        "fecha_hora_hecho_dt",
        "fecha_hecho_fmt",
        "hora_hecho_fmt",
        "año_hecho",
        "mes_hecho",
        "dia_hecho",
        "departamento_hecho",
        "provincia_hecho",
        "distrito_hecho",
        "materia_hecho",
        "tipo_hecho",
        "subtipo_hecho",
        "modalidad_hecho",
        "lat",
        "lon",
        "lat_hecho",
        "long_hecho",
        "servicio_fuente",
        "layer_id_fuente",
    ]

    cols = [c for c in preferidas if c in df_total.columns] + \
           [c for c in df_total.columns if c not in preferidas]

    df_total = df_total[cols]

    df_total.to_csv(OUT_CSV_TOTAL, index=False, encoding="utf-8-sig")

    print(f"\nListo.")
    print(f"CSV consolidado: {OUT_CSV_TOTAL}")
    print(f"Total filas consolidadas: {len(df_total)}")

if __name__ == "__main__":
    main()
