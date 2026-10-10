import os
import unicodedata
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import concurrent.futures
import math
from pathlib import Path

BASE_QUERY_URL = "https://seguridadciudadana.mininter.gob.pe/arcgis/rest/services/servicios_ogc/denuncias/MapServer/0/query"

WHERE = "(año_hecho BETWEEN 2020 AND 2024 AND distrito_hecho = 'HUARAL')"

OUT_DIR = Path.home() / "OneDrive" / "10MO CICLO" / "TALLER DE INVESTIGACIÓN" / "data"
OUT_CSV = os.path.join(OUT_DIR, "denuncias_huaral_url1_2020.csv")

# --- PARÁMETROS DE OPTIMIZACIÓN ---
MAX_WORKERS = 1
CHUNK_SIZE = 200
TIMEOUT = 180

def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return s
    s1 = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s1 = s1.replace('Ñ', 'N').replace('ñ', 'n')
    s1 = s1.strip().lower().replace(' ', '_')
    return s1

def json_to_gdf(esri_json):
    features = esri_json.get("features", [])
    if not features:
        return gpd.GeoDataFrame()
    wkid = esri_json.get("spatialReference", {}).get("wkid", 4326)
    xs, ys, attrs = [], [], []
    for f in features:
        g = f.get("geometry") or {}
        x, y = g.get("x"), g.get("y")
        xs.append(x); ys.append(y)
        attrs.append(f.get("attributes", {}))
    df = pd.DataFrame(attrs)
    geom = [Point(x, y) if (x is not None and y is not None) else None for x, y in zip(xs, ys)]
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    gdf = gdf.set_crs(epsg=wkid or 4326, allow_override=True).to_crs(epsg=4326)
    return gdf

def get_all_object_ids(session, where_clause):
    """Obtiene la lista completa de Object IDs para un filtro dado."""
    data = {"where": where_clause, "returnIdsOnly": "true", "f": "json"}
    r = session.post(BASE_QUERY_URL, data=data, timeout=TIMEOUT, verify=False)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"Error de la API al obtener IDs: {j['error']}")
    oids = j.get("objectIds") or j.get("objectIDs") or []
    return sorted(oids)

def _post_geojson(session, ids_str):
    data = {
        "objectIds": ids_str, "outFields": "*", "returnGeometry": "true",
        "outSR": "4326", "f": "geojson"
    }
    r = session.post(BASE_QUERY_URL, data=data, timeout=TIMEOUT, verify=False)
    r.raise_for_status()
    txt = r.text.strip()
    if txt.startswith("<"):
        raise ValueError("Respuesta inesperada (HTML) en lugar de GeoJSON")
    return gpd.read_file(r.text).to_crs(epsg=4326)

def _post_json(session, ids_str):
    data = {
        "objectIds": ids_str, "outFields": "*", "returnGeometry": "true",
        "outSR": "4326", "f": "json"
    }
    r = session.post(BASE_QUERY_URL, data=data, timeout=TIMEOUT, verify=False)
    r.raise_for_status()
    return json_to_gdf(r.json())

def fetch_chunk(session, id_chunk):
    """Descarga un bloque de registros. Esta es la función que ejecuta cada hilo."""
    ids_str = ",".join(map(str, id_chunk))
    try:
        return _post_geojson(session, ids_str)
    except Exception as e:
        return _post_json(session, ids_str)

def fetch_all_parallel():
    """Descarga todos los registros en paralelo."""
    print(" Iniciando descarga paralela optimizada...", flush=True)
    with requests.Session() as session:
        print(f" Paso 1: Obteniendo todos los IDs para el filtro: {WHERE}", flush=True)
        try:
            all_oids = get_all_object_ids(session, WHERE)
            total_records = len(all_oids)
            if total_records == 0:
                print(" No se encontraron registros para el filtro dado.", flush=True)
                return gpd.GeoDataFrame()
            print(f" Se encontraron {total_records} registros en total.", flush=True)
        except Exception as e:
            print(f" Error fatal al obtener los IDs: {e}", flush=True)
            return gpd.GeoDataFrame()

        chunks = [all_oids[i:i + CHUNK_SIZE] for i in range(0, total_records, CHUNK_SIZE)]
        total_chunks = len(chunks)
        print(f" Paso 2: IDs divididos en {total_chunks} lotes de hasta {CHUNK_SIZE} cada uno.", flush=True)

        print(f" Paso 3: Iniciando descarga con hasta {MAX_WORKERS} hilos paralelos...", flush=True)
        all_gdfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk_num = {executor.submit(fetch_chunk, session, chunk): i for i, chunk in enumerate(chunks)}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_chunk_num)):
                chunk_num = future_to_chunk_num[future]
                try:
                    gdf = future.result()
                    if not gdf.empty:
                        all_gdfs.append(gdf)
                    progress = (i + 1) / total_chunks * 100
                    print(f" Progreso: {progress:.2f}% ({i + 1}/{total_chunks} lotes descargados)", flush=True)
                except Exception as exc:
                    print(f' El lote {chunk_num + 1} generó una excepción: {exc}', flush=True)

    if not all_gdfs:
        print(" La descarga finalizó pero no se obtuvo ningún DataFrame.", flush=True)
        return gpd.GeoDataFrame()

    print(f"\n Descarga completada. Consolidando y procesando datos...", flush=True)
    gdf_all = pd.concat(all_gdfs, ignore_index=True).drop_duplicates(subset=['objectid'])

    if gdf_all.crs is None:
        gdf_all = gdf_all.set_crs(epsg=4326, allow_override=True)
    else:
        gdf_all = gdf_all.to_crs(epsg=4326)

    gdf_all = gdf_all.rename(columns={c: normalize_name(c) for c in gdf_all.columns})

    gdf_all["lat"] = gdf_all.geometry.y
    gdf_all["lon"] = gdf_all.geometry.x
    if "lat_hecho" in gdf_all.columns and "long_hecho" in gdf_all.columns:
        lat_num = pd.to_numeric(gdf_all["lat_hecho"], errors="coerce")
        lon_num = pd.to_numeric(gdf_all["long_hecho"], errors="coerce")
        mask = lat_num.notna() & lon_num.notna()
        gdf_all.loc[mask, "lat"] = lat_num[mask]
        gdf_all.loc[mask, "lon"] = lon_num[mask]

    return gdf_all

def main():
    # Desactivar warnings de request no verificado
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

    os.makedirs(OUT_DIR, exist_ok=True)
    gdf = fetch_all_parallel()
    if gdf.empty:
        print(" No se guardó ningún archivo.", flush=True)
        return

    preferred = [
        "fecha_hecho", "hora_hecho", "fecha_hora_hecho",
        "anio_hecho", "mes_hecho", "dia_hecho",
        "departamento_hecho", "provincia_hecho", "distrito_hecho",
        "tipo_hecho", "id_tipo_hecho", "materia_hecho", "id_materia_hecho",
        "lat", "lon", "lat_hecho", "long_hecho"
    ]
    cols = [c for c in preferred if c in gdf.columns] + \
           [c for c in gdf.columns if c not in (preferred + ["geometry"])]

    df_out = pd.DataFrame(gdf[cols]).dropna(subset=["lat", "lon"])
    df_out = df_out[(df_out["lat"].between(-90, 90)) & (df_out["lon"].between(-180, 180))].copy()

    # Convertir y separar fecha y hora
    if 'fecha_hora_hecho' in df_out.columns:
        df_out['fecha_hora_hecho'] = pd.to_datetime(df_out['fecha_hora_hecho'], unit='ms', errors='coerce')

        # Crear solo fecha_hecho y hora_hecho
        df_out['fecha_hecho'] = df_out['fecha_hora_hecho'].dt.strftime('%d/%m/%Y')
        df_out['hora_hecho'] = df_out['fecha_hora_hecho'].dt.strftime('%H:%M:%S')

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n ¡Éxito! Guardado: {OUT_CSV} Filas: {len(df_out)}", flush=True)
    print(df_out.head(10).to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
