"""
Decodifica las teselas .pbf de Amazon Underworld y saca una tabla de municipios.

Uso:
    python decodificador_mvt.py <ruta_corpus> [salida.parquet] [indice.xlsx]
"""

import gzip
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

try:
    import mapbox_vector_tile as mvt
except ImportError:
    sys.exit("Falta la dependencia: pip install mapbox-vector-tile")


GRUPOS = {
    "au_emc":        "Estado Mayor Central (disidencias FARC-EP)",
    "au_embf":       "Estado Mayor de Bloques y Frentes (disidencias FARC-EP)",
    "au_eln":        "Ejército de Liberación Nacional (ELN)",
    "au_c_d_f":      "Clan del Golfo / Autodefensas Gaitanistas de Colombia (AGC)",
    "au_seg_marq":   "Segunda Marquetalia",
    "au_cv":         "Comando Vermelho (CV)",
    "au_pcc":        "Primeiro Comando da Capital (PCC)",
    "au_choneros":   "Los Choneros",
    "au_lobos":      "Los Lobos",
    "au_others":     "Otros grupos armados",
}

# Ojo: el dataset mezcla idiomas. Comparar solo contra VERDADERO pierde Brasil.
VERDADEROS = {"VERDADERO", "VERDADEIRO", "TRUE", "SI", "SIM", "1"}

# Sin prefijo r a proposito: pandas 3 usa RE2, que no entiende \x ni \u.
PATRON_MOJIBAKE = "[ÃÂâ][\x80-\xff]"


def reparar_mojibake(texto):
    if not isinstance(texto, str):
        return texto
    if not re.search(PATRON_MOJIBAKE, texto):
        return texto
    try:
        return texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def normalizar(texto):
    if not isinstance(texto, str):
        return texto
    texto = reparar_mojibake(texto)
    texto = unicodedata.normalize("NFC", texto)
    texto = "".join(c for c in texto if unicodedata.category(c)[0] != "C" or c in "\n\t")
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_popup(texto):
    # Igual que normalizar pero conservando los saltos de linea: los popups los
    # usan para separar grupos, y no siempre ponen guion. Si se colapsan, se
    # fusionan dos grupos en uno y se pierde informacion sin que nada falle.
    if not isinstance(texto, str):
        return texto
    texto = reparar_mojibake(texto)
    texto = unicodedata.normalize("NFC", texto)
    texto = "".join(c for c in texto if unicodedata.category(c)[0] != "C" or c in "\n\t")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\s*\n\s*", "\n", texto)
    return re.sub(r"\n{2,}", "\n", texto).strip()


def es_verdadero(valor):
    if isinstance(valor, bool):
        return valor
    if valor is None:
        return False
    return str(valor).strip().upper() in VERDADEROS


def leer_tesela(ruta):
    crudo = ruta.read_bytes()

    if len(crudo) < 50:  # los 404 de Mapbox pesan 28 bytes
        return None

    if crudo[:2] == b"\x1f\x8b":
        crudo = gzip.decompress(crudo)

    try:
        return mvt.decode(crudo)
    except Exception as exc:
        print(f"  No se pudo decodificar {ruta.name}: {exc}", file=sys.stderr)
        return None


def cargar_inventario(ruta_xlsx):
    # La clave es z/x/nombre y no solo el nombre porque ADL repite nombres:
    # los 73 documentos usan 13 nombres, AMAZONUW_15.pbf esta en 4 carpetas.
    if not ruta_xlsx or not Path(ruta_xlsx).exists():
        return {}
    try:
        df = pd.read_excel(ruta_xlsx, sheet_name="Inventario de Archivos")
    except Exception as exc:
        print(f"No se pudo leer el inventario {ruta_xlsx}: {exc}", file=sys.stderr)
        return {}

    df = df[df["Observatorio"].astype(str) == "Amazon_Underworld"]
    mapa = {}
    for _, r in df.iterrows():
        nombre = str(r["Nombre estandarizado"]).strip()
        if not nombre.lower().endswith(".pbf"):
            continue
        carpeta = str(r["Carpeta"]).replace("\\", "/").strip("/")
        partes = carpeta.split("/")
        m = re.search(r"(\d+)", nombre)
        if len(partes) < 2 or not m:
            continue
        try:
            z, x, y = int(partes[-2]), int(partes[-1]), int(m.group(1))
        except ValueError:
            continue
        mapa[f"{z}/{x}/{nombre}"] = {
            "zoom": z, "tile_x": x, "tile_y": y,
            "doc_id": str(r["DOC_ID"]).strip(),
            "fuente": f"{carpeta}/{nombre}",
        }
    print(f"Inventario ADL: {len(mapa)} teselas registradas")
    return mapa


def identificar_tesela(ruta, inventario):
    nombre = ruta.name
    padres = ruta.parent.parts

    clave = None
    if len(padres) >= 2:
        clave = f"{padres[-2]}/{padres[-1]}/{nombre}"
    if clave and clave in inventario:
        d = inventario[clave]
        return d["zoom"], d["tile_x"], d["tile_y"], d["doc_id"], d["fuente"]

    # Sin inventario: z y x de las carpetas, y del numero del nombre.
    if len(padres) < 2:
        return None
    try:
        z, x = int(padres[-2]), int(padres[-1])
    except ValueError:
        return None
    m = re.search(r"(\d+)", ruta.stem)
    if not m:
        return None
    y = int(m.group(1))
    return z, x, y, f"F3-AMAZONUW-{z}-{x}-{y}", f"tiles/{z}/{x}/{nombre}"


def coords_tesela(z, x, y):
    import math
    n = 2.0 ** z
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n))))
    return round(lat, 5), round(lon, 5)


def extraer_features(capas, z, x, y, ruta_rel, doc_id, fuente):
    filas = []
    lat, lon = coords_tesela(z, x, y)

    for nombre_capa, capa in capas.items():
        for feature in capa.get("features", []):
            props = feature.get("properties", {})

            # Los que no traen este campo son poligonos de frontera sin datos.
            id_au = props.get("au_ID_concatenated")
            if not id_au:
                continue

            grupos_presentes = [
                nombre for campo, nombre in GRUPOS.items()
                if es_verdadero(props.get(campo))
            ]

            filas.append({
                "id_municipio":   normalizar(id_au),
                "pais":           normalizar(props.get("au_country")),
                "nivel1":         normalizar(props.get("au_level1") or props.get("b_ADM1_ES")),
                "nivel2":         normalizar(props.get("au_level2") or props.get("b_ADM2_ES")),
                "nivel1_pt":      normalizar(props.get("b_ADM1_PT")),
                "nivel2_pt":      normalizar(props.get("b_ADM2_PT")),
                "pcode_adm1":     props.get("b_ADM1_PCODE"),
                "pcode_adm2":     props.get("b_ADM2_PCODE"),
                "poblacion":      props.get("au_population"),
                "area_km2":       props.get("au_area km"),
                "grupos":         grupos_presentes,
                "n_grupos":       len(grupos_presentes),
                "sin_informacion": es_verdadero(props.get("au_no_info")),
                "investigado":    es_verdadero(props.get("au_invest. with presence")),
                "popup_es":       normalizar_popup(props.get("au_popup_window_es")),
                "popup_en":       normalizar_popup(props.get("au_popup_window_en")),
                "popup_pt":       normalizar_popup(props.get("au_popup_window_pt")),
                "geometria":      feature.get("geometry", {}).get("type"),
                "zoom":           z,
                "tile_x":         x,
                "tile_y":         y,
                "capa":           nombre_capa,
                "doc_id":         doc_id,
                "fuente":         fuente,
                "ruta_local":     ruta_rel,
                "lat_tesela":     lat,
                "lon_tesela":     lon,
            })

    return filas


def recorrer_teselas(carpeta_tiles, inventario=None):
    inventario = inventario or {}
    todas = []
    archivos = sorted(Path(carpeta_tiles).rglob("*.pbf"))
    print(f"Teselas encontradas: {len(archivos)}")

    ok, sin_ubicar, vacias = 0, 0, 0
    for ruta in archivos:
        ident = identificar_tesela(ruta, inventario)
        if ident is None:
            print(f"  No se pudo ubicar, se omite: {ruta}", file=sys.stderr)
            sin_ubicar += 1
            continue
        z, x, y, doc_id, fuente = ident

        capas = leer_tesela(ruta)
        if capas is None:
            vacias += 1
            continue

        filas = extraer_features(capas, z, x, y, str(ruta), doc_id, fuente)
        todas.extend(filas)
        ok += 1

    print(f"Teselas decodificadas correctamente: {ok}")
    if sin_ubicar:
        print(f"  {sin_ubicar} teselas sin z/x/y resoluble, revisa el inventario")
    if vacias:
        print(f"  {vacias} teselas vacias o ilegibles")
    print(f"Apariciones de municipio, con duplicados entre zooms: {len(todas)}")
    return pd.DataFrame(todas)


def deduplicar(df):
    # Nos quedamos con el zoom mas bajo. Esas teselas son pocas y panoramicas,
    # asi que los municipios quedan concentrados en menos archivos fuente.
    # fuentes_todas guarda el resto por si hay que revertir sin reprocesar.
    if df.empty:
        return df

    antes = len(df)

    fuentes = (df.groupby("id_municipio")["fuente"]
                 .apply(lambda s: sorted(set(s)))
                 .rename("fuentes_todas"))

    df = df.sort_values(["id_municipio", "zoom", "tile_x", "tile_y"])
    df = df.drop_duplicates(subset="id_municipio", keep="first")
    df = df.merge(fuentes, on="id_municipio", how="left")
    df["n_apariciones"] = df["fuentes_todas"].str.len()

    print(f"Deduplicacion: {antes} apariciones, quedan {len(df)} municipios unicos")
    return df.reset_index(drop=True)


def verificar(df):
    problemas = []

    if df["id_municipio"].duplicated().any():
        problemas.append("Hay id_municipio duplicados tras la deduplicacion")

    sin_nombre = df["nivel2"].isna().sum()
    if sin_nombre:
        problemas.append(f"{sin_nombre} municipios sin nombre en nivel2")

    mojibake = df["nivel2"].fillna("").str.contains(PATRON_MOJIBAKE).sum()
    if mojibake:
        problemas.append(f"{mojibake} nombres con mojibake sin reparar")

    print("\nVerificacion")
    if problemas:
        for p in problemas:
            print(f"  {p}")
    else:
        print("  Todo correcto")

    print(f"\nMunicipios: {len(df)}")
    print(f"Paises: {df['pais'].value_counts().to_dict()}")
    print(f"Con al menos un grupo armado: {(df['n_grupos'] > 0).sum()}")
    poblacion = pd.to_numeric(df["poblacion"], errors="coerce").sum()
    print(f"Poblacion total cubierta: {poblacion:,.0f}")
    print(f"Documentos referenciados: {df['doc_id'].nunique()}")
    return not problemas


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    base = Path(sys.argv[1])
    carpeta_tiles = base / "tiles" if (base / "tiles").is_dir() else base
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("municipios_amazonia.parquet")

    ruta_inv = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    if ruta_inv is None:
        for cand in (base / "Indice_Datos_Codefest.xlsx",
                     Path("Indice_Datos_Codefest.xlsx")):
            if cand.exists():
                ruta_inv = cand
                break
    inventario = cargar_inventario(ruta_inv)
    if not inventario:
        # Sin inventario los doc_id salen de la ruta y ya no coinciden con los
        # de ADL, que es lo que se compara al corregir.
        print("Sin inventario de ADL: doc_id y fuente se derivaran de la ruta")

    df = recorrer_teselas(carpeta_tiles, inventario)
    if df.empty:
        sys.exit("No se extrajo ningun municipio. Revisa la ruta.")

    df = deduplicar(df)
    verificar(df)

    try:
        df.to_parquet(salida, index=False)
        print(f"\nGuardado en {salida}")
    except ImportError:
        salida = salida.with_suffix(".jsonl")
        df.to_json(salida, orient="records", lines=True, force_ascii=False)
        print(f"\npyarrow no disponible, guardado en {salida}")

    df.head(50).to_csv(salida.with_suffix(".muestra.csv"), index=False)


if __name__ == "__main__":
    main()
