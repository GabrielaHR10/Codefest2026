"""
Arma un chunk de texto por municipio a partir de la tabla del decodificador.

Uso:
    python generador_chunks.py municipios.parquet [inventario.xlsx]
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


FENOMENO = 3
FORMATO = "pbf"
LIMITE_PALABRAS = 250

# Las 50 consultas que entrego ADL estan todas en espanol, aunque el documento
# del reto prometia tres idiomas. Con estos en False los chunks quedan solo en
# espanol y bajan de 86 a 65 palabras. Medir contra el dev set antes de cambiar.
ANCLAJE_EN = True
APERTURA_PT = True

# Tiene que ser el mismo encoder que use el equipo, num_tokens depende de el.
MODELO_ENCODER = "BAAI/bge-m3"

_TOKENIZER = None
_AVISO_TOKENIZER = False


def obtener_tokenizer():
    global _TOKENIZER, _AVISO_TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    if _AVISO_TOKENIZER:
        return None
    try:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(MODELO_ENCODER)
        print(f"Tokenizer cargado: {MODELO_ENCODER}")
        return _TOKENIZER
    except Exception as exc:
        _AVISO_TOKENIZER = True
        print(f"No se pudo cargar el tokenizer de {MODELO_ENCODER}: {exc}")
        print("num_tokens se contara por palabras. Hay que regenerarlo con el "
              "tokenizer real antes de la entrega.")
        return None


def contar_tokens(texto):
    tok = obtener_tokenizer()
    if tok is None:
        return len(texto.split())
    return len(tok.encode(texto, add_special_tokens=False))


# Revisadas a mano una por una. La mayoria de los nombres que marca el
# verificador no llevan tilde de verdad, asi que aqui solo van los confirmados:
# meter una de mas corrompe el nombre en vez de arreglarlo.
CORRECCIONES = {
    "Bolivar": "Bolívar",
    "Vaupes": "Vaupés",
    "Caqueta": "Caquetá",
    "Guainia": "Guainía",
    "Cordoba": "Córdoba",
    "Bogota": "Bogotá",
    "Boyaca": "Boyacá",
    "Atlantico": "Atlántico",
    "Narino": "Nariño",
    "Sucumbios": "Sucumbíos",
    "La Joya De Los Sachas": "La Joya de los Sachas",
    "Peru": "Perú",
    "San Martin": "San Martín",
    "Junin": "Junín",
    "Huanuco": "Huánuco",
    "Rondonia": "Rondônia",
    "Para": "Pará",
    "Maranhao": "Maranhão",
    "Piaui": "Piauí",
    "Ceara": "Ceará",
    "Goias": "Goiás",
    "Potosi": "Potosí",
}

PAIS_EN = {
    "Colombia": "Colombia",
    "Brasil": "Brazil",
    "Venezuela": "Venezuela",
    "Ecuador": "Ecuador",
    "Peru": "Peru",
    "Perú": "Peru",
    "Bolivia": "Bolivia",
}

REGION_ES = {
    "Colombia": "la Amazonía colombiana",
    "Brasil": "la Amazonía brasileña",
    "Venezuela": "la región amazónica venezolana",
    "Ecuador": "la Amazonía ecuatoriana",
    "Peru": "la Amazonía peruana",
    "Perú": "la Amazonía peruana",
    "Bolivia": "la Amazonía boliviana",
}

REGION_EN = {
    "Colombia": "the Colombian Amazon",
    "Brasil": "the Brazilian Amazon",
    "Venezuela": "the Venezuelan Amazon",
    "Ecuador": "the Ecuadorian Amazon",
    "Peru": "the Peruvian Amazon",
    "Perú": "the Peruvian Amazon",
    "Bolivia": "the Bolivian Amazon",
}

# Lleva el articulo dentro para que concuerde en genero: 'del departamento de'
# pero 'de la provincia de'.
DIVISION_ES = {
    "Colombia": "del departamento de",
    "Brasil":   "del estado de",
    "Venezuela": "del estado",
    "Ecuador":  "de la provincia de",
    "Peru":     "del departamento de",
    "Perú":     "del departamento de",
    "Bolivia":  "del departamento de",
}

# Estos dos estados no admiten articulo en portugues: 'de Roraima', no 'do'.
PT_SIN_ARTICULO = {"Roraima", "Rondônia", "Rondonia"}

DIVISION_EN = {
    "Colombia": "Department",
    "Brasil": "State",
    "Venezuela": "State",
    "Ecuador": "Province",
    "Peru": "Department",
    "Perú": "Department",
    "Bolivia": "Department",
}


def sin_tildes(texto):
    if not isinstance(texto, str):
        return ""
    descompuesto = unicodedata.normalize("NFD", texto)
    return unicodedata.normalize(
        "NFC", "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")
    )


def nombre_con_variantes(nombre):
    # Devuelve 'Bolívar (Bolivar)'. Las consultas suelen venir sin tildes y
    # para BM25 son palabras distintas, asi que interesa que esten las dos.
    s = texto_o_vacio(nombre)
    if not s:
        return ""
    corregido = CORRECCIONES.get(s, s)
    plano = sin_tildes(corregido)
    if plano != corregido:
        return f"{corregido} ({plano})"
    return corregido


def limpio(nombre):
    s = texto_o_vacio(nombre)
    return CORRECCIONES.get(s, s) if s else ""


def numero_es(valor):
    try:
        return f"{int(valor):,}".replace(",", ".")
    except (TypeError, ValueError):
        return None


def contar_palabras(texto):
    return len(texto.split())


def como_lista(valor):
    # Una columna de listas vuelve de parquet como ndarray, y ahi 'valor or []'
    # lanza ValueError en vez de dar lista vacia.
    if valor is None:
        return []
    if isinstance(valor, str):
        s = valor.strip()
        if not s or s.lower() == "nan":
            return []
        if s.startswith("["):
            try:
                return [str(v) for v in json.loads(s)]
            except json.JSONDecodeError:
                return [s]
        return [s]
    try:
        return [str(v) for v in valor if v is not None and str(v).strip()]
    except TypeError:
        return []


def texto_o_vacio(valor):
    if valor is None or not isinstance(valor, str):
        return ""
    s = valor.strip()
    return "" if s.lower() == "nan" else s


def parsear_popup(popup):
    crudo = texto_o_vacio(popup)
    if not crudo:
        return []

    # Cortar primero por salto de linea: hay items que no llevan guion propio y
    # cortar solo por ' - ' fusionaria dos grupos en una sola cadena.
    partes = []
    for linea in crudo.split("\n"):
        linea = linea.strip().lstrip("-").strip()
        if linea:
            partes.extend(p.strip() for p in re.split(r"\s+-\s+", linea) if p.strip())

    resultados = []
    for parte in partes:
        # 'EMBF: Frente Rodrigo Cadete' pasa a 'Frente Rodrigo Cadete (EMBF)',
        # pero solo si lo de la izquierda parece una sigla de verdad.
        sigla, _, nombre = parte.partition(":")
        sigla, nombre = sigla.strip(), nombre.strip()
        es_sigla = nombre and len(sigla) <= 12 and len(sigla.split()) <= 2
        if es_sigla:
            resultados.append(f"{nombre} ({sigla})")
        elif parte:
            resultados.append(parte)

    vistos, unicos = set(), []
    for r in resultados:
        if r.lower() not in vistos:
            vistos.add(r.lower())
            unicos.append(r)
    return unicos


def enumerar(items, conector="y"):
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {conector} {items[-1]}"


def _nucleo(texto):
    base = re.sub(r"\([^)]*\)", "", texto)
    return re.sub(r"[^a-z0-9 ]", "", sin_tildes(base).lower()).strip()


def resolver_grupos(fila):
    # Unica fuente de verdad sobre que grupos tiene un municipio. La usan la
    # plantilla y el campo n_grupos, para que no puedan contradecirse.
    grupos = como_lista(fila.get("grupos"))
    facciones = parsear_popup(fila.get("popup_es"))

    # 'Otros grupos armados' no dice nada. Se quita si el popup nombra grupos
    # concretos, pero se deja si no hay popup: ahi es la unica senal.
    GENERICO = "Otros grupos armados"
    if GENERICO in grupos and facciones:
        grupos = [g for g in grupos if g != GENERICO]

    ya_listados = {_nucleo(g) for g in grupos}
    facciones_nuevas = [f for f in facciones if _nucleo(f) not in ya_listados]
    return grupos, facciones_nuevas


def construir_texto(fila):
    # Todo con plantillas y concatenacion, sin ningun modelo generativo.
    pais = limpio(fila.get("pais")) or "país no identificado"
    muni = nombre_con_variantes(fila.get("nivel2"))
    depto = nombre_con_variantes(fila.get("nivel1"))
    muni_simple = limpio(fila.get("nivel2"))
    depto_simple = limpio(fila.get("nivel1"))

    division = DIVISION_ES.get(pais, "del departamento de")
    region = REGION_ES.get(pais, "la cuenca amazónica")

    oraciones = []

    muni_pt = limpio(fila.get("nivel2_pt"))
    depto_pt = limpio(fila.get("nivel1_pt"))
    apertura_pt = (APERTURA_PT and pais == "Brasil"
                   and bool(muni_pt) and bool(depto_pt))

    if apertura_pt:
        prep = "de" if depto_pt in PT_SIN_ARTICULO else "do"
        if muni_pt == muni_simple and depto_pt == depto_simple:
            oraciones.append(f"{muni} é um município do estado {prep} {depto}, Brasil")
        else:
            oraciones.append(
                f"{muni_pt} é um município do estado {prep} {depto_pt}, Brasil "
                f"— en español, {muni}, {division} {depto} —"
            )
    else:
        oraciones.append(f"{muni} es un municipio {division} {depto}, {pais}")

    # Si la apertura va en portugues, su cola tambien: cerrar una oracion
    # portuguesa en espanol queda mal en los dos idiomas.
    poblacion = numero_es(fila.get("poblacion"))
    area = fila.get("area_km2")
    detalles = []
    if apertura_pt:
        if poblacion:
            detalles.append(f"uma população de {poblacion} habitantes")
        if isinstance(area, (int, float)) and area == area:
            detalles.append(f"uma extensão de {numero_es(round(area))} km²")
        cola = (f", com {enumerar(detalles, 'e')} na Amazônia brasileira."
                if detalles else ", na Amazônia brasileira.")
    else:
        if poblacion:
            detalles.append(f"una población de {poblacion} habitantes")
        if isinstance(area, (int, float)) and area == area:
            detalles.append(f"una extensión de {numero_es(round(area))} km²")
        cola = (f", con {enumerar(detalles)} en {region}."
                if detalles else f", en {region}.")
    oraciones[0] += cola

    grupos, facciones_nuevas = resolver_grupos(fila)
    hay_presencia = bool(grupos or facciones_nuevas)

    # Con apertura en portugues la region no se ha nombrado aun en espanol.
    donde = f"en este municipio de {region}" if apertura_pt else "en este territorio"

    if grupos:
        cantidad = {1: "una estructura armada", 2: "dos estructuras armadas",
                    3: "tres estructuras armadas"}.get(
            len(grupos), f"{len(grupos)} estructuras armadas")
        oraciones.append(
            f"Según el observatorio Amazon Underworld, {donde} se "
            f"documenta la presencia de {cantidad}: {enumerar(grupos)}."
        )

    if facciones_nuevas:
        # 'También' presupone algo antes; si no hubo banderas hay que enunciar
        # la frase entera.
        if grupos:
            etiqueta = ("También se identifica la estructura local"
                        if len(facciones_nuevas) == 1
                        else "También se identifican las estructuras locales")
        else:
            etiqueta = (f"Según el observatorio Amazon Underworld, {donde} se "
                        f"documenta la presencia de la estructura local"
                        if len(facciones_nuevas) == 1 else
                        f"Según el observatorio Amazon Underworld, {donde} se "
                        f"documenta la presencia de las estructuras locales")
        oraciones.append(f"{etiqueta} {enumerar(facciones_nuevas)}.")

    # Solo si no hay nada de nada. Mirando solo las banderas salian chunks que
    # nombraban un grupo y a la vez negaban tener informacion.
    if not grupos and not facciones_nuevas:
        ubicacion_es = f"en este municipio de {region}" if apertura_pt else "en este municipio"
        oraciones.append(
            f"El observatorio Amazon Underworld no registra información "
            f"verificada sobre presencia de grupos armados {ubicacion_es}."
        )
        oraciones.append(
            "La ausencia de registro no equivale a ausencia de actividad: "
            "refleja los límites de cobertura del observatorio."
        )

    if fila.get("investigado"):
        oraciones.append(
            "El municipio fue objeto de investigación directa en terreno "
            "por parte del observatorio."
        )
    elif hay_presencia:
        oraciones.append(
            "El municipio no fue objeto de investigación directa en terreno."
        )

    pais_en = PAIS_EN.get(pais, pais)
    region_en = REGION_EN.get(pais, "the Amazon basin")
    division_en = DIVISION_EN.get(pais, "Department")

    ubicacion_en = f"{muni_simple}, {depto_simple} {division_en}, {pais_en}"
    if not ANCLAJE_EN:
        return " ".join(oraciones)
    if hay_presencia:
        oraciones.append(
            f"Armed group presence in {ubicacion_en}, in {region_en}, "
            f"is documented by the Amazon Underworld observatory."
        )
    else:
        oraciones.append(
            f"No verified information on armed group presence is available "
            f"for {ubicacion_en}, in {region_en}."
        )

    return " ".join(oraciones)


# ADL repite nombres de archivo: los 73 documentos usan solo 13 nombres, asi
# que el nombre suelto no identifica nada. En False, esos 73 se colapsan en 13.
FUENTE_CON_RUTA = True


def cargar_inventario(ruta):
    # Solo hace falta para parquets viejos; ahora doc_id y fuente ya vienen
    # resueltos desde el decodificador.
    if not ruta or not Path(ruta).exists():
        return {}
    df = pd.read_excel(ruta, sheet_name="Inventario de Archivos")
    df = df[df["Observatorio"].astype(str) == "Amazon_Underworld"]
    mapa = {}
    for _, r in df.iterrows():
        carpeta = str(r["Carpeta"]).replace("\\", "/").strip("/")
        nombre = str(r["Nombre estandarizado"]).strip()
        if not nombre.lower().endswith(".pbf"):
            continue
        m = re.search(r"(\d+)", nombre)
        partes = carpeta.split("/")
        if len(partes) < 2 or not m:
            continue
        clave = f"{partes[-2]}/{partes[-1]}/{int(m.group(1))}"
        fuente = f"{carpeta}/{nombre}" if FUENTE_CON_RUTA else nombre
        mapa[clave] = (str(r["DOC_ID"]).strip(), fuente)
    return mapa


def clave_zxy(fila):
    return f"{fila['zoom']}/{fila['tile_x']}/{fila['tile_y']}"


def clave_zxy_serie(df):
    return (df["zoom"].astype(str) + "/" + df["tile_x"].astype(str)
            + "/" + df["tile_y"].astype(str))


def generar(df, inventario=None):
    mapa = cargar_inventario(inventario) if inventario else {}

    # posicion es el orden del chunk dentro de su documento y empieza en 0.
    df = df.sort_values(["zoom", "tile_x", "tile_y", "id_municipio"]).reset_index(drop=True)
    df["posicion"] = df.groupby(clave_zxy_serie(df)).cumcount()

    registros = []
    for _, fila in df.iterrows():
        clave = clave_zxy(fila)
        doc_id = texto_o_vacio(fila.get("doc_id"))
        fuente = texto_o_vacio(fila.get("fuente"))
        if not (doc_id and fuente):
            doc_id, fuente = mapa.get(
                clave, (f"F3-AMAZONUW-{clave.replace('/', '-')}",
                        f"tiles/{clave.rsplit('/', 1)[0]}/AMAZONUW_{fila['tile_y']}.pbf"))
        if not FUENTE_CON_RUTA:
            fuente = fuente.rsplit("/", 1)[-1]
        texto = construir_texto(fila)
        grupos_fila, facciones_fila = resolver_grupos(fila)

        registros.append({
            "doc_id":     str(doc_id),
            "chunk_id":   f"{doc_id}-chunk-{int(fila['posicion']):04d}",
            "fuente":     fuente,
            "formato":    FORMATO,
            "fenomeno":   FENOMENO,
            "posicion":   int(fila["posicion"]),
            "num_tokens": contar_tokens(texto),
            "texto":      texto,
            "id_municipio":  fila["id_municipio"],
            "pais":          limpio(fila.get("pais")),
            "nivel1":        limpio(fila.get("nivel1")),
            "nivel2":        limpio(fila.get("nivel2")),
            "n_grupos":      len(grupos_fila),
            "n_facciones":   len(facciones_fila),
            "idiomas":       ["es", "en", "pt"] if fila.get("pais") == "Brasil" else ["es", "en"],
            "zoom":          int(fila["zoom"]),
            "n_palabras":    contar_palabras(texto),
            "observatorio":  "Amazon_Underworld",
            "nivel":         "municipio",
            "encoder_tokens": MODELO_ENCODER if _TOKENIZER else "conteo_palabras",
        })
    return registros


def validar(registros):
    problemas = []

    obligatorios = ["doc_id", "chunk_id", "fuente", "formato",
                    "fenomeno", "posicion", "num_tokens", "texto"]

    for r in registros:
        faltantes = [c for c in obligatorios if r.get(c) in (None, "")]
        if faltantes:
            problemas.append(f"{r.get('chunk_id')}: faltan campos {faltantes}")

        if r["n_palabras"] > LIMITE_PALABRAS:
            problemas.append(
                f"{r['chunk_id']}: {r['n_palabras']} palabras, limite {LIMITE_PALABRAS}")

        texto = r["texto"].strip()
        if not texto.endswith((".", "!", "?")):
            problemas.append(f"{r['chunk_id']}: no termina en punto")

        if "None" in texto or "nan" in texto.split():
            problemas.append(f"{r['chunk_id']}: contiene valores sin resolver")

    ids = [r["chunk_id"] for r in registros]
    if len(ids) != len(set(ids)):
        problemas.append("Hay chunk_id duplicados")

    print("\nValidacion")
    if not registros:
        print("  No se genero ningun chunk. Revisa la entrada.")
        return False

    if problemas:
        for p in problemas[:20]:
            print(f"  {p}")
        if len(problemas) > 20:
            print(f"  y {len(problemas) - 20} mas")
    else:
        print("  Todos los chunks cumplen los requisitos")

    palabras = [r["n_palabras"] for r in registros]
    print(f"\nChunks generados: {len(registros)}")
    print(f"Palabras por chunk: min {min(palabras)}, "
          f"media {sum(palabras) / len(palabras):.0f}, max {max(palabras)}")
    return not problemas


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    entrada = Path(sys.argv[1])
    inventario = sys.argv[2] if len(sys.argv) > 2 else None

    if entrada.suffix == ".parquet":
        df = pd.read_parquet(entrada)
    else:
        df = pd.read_json(entrada, lines=True)

    registros = generar(df, inventario)
    validar(registros)

    salida = Path("metadata_mapas.jsonl")
    with open(salida, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nGuardado en {salida}")

    print("\nEjemplos")
    for r in registros[:2]:
        print(f"\n{r['chunk_id']} ({r['n_palabras']} palabras)")
        print(r["texto"])


if __name__ == "__main__":
    main()
