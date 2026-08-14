"""
Revisa que los chunks de mapas cumplan lo que pide el reto.

Sale con codigo 1 si encuentra fallos, asi sirve para CI.

Uso:
    python verificar_mapas.py municipios.parquet metadata_mapas.jsonl
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


OBLIGATORIOS = ["doc_id", "chunk_id", "fuente", "formato",
                "fenomeno", "posicion", "num_tokens", "texto"]

LIMITE_PALABRAS = 250

ESPERADO_TESELAS = 73
ESPERADO_MUNICIPIOS = 985
PAISES_ESPERADOS = {"Colombia", "Brasil", "Venezuela", "Ecuador", "Peru", "Perú", "Bolivia"}

PATRON_MOJIBAKE = "[ÃÂâ][\x80-\xff]"

fallos = []
avisos = []


def error(msg):
    fallos.append(msg)
    print(f"  FALLO: {msg}")


def aviso(msg):
    avisos.append(msg)
    print(f"  Aviso: {msg}")


def ok(msg):
    print(f"  OK: {msg}")


def sin_tildes(t):
    d = unicodedata.normalize("NFD", str(t))
    return unicodedata.normalize("NFC", "".join(c for c in d
                                                if unicodedata.category(c) != "Mn"))


def verificar_decodificacion(df):
    print("\n1. Decodificacion y deduplicacion")

    n = len(df)
    desvio = abs(n - ESPERADO_MUNICIPIOS) / ESPERADO_MUNICIPIOS
    if desvio > 0.10:
        error(f"{n} municipios, se esperaban unos {ESPERADO_MUNICIPIOS}. "
              f"Revisa el filtro de au_ID_concatenated o la deduplicacion.")
    else:
        ok(f"{n} municipios unicos, se esperaban unos {ESPERADO_MUNICIPIOS}")

    if df["id_municipio"].duplicated().any():
        dups = df[df["id_municipio"].duplicated(keep=False)]["id_municipio"].unique()
        error(f"{len(dups)} id_municipio duplicados: {list(dups[:5])}")
    else:
        ok("Sin duplicados en id_municipio")

    teselas = df.groupby(["zoom", "tile_x", "tile_y"]).size()
    if len(teselas) > ESPERADO_TESELAS:
        error(f"{len(teselas)} teselas referenciadas, solo hay {ESPERADO_TESELAS}")
    else:
        ok(f"{len(teselas)} teselas aportan municipios")

    dist = df["zoom"].value_counts().sort_index()
    print(f"       Municipios por zoom: {dist.to_dict()}")
    if dist.index.min() > 3:
        aviso("No hay municipios en zoom 3, revisa que esas teselas se leyeran")

    # Peru y Perú son el mismo pais, se comparan sin tildes.
    paises = set(df["pais"].dropna().unique())
    vistos = {sin_tildes(p) for p in paises}
    faltantes = {sin_tildes(p) for p in PAISES_ESPERADOS if sin_tildes(p) not in vistos}
    if faltantes:
        aviso(f"Paises sin municipios: {sorted(faltantes)}. "
              f"Puede estar bien si las teselas usadas no los cubren.")
    else:
        ok(f"Los {len(vistos)} paises esperados estan presentes")
    print(f"       {df['pais'].value_counts().to_dict()}")

    # fuente es la clave con la que se emparejan los documentos al corregir.
    if "fuente" in df and "doc_id" in df:
        vacios = df["fuente"].isna().sum() + (df["fuente"].astype(str) == "").sum()
        if vacios:
            error(f"{vacios} municipios sin fuente")
        pares = df[["doc_id", "fuente"]].drop_duplicates()
        if pares["doc_id"].duplicated().any():
            error("Hay doc_id con mas de una fuente asociada")
        elif pares["fuente"].duplicated().any():
            n = int(pares["fuente"].duplicated().sum())
            aviso(f"{n} fuentes repetidas entre doc_id distintos. ADL reutiliza "
                  f"nombres de archivo, hay que usar la ruta completa.")
        else:
            ok(f"{len(pares)} pares doc_id y fuente univocos")

    for campo in ["nivel2", "pais", "id_municipio"]:
        vacios = df[campo].isna().sum()
        if vacios:
            error(f"{vacios} municipios sin {campo}")

    texto_todo = " ".join(df["nivel1"].fillna("").astype(str)
                          + " " + df["nivel2"].fillna("").astype(str))
    if re.search(PATRON_MOJIBAKE, texto_todo):
        error("Hay mojibake sin reparar en los nombres de lugar")
    else:
        ok("Sin mojibake en toponimos")

    if "poblacion" in df:
        raras = df[(df["poblacion"] < 0) | (df["poblacion"] > 20_000_000)]
        if len(raras):
            aviso(f"{len(raras)} municipios con poblacion fuera de rango plausible")

    con_grupos = (df["n_grupos"] > 0).sum()
    print(f"       Con presencia armada: {con_grupos} "
          f"({con_grupos / len(df):.0%}), sin registro: {len(df) - con_grupos}")


def verificar_esquema(chunks):
    print("\n2. Campos obligatorios de la Tabla 1")

    sin_campo = Counter()
    for c in chunks:
        for campo in OBLIGATORIOS:
            if c.get(campo) in (None, ""):
                sin_campo[campo] += 1
    if sin_campo:
        for campo, n in sin_campo.items():
            error(f"{n} chunks sin el campo obligatorio {campo}")
    else:
        ok(f"Los {len(OBLIGATORIOS)} campos obligatorios estan en todos los chunks")

    for c in chunks[:1]:
        tipos = {"doc_id": str, "chunk_id": str, "fuente": str, "formato": str,
                 "fenomeno": int, "posicion": int, "num_tokens": int, "texto": str}
        for campo, tipo in tipos.items():
            if not isinstance(c.get(campo), tipo):
                error(f"El campo {campo} es {type(c.get(campo)).__name__}, "
                      f"se esperaba {tipo.__name__}")

    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        rep = [k for k, v in Counter(ids).items() if v > 1]
        error(f"{len(rep)} chunk_id duplicados: {rep[:3]}")
    else:
        ok("chunk_id unicos")

    fen = {c["fenomeno"] for c in chunks}
    if fen != {3}:
        error(f"fenomeno deberia ser 3 en todos, se encontro {fen}")
    else:
        ok("fenomeno es 3 en todos")

    por_doc = {}
    for c in chunks:
        por_doc.setdefault(c["doc_id"], []).append(c["posicion"])
    malos = [d for d, ps in por_doc.items() if sorted(ps) != list(range(len(ps)))]
    if malos:
        error(f"{len(malos)} documentos con posicion no consecutiva desde 0: {malos[:3]}")
    else:
        ok(f"posicion consecutiva desde 0 en los {len(por_doc)} documentos")

    formatos = {c["formato"] for c in chunks}
    if not formatos <= {"pdf", "html", "md"}:
        aviso(f"formato {formatos} no esta en la lista de la Tabla 1, que solo "
              f"nombra pdf, html y md. Confirmar con la organizacion.")


def verificar_texto(chunks):
    print("\n3. Calidad del texto")

    largos = [c for c in chunks if len(c["texto"].split()) > LIMITE_PALABRAS]
    if largos:
        error(f"{len(largos)} chunks pasan de {LIMITE_PALABRAS} palabras")
    else:
        pal = [len(c["texto"].split()) for c in chunks]
        ok(f"Todos bajo el limite. Palabras: min {min(pal)}, "
           f"media {sum(pal)/len(pal):.0f}, max {max(pal)}")

    cortados = [c for c in chunks if not c["texto"].strip().endswith((".", "!", "?"))]
    if cortados:
        error(f"{len(cortados)} chunks no terminan en punto: "
              f"{[c['chunk_id'] for c in cortados[:3]]}")
    else:
        ok("Todos terminan en oracion completa")

    patrones = {
        "valor nulo":        r"\b(None|nan|NaN|null|undefined)\b",
        "placeholder":       r"\{[a-z_]+\}",
        "parentesis vacio":  r"\(\s*\)",
        "doble espacio":     r"  +",
        "HTML":              r"<[a-zA-Z/][^>]*>",
        "coma huerfana":     r"\s,|,,",
        "palabra repetida":  r"\b(de|del|en|y)\s+\1\b",
    }
    for nombre, patron in patrones.items():
        malos = [c for c in chunks if re.search(patron, c["texto"])]
        if malos:
            aviso(f"{len(malos)} chunks con {nombre}, por ejemplo "
                  f"{malos[0]['chunk_id']}")

    sin_en = [c for c in chunks
              if "Amazon" not in c["texto"] or
              not re.search(r"\b(is documented by|is available for)\b", c["texto"])]
    if not sin_en:
        ok("Frase final en ingles presente en todos")
    elif len(sin_en) == len(chunks):
        aviso("Ningun chunk lleva la frase en ingles. Esta bien si se puso "
              "ANCLAJE_EN en False a proposito.")
    else:
        error(f"{len(sin_en)} de {len(chunks)} chunks sin la frase en ingles, "
              f"deberia estar en todos o en ninguno")

    con_pt = [c for c in chunks if "é um município" in c["texto"]]
    pt_no_brasil = [c for c in con_pt if c.get("pais") != "Brasil"]
    brasil = [c for c in chunks if c.get("pais") == "Brasil"]
    if pt_no_brasil:
        error(f"{len(pt_no_brasil)} chunks en portugues fuera de Brasil")
    elif brasil and len(con_pt) < len(brasil) * 0.9:
        aviso(f"Solo {len(con_pt)} de {len(brasil)} municipios brasilenos "
              f"abren en portugues")
    else:
        ok(f"Portugues solo en Brasil, {len(con_pt)} chunks")

    # Un chunk no puede nombrar un grupo y a la vez decir que no hay datos.
    contradictorios = [
        c for c in chunks
        if re.search(r"no registra información|No verified information", c["texto"])
        and re.search(r"estructuras? locales?", c["texto"])
    ]
    if contradictorios:
        error(f"{len(contradictorios)} chunks se contradicen, niegan tener "
              f"informacion y a la vez nombran grupos: "
              f"{[c['chunk_id'] for c in contradictorios[:3]]}")
    else:
        ok("Ningun chunk se contradice a si mismo")

    P = {"una": 1, "dos": 2, "tres": 3}

    def _declarado(t):
        m = re.search(r"presencia de (?:la|las)?\s*(una|dos|tres|\d+)?\s*estructuras?", t)
        if not m:
            return None
        g = m.group(1)
        return 1 if g is None else P.get(g, int(g) if g.isdigit() else None)

    desalineados = [c for c in chunks
                    if (n := _declarado(c["texto"])) is not None
                    and "estructuras locales" not in c["texto"]
                    and "estructura local" not in c["texto"]
                    and n != c.get("n_grupos")]
    if desalineados:
        aviso(f"{len(desalineados)} chunks con n_grupos distinto de lo que dice "
              f"su texto: {[c['chunk_id'] for c in desalineados[:3]]}")
    else:
        ok("n_grupos coincide con el texto")

    # La apertura brasilena va en portugues, su cola tambien.
    mezcla = [c for c in chunks
              if re.search(r"é um município[^.]*\b(con una|y una extensión)\b",
                           c["texto"])]
    if mezcla:
        error(f"{len(mezcla)} chunks mezclan portugues y espanol en la misma "
              f"oracion: {[c['chunk_id'] for c in mezcla[:3]]}")
    else:
        ok("Sin mezcla de idiomas dentro de una oracion")

    candidatos = set()
    for c in chunks:
        for campo in ("nivel1", "nivel2"):
            v = c.get(campo)
            if isinstance(v, str) and len(v) > 4 and v == sin_tildes(v):
                candidatos.add(v)
    if candidatos:
        aviso(f"{len(candidatos)} toponimos sin ninguna tilde. Hay que mirar a "
              f"mano cuales faltan en CORRECCIONES:")
        print(f"       {sorted(candidatos)[:25]}")

    # Si demasiados chunks son casi iguales, el encoder no los distingue.
    firmas = Counter(re.sub(r"[A-ZÁÉÍÓÚÑ][\wáéíóúñçã]*", "X", c["texto"])
                     for c in chunks)
    repetida, n = firmas.most_common(1)[0]
    if n > len(chunks) * 0.5:
        aviso(f"{n} de {len(chunks)} chunks comparten la misma estructura "
              f"({n/len(chunks):.0%}), riesgo de redundancia en el ranking")
    else:
        ok(f"Estructuras distintas: {len(firmas)} patrones en {len(chunks)} chunks")


def muestra_para_revisar(chunks, n_por_pais=3):
    print("\n4. Muestra para leer a ojo")
    print("Los scripts no detectan un texto correcto pero absurdo.\n")

    por_pais = {}
    for c in chunks:
        por_pais.setdefault(c.get("pais", "sin pais"), []).append(c)

    seleccion = []
    for pais, lista in sorted(por_pais.items()):
        lista = sorted(lista, key=lambda c: -c.get("n_grupos", 0))
        con = [c for c in lista if c.get("n_grupos", 0) > 0]
        sin = [c for c in lista if c.get("n_grupos", 0) == 0]
        elegidos = con[:1] + con[len(con)//2:len(con)//2+1] + sin[:1]
        seleccion.extend(elegidos[:n_por_pais])

    for c in seleccion:
        print(f"{c['chunk_id']}, {c.get('pais')}, "
              f"{len(c['texto'].split())} palabras, {c['num_tokens']} tokens")
        print(c["texto"])
        print()

    salida = Path("muestra_revision.txt")
    with open(salida, "w", encoding="utf-8") as f:
        for c in seleccion:
            f.write(f"{c['chunk_id']} {c.get('pais')}\n{c['texto']}\n\n")
    print(f"Guardada en {salida} para revisar con el equipo.")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    df = (pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet")
          else pd.read_json(sys.argv[1], lines=True))
    chunks = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]

    verificar_decodificacion(df)
    verificar_esquema(chunks)
    verificar_texto(chunks)
    muestra_para_revisar(chunks)

    print()
    if fallos:
        print(f"Resultado: {len(fallos)} fallos y {len(avisos)} avisos")
        print("Los fallos hay que corregirlos antes de indexar.")
        sys.exit(1)
    elif avisos:
        print(f"Resultado: sin fallos, {len(avisos)} avisos por revisar")
    else:
        print("Resultado: todo correcto")


if __name__ == "__main__":
    main()
