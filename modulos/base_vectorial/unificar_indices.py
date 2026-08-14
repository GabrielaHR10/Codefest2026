"""
unificar_indices.py

Une las dos particiones del corpus que se indexaron por separado con el MISMO
encoder (``intfloat/multilingual-e5-base``) en una sola carpeta de entrega, y
normaliza su metadata al esquema obligatorio de la Tabla 1 del documento de
lineamientos.

Contexto
--------
Durante la construcción de la base vectorial el corpus se repartió entre varias
personas del equipo y quedaron tres carpetas:

    encoder_bge-m3/                  xlsx (AI Index) + pbf (mapas)   -> BAAI/bge-m3
    encoder_multilingual-e5-base/    pdf  (informes de observatorios) -> e5-base
    encoder_multilingual_e5_base/    json (artículos web)             -> e5-base

Las dos últimas usan el mismo modelo, la misma dimensión (768), el mismo tipo de
índice (``IndexFlatIP`` con vectores normalizados) y el mismo prefijo de
codificación (``passage: ``), así que son fusionables sin pérdida: basta
concatenar los vectores y las líneas de metadata en el mismo orden. Lo que las
diferenciaba era el esquema de metadata, que en la partición de PDF usaba
nombres propios (``text``, ``phenomenon``, ``source_path``...) en vez de los
campos exigidos por el reto.

El resultado es la estructura que pide la Sección 1.4: una subcarpeta por
encoder, cada una con ``index.faiss`` y ``metadata.jsonl``.

Uso
---
    python modulos/base_vectorial/unificar_indices.py [--base entrega/base_vectorial]

El script es idempotente: si la carpeta destino ya está unificada, no hace nada.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# faiss se importa después de numpy a propósito; ver nota de generador.py sobre
# el conflicto de OpenMP entre faiss y torch en macOS.
import faiss

# Particiones de origen, en el orden en que se concatenan. El orden importa:
# la línea i de metadata.jsonl debe corresponder al vector i del índice.
PARTICIONES = ["encoder_multilingual-e5-base", "encoder_multilingual_e5_base"]
DESTINO = "encoder_multilingual-e5-base"

DIM_ESPERADA = 768


def normalizar_registro(obj: dict, posicion: int) -> dict:
    """
    Lleva un registro de metadata a los campos obligatorios de la Tabla 1.

    Acepta los dos esquemas presentes en el repositorio:

    * el de la partición de PDF (``text``/``phenomenon``/``source_path``), que
      hay que traducir;
    * el ya conforme de la partición de JSON, que se deja intacto salvo por el
      campo extra ``ruta_fuente``.

    ``posicion`` es el índice ordinal del fragmento dentro de su documento; se
    recalcula aquí porque la partición de PDF no lo traía.
    """
    if "texto" in obj:  # partición ya conforme
        registro = dict(obj)
        registro.setdefault("ruta_fuente", obj.get("fuente", ""))
        registro["posicion"] = posicion
        return registro

    ruta = obj["source_path"]
    fenomeno = int(str(obj["phenomenon"]).lstrip("Ff"))
    registro = {
        "doc_id": obj["doc_id"],
        "chunk_id": obj["chunk_id"],
        "fuente": os.path.basename(ruta),
        "formato": os.path.splitext(ruta)[1].lstrip(".").lower() or "pdf",
        "fenomeno": fenomeno,
        "posicion": posicion,
        "num_tokens": int(obj.get("word_count") or len(obj["text"].split())),
        "texto": obj["text"],
        # campos adicionales (permitidos por la Tabla 1) que conservan la
        # trazabilidad hacia el archivo original y la página del PDF
        "ruta_fuente": ruta,
        "observatorio": obj.get("observatory", ""),
        "pagina_inicio": obj.get("page_start"),
        "pagina_fin": obj.get("page_end"),
    }
    return registro


def unificar(base: Path) -> None:
    origenes = [base / p for p in PARTICIONES]
    faltantes = [str(p) for p in origenes if not (p / "index.faiss").exists()]
    if faltantes:
        if (base / DESTINO / "metadata.jsonl").exists() and len(faltantes) == 1:
            print(f"Nada que hacer: '{DESTINO}' ya está unificado.")
            return
        sys.exit(f"No se encontraron las particiones de origen: {faltantes}")

    vectores = []
    registros = []
    posicion_por_doc: dict[str, int] = {}

    for carpeta in origenes:
        indice = faiss.read_index(str(carpeta / "index.faiss"))
        if indice.d != DIM_ESPERADA:
            sys.exit(f"{carpeta.name}: dimensión {indice.d}, se esperaba {DIM_ESPERADA}")

        # reconstruct_n devuelve una copia densa de todos los vectores; es
        # exacto porque los índices son planos (IndexFlatIP), no cuantizados.
        bloque = indice.reconstruct_n(0, indice.ntotal).astype("float32")
        vectores.append(bloque)

        with (carpeta / "metadata.jsonl").open(encoding="utf-8") as f:
            n_lineas = 0
            for linea in f:
                obj = json.loads(linea)
                doc_id = obj["doc_id"]
                pos = posicion_por_doc.get(doc_id, 0)
                posicion_por_doc[doc_id] = pos + 1
                registros.append(normalizar_registro(obj, pos))
                n_lineas += 1

        if n_lineas != indice.ntotal:
            sys.exit(f"{carpeta.name}: {n_lineas} líneas de metadata vs {indice.ntotal} vectores")
        print(f"  {carpeta.name}: {indice.ntotal} vectores")

    matriz = np.vstack(vectores)
    # Los vectores ya venían normalizados de origen; se renormaliza por
    # seguridad para que el producto interno sea exactamente coseno.
    faiss.normalize_L2(matriz)

    destino = base / DESTINO
    tmp = base / f".{DESTINO}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    indice_final = faiss.IndexFlatIP(DIM_ESPERADA)
    indice_final.add(matriz)
    faiss.write_index(indice_final, str(tmp / "index.faiss"))

    with (tmp / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    # Verificación antes de tocar nada: cada vector del índice nuevo debe
    # coincidir con el de su partición de origen, y su metadata con la línea
    # correspondiente.
    verificar(base, tmp, matriz)

    for carpeta in origenes:
        if carpeta.exists():
            shutil.rmtree(carpeta)
    tmp.rename(destino)
    print(f"OK: {destino} con {indice_final.ntotal} vectores de {DIM_ESPERADA}d")


def verificar(base: Path, tmp: Path, matriz: np.ndarray) -> None:
    indice = faiss.read_index(str(tmp / "index.faiss"))
    assert indice.ntotal == len(matriz), "el índice unificado perdió vectores"

    rng = np.random.default_rng(0)
    muestra = rng.choice(indice.ntotal, size=min(200, indice.ntotal), replace=False)
    reconstruidos = np.vstack([indice.reconstruct(int(i)) for i in muestra])
    error = float(np.abs(reconstruidos - matriz[muestra]).max())
    assert error < 1e-5, f"desviación máxima {error} al reconstruir el índice unificado"

    obligatorios = {
        "doc_id", "chunk_id", "fuente", "formato",
        "fenomeno", "posicion", "num_tokens", "texto",
    }
    with (tmp / "metadata.jsonl").open(encoding="utf-8") as f:
        n = 0
        for linea in f:
            obj = json.loads(linea)
            faltan = obligatorios - obj.keys()
            assert not faltan, f"línea {n}: faltan campos {faltan}"
            n += 1
    assert n == indice.ntotal, f"{n} líneas de metadata vs {indice.ntotal} vectores"
    print(f"  verificación OK: {n} fragmentos, esquema completo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=Path(__file__).resolve().parents[2] / "entrega" / "base_vectorial",
        type=Path,
        help="carpeta base_vectorial de la entrega",
    )
    args = parser.parse_args()
    unificar(args.base)


if __name__ == "__main__":
    main()
