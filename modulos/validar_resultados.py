"""
validar_resultados.py

Comprueba que ``resultados.jsonl`` cumple exactamente el esquema de la Sección 9
del documento de lineamientos antes de entregarlo. La evaluación es automática y
penaliza o descarta los archivos mal formados (Sección 9.3.2), así que conviene
correr esto siempre después de ``generador.py``.

    python modulos/validar_resultados.py [entrega/resultados.jsonl]

Termina con código 1 si encuentra algún fallo, para poder encadenarlo en una
verificación automática.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

N_CONSULTAS = 50
N_DOCUMENTOS = 3
N_FRAGMENTOS = 10
MAX_PALABRAS = 250

CAMPOS_FRAGMENTO = {"rank", "chunk_id", "doc_id", "text"}
CAMPOS_DOCUMENTO = {"rank", "doc_id"}


def validar(ruta: Path, ruta_metadata: list[Path] | None = None) -> list[str]:
    fallos: list[str] = []
    lineas = [l for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]

    if len(lineas) != N_CONSULTAS:
        fallos.append(f"el archivo tiene {len(lineas)} líneas, se esperaban {N_CONSULTAS}")

    esperados = [f"q{i:03d}" for i in range(1, N_CONSULTAS + 1)]
    vistos: list[str] = []
    max_palabras_visto = 0

    for numero, linea in enumerate(lineas, 1):
        try:
            obj = json.loads(linea)
        except json.JSONDecodeError as error:
            fallos.append(f"línea {numero}: JSON inválido ({error})")
            continue

        prefijo = f"línea {numero}"
        faltan = {"query_id", "documents", "fragments"} - obj.keys()
        if faltan:
            fallos.append(f"{prefijo}: faltan campos {sorted(faltan)}")
            continue
        vistos.append(obj["query_id"])
        prefijo = f"{obj['query_id']}"

        documentos = obj["documents"]
        if len(documentos) != N_DOCUMENTOS:
            fallos.append(f"{prefijo}: {len(documentos)} documentos, se esperaban {N_DOCUMENTOS}")
        for i, doc in enumerate(documentos, 1):
            if CAMPOS_DOCUMENTO - doc.keys():
                fallos.append(f"{prefijo}: documento {i} sin {sorted(CAMPOS_DOCUMENTO - doc.keys())}")
            elif doc["rank"] != i:
                fallos.append(f"{prefijo}: documento en posición {i} con rank={doc['rank']}")
        ids_documento = [d.get("doc_id") for d in documentos]
        if len(set(ids_documento)) != len(ids_documento):
            fallos.append(f"{prefijo}: documentos repetidos {ids_documento}")

        fragmentos = obj["fragments"]
        if len(fragmentos) != N_FRAGMENTOS:
            fallos.append(f"{prefijo}: {len(fragmentos)} fragmentos, se esperaban {N_FRAGMENTOS}")
        for i, frag in enumerate(fragmentos, 1):
            if CAMPOS_FRAGMENTO - frag.keys():
                fallos.append(f"{prefijo}: fragmento {i} sin {sorted(CAMPOS_FRAGMENTO - frag.keys())}")
                continue
            if frag["rank"] != i:
                fallos.append(f"{prefijo}: fragmento en posición {i} con rank={frag['rank']}")
            n = len(frag["text"].split())
            max_palabras_visto = max(max_palabras_visto, n)
            if n > MAX_PALABRAS:
                fallos.append(f"{prefijo}: fragmento {i} con {n} palabras (máx. {MAX_PALABRAS})")
            if not frag["text"].strip():
                fallos.append(f"{prefijo}: fragmento {i} vacío")

    if vistos != esperados[:len(vistos)]:
        fallos.append("los query_id no van en orden q001..q050")

    # Trazabilidad: los identificadores reportados deben existir en la base
    # vectorial entregada (Sección 5.3).
    if ruta_metadata:
        chunks, docs = set(), set()
        for ruta_meta in ruta_metadata:
            with ruta_meta.open(encoding="utf-8") as f:
                for linea in f:
                    registro = json.loads(linea)
                    chunks.add(registro["chunk_id"])
                    docs.add(registro["doc_id"])
        for linea in lineas:
            obj = json.loads(linea)
            for frag in obj["fragments"]:
                if frag["chunk_id"] not in chunks:
                    fallos.append(f"{obj['query_id']}: chunk_id desconocido {frag['chunk_id']}")
                if frag["doc_id"] not in docs:
                    fallos.append(f"{obj['query_id']}: doc_id desconocido {frag['doc_id']}")
            for doc in obj["documents"]:
                if doc["doc_id"] not in docs:
                    fallos.append(f"{obj['query_id']}: doc_id desconocido {doc['doc_id']}")

    print(f"Consultas: {len(lineas)}")
    print(f"Palabras del fragmento más largo: {max_palabras_visto} (límite {MAX_PALABRAS})")
    return fallos


def main() -> None:
    raiz = Path(__file__).resolve().parents[1]
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else raiz / "entrega" / "resultados.jsonl"
    metadatas = sorted((raiz / "entrega" / "base_vectorial").glob("*/metadata.jsonl"))
    fallos = validar(ruta, metadatas or None)
    if fallos:
        print(f"\n{len(fallos)} FALLOS:")
        for fallo in fallos[:50]:
            print("  -", fallo)
        sys.exit(1)
    print("\nSin fallos: el archivo cumple el esquema de la Sección 9.")


if __name__ == "__main__":
    main()
