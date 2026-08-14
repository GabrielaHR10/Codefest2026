"""
Mete los chunks de mapas en la base vectorial del equipo.

No vale con juntar los jsonl: FAISS solo guarda vectores numerados y el
metadata.jsonl los enlaza por posicion, asi que hay que codificar los textos y
anadir vectores y lineas en el mismo orden. Si no, el sistema empieza a
devolver el texto equivocado sin dar ningun error.

Uso:
    python integrar_mapas.py <dir_encoder> [metadata_mapas.jsonl] [modelo]

El segundo argumento es opcional, por defecto busca metadata_mapas.jsonl en el
directorio actual, que es donde lo deja generador_chunks.py.

El tercero solo hace falta si el equipo anade un segundo encoder. Hay un indice
por modelo, no por modulo: todos los chunks del corpus van dentro del mismo
indice, y lo que cambia entre carpetas es con que modelo se codificaron.

    python integrar_mapas.py entrega/base_vectorial/encoder_e5-large \\
        metadata_mapas.jsonl intfloat/multilingual-e5-large
"""

import json
import shutil
import sys
from pathlib import Path

MODELO = "BAAI/bge-m3"
LOTE = 16

# BGE-m3 no necesita prefijo en los pasajes, pero la familia E5 exige
# 'passage: ' y sin el rinde bastante peor, sin dar ningun sintoma. Si se anade
# un encoder, hay que poner aqui el prefijo que le toque y usar el mismo que
# usaron los demas chunks de ese indice.
PREFIJOS = {
    "intfloat/multilingual-e5-large": "passage: ",
    "intfloat/multilingual-e5-base": "passage: ",
}
PREFIJO_PASAJE = ""


def cargar_jsonl(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    dir_encoder = Path(sys.argv[1])
    ruta_mapas = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("metadata_mapas.jsonl")
    modelo_nombre = sys.argv[3] if len(sys.argv) > 3 else MODELO
    prefijo = PREFIJOS.get(modelo_nombre, PREFIJO_PASAJE)
    if not ruta_mapas.exists():
        sys.exit(f"No existe {ruta_mapas}.\n"
                 f"Genera primero los chunks con los tres pasos del README.")

    ruta_index = dir_encoder / "index.faiss"
    ruta_meta = dir_encoder / "metadata.jsonl"
    for r in (ruta_index, ruta_meta):
        if not r.exists():
            sys.exit(f"No existe: {r}")

    try:
        import faiss
        import numpy as np
    except ImportError:
        sys.exit("Falta faiss: pip install faiss-cpu numpy")

    meta = cargar_jsonl(ruta_meta)
    mapas = cargar_jsonl(ruta_mapas)
    index = faiss.read_index(str(ruta_index))

    print(f"Indice actual: {index.ntotal} vectores, dimension {index.d}, "
          f"{type(index).__name__}")
    print(f"Metadata actual: {len(meta)} lineas")
    print(f"Mapas a anadir: {len(mapas)} chunks")

    if index.ntotal != len(meta):
        sys.exit(f"\nSe detiene: el indice y la metadata ya estan desalineados, "
                 f"{index.ntotal} vectores frente a {len(meta)} lineas. "
                 f"Hay que arreglar eso antes de anadir nada.")

    ya = {c.get("chunk_id") for c in meta}
    nuevos = [c for c in mapas if c.get("chunk_id") not in ya]
    if not nuevos:
        print("\nLos mapas ya estaban integrados, no se hace nada.")
        return
    if len(nuevos) != len(mapas):
        sys.exit(f"\nSe detiene: {len(mapas) - len(nuevos)} chunks de mapas ya "
                 f"estan en la metadata y {len(nuevos)} no. Integracion a "
                 f"medias, hay que revisarlo a mano.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Falta el encoder: pip install sentence-transformers")

    print(f"\nCargando {modelo_nombre}")
    if prefijo:
        print(f"Prefijo de pasaje: {prefijo!r}")
    modelo = SentenceTransformer(modelo_nombre)
    dim = modelo.get_sentence_embedding_dimension()
    if dim != index.d:
        sys.exit(f"Se detiene: el encoder da vectores de {dim} dimensiones y el "
                 f"indice espera {index.d}, no es el mismo encoder.")

    textos = [prefijo + c["texto"] for c in nuevos]
    print(f"Codificando {len(textos)} chunks")
    vecs = modelo.encode(textos, batch_size=LOTE, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")

    # El indice usa producto interno con vectores normalizados, que equivale a
    # similitud coseno. Sin normalizar, los chunks competirian por su longitud.
    normas = np.linalg.norm(vecs, axis=1)
    if not np.allclose(normas, 1, atol=1e-4):
        sys.exit(f"Se detiene: los vectores no salieron normalizados, "
                 f"normas entre {normas.min():.4f} y {normas.max():.4f}.")

    for r in (ruta_index, ruta_meta):
        shutil.copy2(r, r.with_suffix(r.suffix + ".bak"))
    print(f"\nCopia de seguridad en {dir_encoder}, archivos .bak")

    base = index.ntotal
    index.add(vecs)
    faiss.write_index(index, str(ruta_index))

    with open(ruta_meta, "a", encoding="utf-8") as f:
        for c in nuevos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("\nVerificacion")
    index2 = faiss.read_index(str(ruta_index))
    meta2 = cargar_jsonl(ruta_meta)
    todo_ok = True

    if index2.ntotal != len(meta2):
        print(f"  FALLO: {index2.ntotal} vectores frente a {len(meta2)} lineas")
        todo_ok = False
    else:
        print(f"  OK: {index2.ntotal} vectores y {len(meta2)} lineas")

    if [c["chunk_id"] for c in meta2[:len(meta)]] == [c["chunk_id"] for c in meta]:
        print(f"  OK: las {len(meta)} lineas previas conservan su posicion")
    else:
        print("  FALLO: las lineas previas cambiaron de orden")
        todo_ok = False

    # Prueba de alineamiento: buscar unos vectores dentro del propio indice y
    # comprobar que cada uno se encuentra a si mismo. Si la metadata estuviera
    # corrida aunque fuera una linea, esto lo detecta.
    rng = np.random.default_rng(0)
    muestra = rng.choice(index2.ntotal, size=min(20, index2.ntotal), replace=False)
    fallos = []
    for i in muestra:
        v = index2.reconstruct(int(i)).reshape(1, -1)
        _, idx = index2.search(v, 1)
        if int(idx[0][0]) != int(i):
            fallos.append(int(i))
    if fallos:
        print(f"  FALLO: {len(fallos)} vectores no se recuperan a si mismos: {fallos[:5]}")
        todo_ok = False
    else:
        print(f"  OK: {len(muestra)} vectores al azar recuperan su propia linea")

    print(f"\n  Anadidos {len(nuevos)} chunks en las posiciones "
          f"{base} a {index2.ntotal - 1}")
    if not todo_ok:
        print("\n  Algo salio mal, restaura los .bak antes de seguir.")
        sys.exit(1)
    print("\nIntegracion correcta.")


if __name__ == "__main__":
    main()
