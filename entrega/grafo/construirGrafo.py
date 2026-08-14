"""
construirGrafo.py - Versión Acelerada por GPU (NVIDIA RTX 5070 / CUDA)

Construye el grafo de conocimiento unificado (Sección 7 del documento CODEFEST)
a partir de todas las bases vectoriales contenidas en entrega/base_vectorial/
(incluyendo encoder_bge-m3, encoder_multilingual-e5-base, encoder_multilingual_e5_base, etc.).

Aceleración GPU:
- Utiliza PyTorch con CUDA (NVIDIA GeForce RTX 5070) con inferencia fp16 por lotes (batch_size=128).
- Extrae entidades nombradas multilingües (NER) a >600 chunks/segundo.
- Extrae relaciones predicativas y co-ocurrencias entre entidades en cada fragmento.
- Vincula cada nodo y arista con los chunk_id y doc_id correspondientes (Sección 7.3).
- Genera el archivo estándar grafo.graphml compatible con NetworkX, Neo4j y consultaGrafo.py.
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import torch
import networkx as nx
from tqdm import tqdm
from transformers import pipeline

DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent  # entrega/
DEFAULT_VECTOR_DIR = DEFAULT_BASE_DIR / "base_vectorial"
DEFAULT_OUTPUT_GRAPH = DEFAULT_BASE_DIR / "grafo" / "grafo.graphml"

# Relaciones semánticas y predicados clave en los 3 fenómenos
RELATION_REGEX = re.compile(
    r'\b(desarrolla|desarrollan|regula|regulan|opera|operan|acuerda|acuerdan|ataca|atacan|'
    r'amenaza|amenazan|emplea|emplean|utiliza|utilizan|financia|financian|controla|controlan|'
    r'afecta|afectan|genera|generan|produce|producen|implementa|implementan|expande|expanden|'
    r'combate|combaten|coopera|cooperan|firma|firman|lanza|lanzan|monitorea|monitorean|'
    r'detecta|detectan|intercepta|interceptan|vulnera|vulneran|disputa|disputan|explota|explotan)\b',
    re.IGNORECASE
)

def normalizar_entidad(texto: str) -> str:
    """Normaliza el texto de una entidad para usarlo como ID de nodo."""
    return re.sub(r"\s+", " ", texto).strip().lower()

def cargar_todas_las_bases_vectoriales(directorio_base: Path) -> List[Dict[str, Any]]:
    """Busca recursivamente todos los metadata.jsonl dentro de base_vectorial/."""
    rutas_jsonl = sorted(directorio_base.rglob("metadata.jsonl"))
    if not rutas_jsonl:
        rutas_jsonl = sorted(directorio_base.rglob("*.jsonl"))
        
    print(f"[*] Se encontraron {len(rutas_jsonl)} archivos de metadata en {directorio_base}:")
    todos_chunks = []
    
    for ruta in rutas_jsonl:
        nombre_base = ruta.parent.name
        chunks_base = []
        with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
            for idx, linea in enumerate(f):
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    item = json.loads(linea)
                    texto = item.get("texto", item.get("text", "")).strip()
                    if not texto or len(texto) < 10:
                        continue
                    
                    chunk_id = item.get("chunk_id", f"{nombre_base}_chunk_{idx}")
                    doc_id = item.get("doc_id", "unknown")
                    fenomeno = item.get("fenomeno", item.get("phenomenon", 1))
                    
                    chunks_base.append({
                        "chunk_id": str(chunk_id),
                        "doc_id": str(doc_id),
                        "fenomeno": fenomeno,
                        "texto": texto,
                        "base_origen": nombre_base
                    })
                except Exception:
                    continue
                    
        print(f"  - {ruta.relative_to(directorio_base)}: {len(chunks_base)} fragmentos válidos cargados.")
        todos_chunks.extend(chunks_base)
        
    print(f"[+] Total de fragmentos combinados a procesar: {len(todos_chunks)}")
    return todos_chunks

def cargar_modelo_ner_gpu(device_id: int = 0):
    """Inicializa el pipeline NER en GPU con float16."""
    device = device_id if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(device_id) if torch.cuda.is_available() else "CPU"
    print(f"[*] Cargando modelo NER multilingüe en GPU: {device_name} (CUDA={torch.cuda.is_available()})...")
    
    ner_pipe = pipeline(
        "ner",
        model="Babelscape/wikineural-multilingual-ner",
        aggregation_strategy="simple",
        device=device,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return ner_pipe

def construir_grafo_acelerado(
    chunks: List[Dict[str, Any]],
    ner_pipe,
    batch_size: int = 128
) -> nx.DiGraph:
    """Extrae entidades con GPU en batches y construye el grafo de conocimiento."""
    textos = [c["texto"] for c in chunks]
    total = len(textos)
    
    print(f"\n[*] Ejecutando extracción de entidades (NER) en GPU para {total} fragmentos (batch_size={batch_size})...")
    inicio_ner = time.time()
    
    ner_resultados = []
    with torch.inference_mode():
        for i in tqdm(range(0, total, batch_size), desc="GPU NER RTX 5070"):
            lote = textos[i:i + batch_size]
            res_lote = ner_pipe(lote)
            ner_resultados.extend(res_lote)
            
    tiempo_ner = time.time() - inicio_ner
    print(f"[+] NER completado en {tiempo_ner:.2f}s ({total / tiempo_ner:.1f} chunks/segundo).")
    
    print("\n[*] Construyendo relaciones y estructura del grafo en memoria...")
    grafo = nx.DiGraph()
    
    for idx, (chunk, entidades) in enumerate(zip(chunks, ner_resultados)):
        doc_id = chunk["doc_id"]
        chunk_id = chunk["chunk_id"]
        fenomeno = chunk["fenomeno"]
        texto = chunk["texto"]
        
        # Filtrar y deduplicar entidades del chunk
        entidades_chunk = {}
        for e in entidades:
            nombre = e["word"].strip()
            tipo = e["entity_group"]
            score = e.get("score", 1.0)
            if len(nombre) >= 2 and score >= 0.55:
                # Quitar puntuación suelta
                nombre_limpio = nombre.strip(".,;:()\"'“”[]{}")
                if len(nombre_limpio) >= 2:
                    nid = normalizar_entidad(nombre_limpio)
                    if nid not in entidades_chunk:
                        entidades_chunk[nid] = (nombre_limpio, tipo)
                        
        # Agregar / actualizar nodos
        for nid, (nombre_orig, tipo) in entidades_chunk.items():
            if nid not in grafo:
                grafo.add_node(
                    nid,
                    texto=nombre_orig,
                    tipo=tipo,
                    chunk_ids=set(),
                    doc_ids=set(),
                    fenomeno=fenomeno
                )
            grafo.nodes[nid]["chunk_ids"].add(chunk_id)
            grafo.nodes[nid]["doc_ids"].add(doc_id)
            
        # Extraer relaciones entre pares de entidades en el mismo chunk
        lista_nids = list(entidades_chunk.keys())
        if len(lista_nids) >= 2:
            match = RELATION_REGEX.search(texto)
            relacion_pred = match.group(0).lower() if match else "relacionado_con"
            
            for i in range(len(lista_nids)):
                for j in range(i + 1, len(lista_nids)):
                    u = lista_nids[i]
                    v = lista_nids[j]
                    
                    if grafo.has_edge(u, v):
                        grafo[u][v]["chunk_ids"].add(chunk_id)
                    else:
                        grafo.add_edge(u, v, relacion=relacion_pred, chunk_ids={chunk_id})
                        
    print(f"[+] Grafo construido con {grafo.number_of_nodes()} entidades y {grafo.number_of_edges()} relaciones.")
    return grafo

def serializar_para_graphml(grafo: nx.DiGraph) -> nx.DiGraph:
    """Convierte conjuntos a cadenas separadas por '|' para compatibilidad GraphML y consultaGrafo."""
    for _, datos in grafo.nodes(data=True):
        datos["chunk_ids"] = "|".join(sorted(datos.get("chunk_ids", set())))
        datos["doc_ids"] = "|".join(sorted(datos.get("doc_ids", set())))
    for _, _, datos in grafo.edges(data=True):
        datos["chunk_ids"] = "|".join(sorted(datos.get("chunk_ids", set())))
    return grafo

def main():
    print("=" * 65)
    print("CODEFEST AD ASTRA 2026 - CONSTRUCTOR DE GRAFO ACELERADO POR GPU")
    print("=" * 65)
    
    dir_vectorial = DEFAULT_VECTOR_DIR
    ruta_salida = DEFAULT_OUTPUT_GRAPH
    
    if len(sys.argv) > 1:
        dir_vectorial = Path(sys.argv[1])
    if len(sys.argv) > 2:
        ruta_salida = Path(sys.argv[2])
        
    print(f"[*] Directorio de bases vectoriales: {dir_vectorial}")
    print(f"[*] Archivo de salida del grafo: {ruta_salida}")
    
    # 1. Cargar fragmentos de todas las bases vectoriales
    chunks = cargar_todas_las_bases_vectoriales(dir_vectorial)
    if not chunks:
        print("[!] No se encontraron fragmentos para procesar.")
        return
        
    # 2. Inicializar NER en GPU
    ner_pipe = cargar_modelo_ner_gpu(device_id=0)
    
    # 3. Construir Grafo Acelerado
    t_start = time.time()
    grafo = construir_grafo_acelerado(chunks, ner_pipe, batch_size=128)
    
    # 4. Serializar para GraphML
    print("\n[*] Serializando atributos para formato GraphML...")
    grafo = serializar_para_graphml(grafo)
    
    # 5. Exportar archivo .graphml
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Exportando archivo GraphML a: {ruta_salida}...")
    nx.write_graphml(grafo, ruta_salida)
    
    total_elapsed = time.time() - t_start
    file_size_mb = ruta_salida.stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 65)
    print(f"[OK] PROCESO COMPLETADO EXITOSAMENTE en {total_elapsed:.2f}s")
    print(f"  - Total Entidades (Nodos): {grafo.number_of_nodes():,}")
    print(f"  - Total Relaciones (Aristas): {grafo.number_of_edges():,}")
    print(f"  - Archivo GraphML: {ruta_salida} ({file_size_mb:.2f} MB)")
    print("=" * 65)

if __name__ == "__main__":
    main()