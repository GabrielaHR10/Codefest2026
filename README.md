# CODEFEST AD ASTRA 2026 — Etapa 1

Base de conocimiento vectorial sobre el corpus de fuentes abiertas de ADL, con
grafo de conocimiento y módulo de recuperación híbrida.

```
entrega/                            <- lo que se evalúa (Sección 1.4)
  generador.py                        recuperación + escritura de resultados
  consultas.txt                       las 50 consultas q001–q050
  resultados.jsonl                    50 líneas: 3 documentos y 10 fragmentos
  informe_tecnico.pdf                 decisiones de diseño
  base_vectorial/
    encoder_bge-m3/                   índice FAISS + metadata (9.886 fragmentos)
      index.faiss
      metadata.jsonl
    encoder_multilingual-e5-base/     índice FAISS + metadata (79.178 fragmentos)
      index.faiss
      metadata.jsonl
  grafo/                              componente bonus (Sección 7)
    grafo.graphml

modulos/                            <- construcción de la base (no se evalúa)
  base_vectorial/unificar_indices.py  fusión de particiones de un mismo encoder
  grafo/construirGrafo.py             NER + extracción de relaciones -> grafo
  grafo/consultaGrafo.py              consulta interactiva del grafo
  mapas/                              decodificación de los mapas .pbf a texto
  validar_resultados.py               verificación del esquema de la Sección 9
```

Los archivos pesados (`*.faiss`, `*.jsonl`, `*.graphml`) se versionan con
**Git LFS**. Tras clonar hace falta `git lfs install && git lfs pull`.

## Reproducir los resultados

```bash
pip install -r requirements.txt
python entrega/generador.py
python modulos/validar_resultados.py
```

`generador.py` lee `entrega/consultas.txt`, interroga los dos índices FAISS y el
grafo, y reescribe `entrega/resultados.jsonl`. Tarda unos 3 minutos: 45 s en
recorrer el grafo de 1 GB una sola vez para las 50 consultas y el resto en la
recuperación. También acepta directamente el PDF de preguntas:

```bash
python entrega/generador.py --consultas Extracto_Preguntas_50_v2.pdf
python entrega/generador.py --sin-grafo          # solo la vía vectorial
python entrega/generador.py --dispositivo cpu    # sin GPU/MPS
```

## La base de conocimiento

El corpus se repartió entre dos encoders complementarios (Sección 4.4), cada uno
con su propio índice `IndexFlatIP` sobre vectores normalizados, que es
exactamente similitud coseno (Sección 8.2) :

| Encoder | Dim. | Fragmentos | Partición del corpus |
|---|---|---|---|
| `BAAI/bge-m3` | 1024 | 9.886 | xlsx del AI Index + 985 fragmentos de mapas `.pbf` |
| `intfloat/multilingual-e5-base` | 768 | 79.178 | pdf de observatorios + json de artículos web |

El grafo de conocimiento tiene 226.821 entidades y 5.118.081 relaciones, cada
una con los `chunk_id` que la sostienen, y cubre las tres particiones.

Cómo se combinan las tres fuentes en la recuperación está explicado en el
encabezado de `entrega/generador.py` y, con más detalle, en
`entrega/informe_tecnico.pdf`.

## Restricción sobre modelos generativos

Ninguna etapa usa modelos decoder (Sección 8.3). Los dos modelos empleados son
encoders (familia BERT/XLM-R) y el grafo se consulta por coincidencia exacta
contra su vocabulario de entidades; el reordenamiento se hace solo con
puntuaciones de similitud coseno y metadata.
