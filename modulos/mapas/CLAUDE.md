# Módulo de mapas — CODEFEST AD ASTRA 2026, Etapa 1

Procesa las 73 teselas Mapbox Vector Tile (`.pbf`) del observatorio **Amazon
Underworld** y genera fragmentos (chunks) para la base de conocimiento
vectorial del equipo. Corresponde al **Fenómeno 3** del reto (dinámicas
territoriales en América Latina).

Este es un módulo enchufable: produce un `metadata_mapas.jsonl` que se
concatena con los chunks del resto del corpus. **No construye el índice FAISS,
ni el `generador.py`, ni el `resultados.jsonl`** — eso es del pipeline central.

---

## Pipeline

```bash
pip install mapbox-vector-tile pandas pyarrow transformers openpyxl

python decodificador_mvt.py <ruta_corpus> municipios.parquet
python generador_chunks.py  municipios.parquet Indice_Datos_Codefest.xlsx
python verificar_mapas.py   municipios.parquet metadata_mapas.jsonl
```

| Script | Entrada | Salida |
|---|---|---|
| `decodificador_mvt.py` | árbol con los `.pbf` | `municipios.parquet` (~985 municipios únicos) |
| `generador_chunks.py` | el parquet anterior | `metadata_mapas.jsonl` |
| `verificar_mapas.py` | ambos | reporte + `muestra_revision.txt` |

`verificar_mapas.py` sale con código 1 si hay fallos, así que sirve en CI.

---

## Qué contienen los datos

Los `.pbf` **no son PBF de OpenStreetMap** — son Mapbox Vector Tiles del mapa
`amazonunderworld.1xih8pjj`. Se decodifican con la librería
`mapbox-vector-tile`, no con `osmium` ni `pyrosm`.

Una sola capa, `au_compilado_R02`, con polígonos de municipios (ADM2) en seis
países: Colombia, Brasil, Venezuela, Ecuador, Perú y Bolivia.

Campos relevantes por feature:

- `au_ID_concatenated` — llave de identidad del municipio (`para-tracuateua`)
- `b_ADM1_ES` / `b_ADM2_ES` — departamento y municipio en español
- `b_ADM1_PT` / `b_ADM2_PT` — lo mismo en portugués (solo Brasil)
- `au_country`, `au_population`, `au_area km`
- Once banderas booleanas de presencia armada: `au_eln`, `au_emc`, `au_embf`,
  `au_c_d_f`, `au_seg_marq`, `au_cv`, `au_pcc`, `au_choneros`, `au_lobos`,
  `au_others`, `au_no_info`
- `au_popup_window_es/en/pt` — **frentes y facciones con nombre propio**
  (Frente Rodrigo Cadete, Família Terror do Amapá, Comandos de la Frontera).
  Esto NO está en las banderas y es lo más valioso del dataset para búsqueda
  léxica: son términos raros que ningún otro documento del corpus contiene.

---

## Decisiones tomadas (no revertir sin discutir con el equipo)

**Fuente de verdad: el `.pbf`, no el `.xlsx` de ADL.** El archivo
`AMAZONUW_amazonunderworld-data.xlsx` es una extracción incompleta: solo las
filas de zoom 6 traen los campos `au_*`, y tiene mojibake generalizado
(`BogotÃ¡`). Decodificando el `.pbf` directamente todo sale completo y con los
acentos correctos. El xlsx solo sirve como control cruzado.

**Chunking: opción A — un chunk por municipio.** ~985 chunks de 80-135
palabras. Se evaluaron alternativas: agregación por departamento (opción B,
chunks más densos pero peor para consultas específicas), un chunk por tesela
(opción C, descartada: excede 250 palabras por mucho y arrastra duplicación) e
híbrido A+B (opción D, posible más adelante si el dev set lo justifica).

**Deduplicación entre zooms: se conserva el ZOOM MÁS BAJO.** El mismo municipio
aparece en los zooms 3, 4, 5 y 6 (~4.4 apariciones cada uno). Se conserva la
del zoom más bajo porque esas teselas son pocas y panorámicas, así que los
municipios quedan concentrados en menos archivos fuente — lo que mejora las
probabilidades en el emparejamiento por `fuente` del ground truth. Desempate
secundario: `tile_x`, `tile_y` ascendente. La columna `fuentes_todas` conserva
todas las teselas donde apareció cada municipio, para poder auditar o revertir
sin reprocesar.

**Estrategia multilingüe.** El reto evalúa con consultas en español, inglés y
portugués distribuidas de forma equilibrada (Sección 10.1), así que ~17 de las
50 consultas vienen en inglés.

- Cuerpo en español
- Apertura en portugués **solo para municipios de Brasil** (fuera de Brasil los
  campos `*_PT` vienen vacíos; el portugués llega gratis por los nombres de
  grupos y la cercanía léxica con el español)
- **Oración final autónoma en inglés** como anclaje, no un bloque de
  traducción. Cumple completitud lingüística y aporta los términos clave
  (`armed group presence`, `Brazilian Amazon`, `municipality`)
- Nombres con tilde y sin tilde cuando difieren: `Bolívar (Bolivar)`. Las
  consultas reales llegan sin tildes con frecuencia y para BM25 son tokens
  distintos

**Se incluyen los municipios sin información registrada** (~11% del total). El
texto dice explícitamente que la ausencia de registro no equivale a ausencia de
actividad. Ayuda si alguna consulta pregunta por vacíos de información o
cobertura del observatorio.

**Los popups se deduplican contra las banderas.** Comparación por núcleo
normalizado (sin tildes, sin paréntesis, minúsculas). Solo sobreviven las
estructuras que no tienen bandera propia. La etiqueta genérica
`"Otros grupos armados"` se descarta cuando el popup nombra facciones
concretas, pero se conserva si no hay popup.

**Encoder: `BAAI/bge-m3`.** Multilingüe nativo, licencia MIT, 8192 tokens de
contexto. Se usa solo para contar `num_tokens`. **Confirmar con el equipo** —
si eligen otro, es cambiar la constante `MODELO_ENCODER` en
`generador_chunks.py`.

---

## Restricciones del reto que afectan a este módulo

**Prohibidos los modelos generativos (arquitecturas decoder) en indexación y
recuperación** (Sección 8.3). Todo el texto de los chunks se construye con
plantillas y concatenación de cadenas — no interviene ningún LLM. Cuidado
también con los modelos de embeddings que son decoders por dentro
(`e5-mistral`, `gte-Qwen`): rinden bien en MTEB pero caen en la prohibición.

**Completitud lingüística** (Sección 3.3): ningún fragmento puede contener
oraciones incompletas. Se cumple por construcción, ya que cada chunk se arma
con oraciones enteras.

**Máximo 250 palabras por fragmento** (Sección 9.2). Los chunks van entre 76 y
136 palabras, muy holgado.

**Campos obligatorios de metadata** (Tabla 1): `doc_id`, `chunk_id`, `fuente`,
`formato`, `fenomeno`, `posicion`, `num_tokens`, `texto`. Se pueden añadir
campos extra, y este módulo añade varios (`pais`, `nivel1`, `nivel2`,
`n_grupos`, `idiomas`, `zoom`, `observatorio`, `nivel`).

**El campo `fuente` es la clave de emparejamiento del ground truth a nivel de
documento** (Sección 10.2.1), no el `doc_id`. Debe ser el nombre del archivo
tal como lo entregó ADL, tomado del inventario, no una ruta inventada.

**`posicion` empieza en 0** y es el índice ordinal del fragmento dentro de su
documento.

---

## Pendientes

- [ ] Correr con las 73 teselas y verificar que salen ~985 municipios únicos
      (~4.400 apariciones brutas antes de deduplicar)
- [ ] **Adaptar el parseo de `z/x/y` a la estructura real de carpetas.** El
      decodificador asume rutas `tiles/{z}/{x}/{y}.pbf`. Los archivos están
      dispersos en el corpus con nombres estandarizados (`AMAZONUW_15.pbf`).
      Mejor enfoque: leer `Indice_Datos_Codefest.xlsx` primero, construir el
      mapa `nombre_archivo → (z, x, y, doc_id, fuente)` desde las columnas
      `Carpeta` y `Nombre estandarizado`, y buscar cada archivo por nombre en
      cualquier subcarpeta. Así la estructura de directorios deja de importar
      y el `doc_id` sale del inventario oficial.
- [ ] Completar la tabla `CORRECCIONES` en `generador_chunks.py`. El
      verificador lista los candidatos; la mayoría legítimamente no lleva tilde
      (Albania, Boa Vista, Barcelos), así que es criterio humano. Confirmados
      pendientes: `Sucumbios → Sucumbíos`, `Rondonia → Rondônia`
- [ ] Validar el parseo de popups en el corpus completo. Solo se observó el
      formato `- SIGLA: Nombre - Nombre (ACRÓNIMO)`; con 985 municipios en seis
      países pueden aparecer variantes (HTML, saltos de línea, otros
      separadores)
- [ ] Leer a ojo los ~18 chunks de `muestra_revision.txt`. Los scripts detectan
      lo mecánico; el sinsentido semántico solo lo ve una persona
- [ ] **Preguntar a la organización si `formato: "pbf"` es válido.** La Tabla 1
      solo lista `pdf`, `html` y `md`. Si el validador automático rechaza
      valores fuera de esa lista, se pierden los ~985 chunks
- [ ] Confirmar el encoder con el equipo y regenerar `num_tokens` con el
      tokenizer real (sin `transformers` instalado cae a conteo de palabras)
- [ ] Acordar con el equipo **quién aplica los prefijos del encoder**. E5 exige
      `passage: ` en fragmentos y `query: ` en consultas; BGE-m3 no lo necesita
      en pasajes. Debería aplicarlo quien codifica, para que sea uniforme
- [ ] Escribir la sección del informe técnico (1.5 páginas): qué son los MVT,
      la pirámide de zooms y el criterio de deduplicación, la estrategia
      multilingüe, y el hallazgo de que el xlsx de ADL está incompleto

---

## Coordinación con el equipo

**Quién controla el orden final de `metadata.jsonl`** — el orden de las líneas
debe coincidir con los IDs internos de FAISS. Si alguien concatena este archivo
con los demás y otro reordena los vectores, todo se desalinea. Una sola persona
debe ser dueña de ese paso.

**El grafo de conocimiento (bonus) lo hace otra persona.** Este módulo no lo
construye. Pero `municipios.parquet` es el insumo ideal: las tripletas están
explícitas en las banderas booleanas, sin necesidad de NER ni extracción de
relaciones, y son verificables al 100% — sirven además como conjunto de
validación para calibrar el recall de su pipeline sobre texto libre. Antes de
exportarlas hay que acordar la convención de nombres de nodos y el vocabulario
de relaciones.

---

## Contexto: dimensión real de este módulo

El corpus completo tiene **1826 archivos**: 954 JSON, 759 PDF, 74 "Otro" (73
`.pbf` + 1 `.avif` mal clasificado), 26 CSV, 8 imágenes, 4 Excel, 1 texto.

Los mapas son el **4% de los archivos** y probablemente el 1-2% de los chunks.
La probabilidad de que muchas de las 50 consultas dependan de ellos es baja. El
valor de este módulo está en la cobertura completa del corpus y en el informe
técnico, no en mover las métricas por sí solo. Si sobra tiempo después de los
pendientes, el mejor uso no es pulir más los mapas: es ayudar con el dev set
etiquetado, que es lo que de verdad mueve NDCG@10 y F1@3.
