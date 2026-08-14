"""
construir_grafo.py

Construye el grafo de conocimiento (Sección 7 del documento CODEFEST)
a partir de los chunks de metadata.jsonl: extrae entidades (NER) y
relaciones (RE) de cada fragmento, y arma un grafo dirigido donde cada
entidad queda vinculada a los chunk_id que la mencionan (Sección 7.3),
para dar evidencia complementaria a la similitud vectorial en la
recuperación (Sección 8.5).

Modelos usados y por qué:
  - NER: GLiNER multilingüe (zero-shot). Es un modelo encoder (no
    decoder/generativo), así que es seguro reutilizarlo también del
    lado de la consulta en retrieval sin violar la restricción de la
    Sección 8.3. Al ser zero-shot, no hay que reentrenarlo para las
    entidades específicas del reto (sistemas de armas, satélites,
    tratados, actores territoriales, etc.) -- basta con darle la lista
    de tipos de entidad (TIPOS_ENTIDAD más abajo).
  - RE: mREBEL (Babelscape/mrebel-large). Es de los pocos modelos
    open-source que hacen extracción de relaciones de extremo a
    extremo (no requiere que le des pares de entidades ya armados) y
    que soporta español/inglés/portugués -- coincide con el requisito
    multilingüe de la Sección 4.3. Se usa SOLO aquí, en la construcción
    del grafo (indexación); la Sección 8.3 prohíbe modelos generativos
    únicamente "en el proceso de recuperación", y esto es preprocesamiento,
    no retrieval, así que no aplica la restricción.

Alternativas descartadas (y por qué):
  - REBEL (no mREBEL): solo inglés, no sirve para un corpus en
    es/en/pt.
  - Heurísticas de dependencias sintácticas (spaCy DependencyMatcher):
    mucho menos robustas en texto libre y en tres idiomas a la vez;
    requieren reglas por idioma.
  - Un LLM genérico para NER+RE: prohibido explícitamente para
    retrieval, y aunque se usara solo en indexación, complica la
    reproducibilidad exacta que exige generador.py (Sección 1.4,
    punto 4) por la naturaleza no determinista de un LLM.

Instalación:
    pip install gliner transformers torch networkx sentencepiece langdetect --break-system-packages
    (langdetect es nueva: se usa para detectar el idioma de cada chunk
    y configurar mREBEL correctamente antes de generar -- ver
    detectar_idioma_mbart())

Uso:
    python construir_grafo.py
    (pide la ruta de metadata.jsonl; genera grafo/grafo.graphml junto a él)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx

# Tipos de entidad para el NER zero-shot (GLiNER). Derivados de las 50
# preguntas de evaluación reales (Extracto_Preguntas_50_v2.pdf),
# agrupadas por fenómeno:
#
#   F1 (q001-q016, IA militar): país, organización, tecnología, sistema
#   de armas, instrumento legal, amenaza/vulnerabilidad, infraestructura
#   crítica.
#   F2 (q017-q032, seguridad espacial): país, satélite, capacidad
#   contraespacial (spoofing, guerra electrónica, láser, energía
#   dirigida), órbita, evento/conflicto.
#   F3 (q033-q050, dinámicas territoriales): grupo armado (aparecen las
#   siglas GAO/GAOR/GDO explícitamente en las preguntas), economía
#   ilícita, recurso natural/mineral, ubicación (departamentos
#   colombianos nombrados: Chocó, Antioquia, Bolívar, Norte de
#   Santander, Arauca, Córdoba, Cauca), población afectada, institución
#   del Estado.
#
# Debe ser EXACTAMENTE la misma lista en construir_grafo.py y
# consulta_grafo.py.
TIPOS_ENTIDAD = [
    "país",
    "organización",
    "grupo armado",
    "institución del Estado",
    "tecnología",
    "sistema de armas",
    "satélite",
    "capacidad contraespacial",
    "órbita",
    "instrumento legal",
    "amenaza o vulnerabilidad",
    "infraestructura crítica",
    "evento o conflicto",
    "economía ilícita",
    "recurso natural o mineral",
    "ubicación",
    "población afectada",
    "riesgo",
    "dificultad",
    "ventajas",
    "inteligencia artificial",
    "evidencia",
    "explotación",
]

UMBRAL_NER = 0.5  # score mínimo de confianza para aceptar una entidad detectada

# mREBEL está basado en mBART-50, que exige indicar explícitamente el
# idioma de ENTRADA con estos códigos antes de generar. Sin esto (o con
# el idioma equivocado), el modelo genera basura repetitiva en vez de
# tripletas -- justo lo que pasaba antes de este fix.
MAPA_IDIOMA_MBART = {"es": "es_XX", "en": "en_XX", "pt": "pt_XX"}
IDIOMA_MBART_DEFAULT = "en_XX"

# Token especial que le dice a mREBEL "generá la salida en formato de
# tripletas" (el equivalente al idioma de SALIDA en un modelo de
# traducción). Es el otro ingrediente que faltaba.
TOKEN_FORMATO_TRIPLETA = "tp_XX"


def detectar_idioma_mbart(texto: str) -> str:
    """
    Detecta el idioma del texto (es/en/pt) y lo traduce al código que
    espera mBART/mREBEL. Si no se puede detectar (texto muy corto,
    librería no instalada, etc.) usa un idioma por defecto en vez de
    fallar.
    """
    try:
        from langdetect import detect
        codigo = detect(texto)[:2]
        return MAPA_IDIOMA_MBART.get(codigo, IDIOMA_MBART_DEFAULT)
    except Exception:
        return IDIOMA_MBART_DEFAULT


def cargar_modelos():
    """Carga GLiNER (NER) y mREBEL (RE). Se hace una sola vez por ejecución."""
    from gliner import GLiNER
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print("Cargando GLiNER (NER)...")
    ner_modelo = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

    print("Cargando mREBEL (RE)...")
    re_tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large")
    re_modelo = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")

    return ner_modelo, re_tokenizer, re_modelo


def extraer_entidades(texto: str, ner_modelo) -> list[dict]:
    """Devuelve las entidades detectadas: [{'texto', 'tipo', 'score'}, ...]."""
    entidades = ner_modelo.predict_entities(texto, TIPOS_ENTIDAD, threshold=UMBRAL_NER)
    return [{"texto": e["text"], "tipo": e["label"], "score": e["score"]} for e in entidades]


def parsear_tripletas(texto_generado: str) -> list[dict]:
    """
    Parsea la salida de mREBEL.

    A diferencia del REBEL original (que usa los tokens FIJOS <subj> y
    <obj> como separadores), mREBEL los reemplaza por el TIPO de
    entidad que el propio modelo predijo para el sujeto y el objeto
    (<concept>, <per>, <loc>, <date>, <org>, etc. -- varía chunk a
    chunk). Confirmado empíricamente con debug_mrebel.py, ej.:

        <triplet> Wuhan <loc> China <loc> country

    El formato real es:
        <triplet> SUJETO <tipo_sujeto> OBJETO <tipo_objeto> RELACIÓN

    O sea: el primer tag "<...>" que aparece después de <triplet> cierra
    el sujeto (y da su tipo), el segundo cierra el objeto (y da su
    tipo), y lo que sigue hasta el próximo <triplet> es la relación.
    Aprovechamos esos tipos para no tener que marcar las entidades de
    RE como "desconocido" en el grafo.
    """
    triplets = []
    texto_generado = texto_generado.strip()
    texto_generado = (
        texto_generado.replace("<s>", "")
        .replace("<pad>", "")
        .replace("</s>", "")
        .replace(TOKEN_FORMATO_TRIPLETA, "")
    )

    sujeto, objeto, relacion = "", "", ""
    tipo_sujeto, tipo_objeto = "", ""
    fase = None  # "sujeto" | "objeto" | "relacion"

    def _guardar():
        if sujeto and objeto and relacion:
            triplets.append({
                "sujeto": sujeto.strip(),
                "tipo_sujeto": tipo_sujeto,
                "objeto": objeto.strip(),
                "tipo_objeto": tipo_objeto,
                "relacion": relacion.strip(),
            })

    for token in texto_generado.split():
        if token == "<triplet>":
            _guardar()
            sujeto, objeto, relacion = "", "", ""
            tipo_sujeto, tipo_objeto = "", ""
            fase = "sujeto"
        elif token.startswith("<") and token.endswith(">"):
            # cualquier otro tag es un marcador de tipo (no literal
            # <subj>/<obj>), y cierra el segmento actual
            if fase == "sujeto":
                tipo_sujeto = token.strip("<>")
                fase = "objeto"
            elif fase == "objeto":
                tipo_objeto = token.strip("<>")
                fase = "relacion"
            # un tag en fase "relacion" (o antes del primer <triplet>) se ignora
        else:
            if fase == "sujeto":
                sujeto += " " + token
            elif fase == "objeto":
                objeto += " " + token
            elif fase == "relacion":
                relacion += " " + token

    _guardar()
    return triplets


def extraer_relaciones(texto: str, re_tokenizer, re_modelo) -> list[dict]:
    """Genera y parsea las tripletas sujeto-relación-objeto de un chunk con mREBEL."""
    import torch

    # 0) mREBEL (mBART-50) necesita saber en qué idioma viene el texto
    #    ANTES de tokenizar. Sin esto, el modelo genera basura repetitiva
    #    en vez de tripletas (es justo lo que pasaba sin este paso).
    re_tokenizer.src_lang = detectar_idioma_mbart(texto)

    # 1) El tokenizer convierte el texto (palabras) en números (ids de
    #    token), que es lo único que entiende el modelo.
    entradas = re_tokenizer(texto, return_tensors="pt", truncation=True, max_length=256)

    # 2) torch.no_grad() apaga el cálculo de gradientes de PyTorch.
    #    Los gradientes solo hacen falta para ENTRENAR un modelo.
    with torch.no_grad():
        # 3) .generate() es el método que hace que el modelo produzca
        #    texto nuevo token por token (a diferencia de un forward()
        #    normal, que solo daría una predicción). num_beams=3 usa
        #    "beam search": en vez de quedarse siempre con el token más
        #    probable en cada paso (lo que puede llevar a una secuencia
        #    subóptima), explora 3 caminos alternativos en paralelo y al
        #    final se queda con el más probable en conjunto -- da
        #    resultados algo mejores que la opción más simple (greedy),
        #    a cambio de un poco más de cómputo. max_length=256 limita
        #    cuántos tokens como máximo puede generar de salida.
        #    forced_bos_token_id es el otro ingrediente que faltaba: le
        #    dice al decoder "generá en formato de tripletas" (el token
        #    tp_XX), en vez de dejarlo arrancar sin instrucción clara
        #    de qué "modo" de salida usar.
        salida = re_modelo.generate(
            **entradas,
            max_length=256,
            num_beams=3,
            forced_bos_token_id=re_tokenizer.convert_tokens_to_ids(TOKEN_FORMATO_TRIPLETA),
        )

    # 4) La salida de .generate() son también números (ids de token), no
    #    texto. .decode() hace el camino inverso al tokenizer: convierte
    #    esos ids de vuelta a texto legible. skip_special_tokens=False
    #    deja adentro los marcadores como <s>, <pad>, </s>, <triplet> y
    #    los tags de tipo de entidad (<loc>, <per>, <concept>, ...) --
    #    los necesitamos para poder parsear la estructura de las
    #    tripletas en el siguiente paso.
    texto_generado = re_tokenizer.decode(salida[0], skip_special_tokens=False)

    # 5) Convierte esa cadena cruda (con los tokens especiales) en una
    #    lista de diccionarios {sujeto, relacion, objeto} ya limpios.
    return parsear_tripletas(texto_generado)


def normalizar_entidad(texto: str) -> str:
    """Normaliza el texto de una entidad para usarlo como id de nodo del grafo."""
    return re.sub(r"\s+", " ", texto).strip().lower()


def _agregar_nodo(grafo: nx.DiGraph, nodo_id: str, tipo: str, texto_original: str, chunk_id: str, doc_id: str):
    """
    Agrega un nodo al grafo si no existe todavía, y en cualquier caso
    le suma el chunk_id y doc_id actuales a sus conjuntos de evidencia.

    Si el nodo ya existía (porque otra entidad, en otro chunk, o el
    NER y el RE por separado, ya lo habían creado), NO se sobreescribe
    su 'tipo' ni su 'texto' original -- se conserva el del primer
    origen que lo creó. Esto significa que el tipo final de un nodo
    depende del orden de procesamiento (¿lo creó primero GLiNER o
    mREBEL?), no de cuál de los dos es más confiable.
    """
    if nodo_id not in grafo:
        grafo.add_node(nodo_id, tipo=tipo, texto=texto_original, chunk_ids=set(), doc_ids=set())
    grafo.nodes[nodo_id]["chunk_ids"].add(chunk_id)
    grafo.nodes[nodo_id]["doc_ids"].add(doc_id)


def construir_grafo(chunks: list[dict], ner_modelo, re_tokenizer, re_modelo) -> nx.DiGraph:
    """
    Recorre los chunks, extrae entidades y relaciones de cada uno, y
    arma el grafo dirigido. Cada nodo (entidad) y cada arista (relación)
    guarda el conjunto de chunk_id que la mencionan/evidencian.

    Dos fuentes de nodos por chunk:
      - GLiNER (extraer_entidades): aporta nodos, pero NUNCA aristas
        -- una entidad detectada solo por NER queda sin conexión a
        menos que también participe como sujeto u objeto de alguna
        tripleta de mREBEL (en este chunk o en otro).
      - mREBEL (extraer_relaciones): aporta pares de nodos (sujeto y
        objeto) MÁS la arista dirigida entre ellos, con la relación y
        el chunk que la evidencia.
    """
    grafo = nx.DiGraph()

    for i, chunk in enumerate(chunks):
        texto = chunk["texto"]
        chunk_id = chunk["chunk_id"]
        doc_id = chunk["doc_id"]

        for ent in extraer_entidades(texto, ner_modelo):
            nodo_id = normalizar_entidad(ent["texto"])
            if nodo_id:
                _agregar_nodo(grafo, nodo_id, ent["tipo"], ent["texto"], chunk_id, doc_id)

        for t in extraer_relaciones(texto, re_tokenizer, re_modelo):
            sujeto_id = normalizar_entidad(t["sujeto"])
            objeto_id = normalizar_entidad(t["objeto"])
            if not sujeto_id or not objeto_id:
                continue

            # mREBEL ya nos da el tipo de sujeto/objeto que predijo (ver
            # parsear_tripletas) -- lo usamos en vez de "desconocido"
            _agregar_nodo(grafo, sujeto_id, t["tipo_sujeto"] or "desconocido", t["sujeto"], chunk_id, doc_id)
            _agregar_nodo(grafo, objeto_id, t["tipo_objeto"] or "desconocido", t["objeto"], chunk_id, doc_id)

            if grafo.has_edge(sujeto_id, objeto_id):
                grafo[sujeto_id][objeto_id]["chunk_ids"].add(chunk_id)
            else:
                grafo.add_edge(sujeto_id, objeto_id, relacion=t["relacion"], chunk_ids={chunk_id})

        if (i + 1) % 200 == 0:
            print(f"  procesados {i + 1}/{len(chunks)} chunks...")

    return grafo


def serializar_para_graphml(grafo: nx.DiGraph) -> nx.DiGraph:
    """GraphML no admite sets como atributos: se convierten a texto separado por '|'."""
    for _, datos in grafo.nodes(data=True):
        datos["chunk_ids"] = "|".join(sorted(datos.get("chunk_ids", set())))
        datos["doc_ids"] = "|".join(sorted(datos.get("doc_ids", set())))
    for _, _, datos in grafo.edges(data=True):
        datos["chunk_ids"] = "|".join(sorted(datos.get("chunk_ids", set())))
    return grafo


def cargar_chunks_de_archivo(ruta_metadata: Path) -> list[dict]:
    """Lee un .jsonl línea por línea y devuelve la lista de chunks
    (cada línea es un dict con doc_id, chunk_id, texto, etc.)."""
    chunks = []
    with open(ruta_metadata, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                chunks.append(json.loads(linea))
    return chunks


def cargar_chunks(ruta_metadata: Path) -> list[dict]:
    """
    Carga los chunks a procesar, aceptando dos casos:

      - ruta_metadata es un ARCHIVO .jsonl: se lee ese único archivo
        (comportamiento original).
      - ruta_metadata es una CARPETA: se buscan recursivamente todos
        los .jsonl que contenga (ej. una carpeta "metadata" con el
        metadata.jsonl de cada integrante del equipo, ya sea sueltos
        ahí mismo o cada uno en su propia subcarpeta) y se juntan
        todos sus chunks en una sola lista, para construir un único
        grafo combinado con el corpus completo del equipo.

    No se deduplican chunk_id entre archivos: si dos integrantes
    generaron IDs iguales por coincidencia, ambos chunks se procesan
    igual (el chunk_id sigue sirviendo solo para trazabilidad interna,
    no es la clave de emparejamiento con el ground truth -- ver
    Sección 10.2.1 del documento CODEFEST).
    """
    if ruta_metadata.is_dir():
        rutas_jsonl = sorted(ruta_metadata.rglob("*.jsonl"))
        if not rutas_jsonl:
            print(f"Advertencia: no se encontró ningún .jsonl dentro de '{ruta_metadata}'")
            return []

        chunks = []
        for ruta in rutas_jsonl:
            chunks_archivo = cargar_chunks_de_archivo(ruta)
            print(f"  {ruta.relative_to(ruta_metadata)}: {len(chunks_archivo)} chunks")
            chunks.extend(chunks_archivo)
        return chunks

    return cargar_chunks_de_archivo(ruta_metadata)


def main():
    """CLI interactiva: pide la ruta de metadata.jsonl (o de una
    carpeta con varios .jsonl, ej. los de todo el equipo), procesa
    todos los chunks encontrados y escribe grafo/grafo.graphml junto a
    esa ruta."""
    ruta_metadata = input(
        "Ruta de metadata.jsonl, o de una carpeta con varios .jsonl (ej. 'metadata'): "
    ).strip()
    ruta_metadata = Path(ruta_metadata) if ruta_metadata else Path("metadata.jsonl")
    if not ruta_metadata.exists():
        print(f"Error: no se encontró '{ruta_metadata}'")
        return

    print("Cargando chunks...")
    chunks = cargar_chunks(ruta_metadata)
    print(f"{len(chunks)} chunks a procesar en total.")
    if not chunks:
        return

    ner_modelo, re_tokenizer, re_modelo = cargar_modelos()

    print("Construyendo grafo (NER + RE por chunk)...")
    grafo = construir_grafo(chunks, ner_modelo, re_tokenizer, re_modelo)
    grafo = serializar_para_graphml(grafo)

    carpeta_base = ruta_metadata if ruta_metadata.is_dir() else ruta_metadata.parent
    carpeta_grafo = carpeta_base / "grafo"
    carpeta_grafo.mkdir(parents=True, exist_ok=True)
    ruta_grafo = carpeta_grafo / "grafo.graphml"

    nx.write_graphml(grafo, ruta_grafo)

    print(f"\nListo: {grafo.number_of_nodes()} entidades, {grafo.number_of_edges()} relaciones -> {ruta_grafo}")


if __name__ == "__main__":
    main()