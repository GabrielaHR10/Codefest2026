#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generador.py — CODEFEST AD ASTRA 2026, Etapa 1

Lee el archivo de consultas, interroga la base de conocimiento (los dos índices
FAISS más el grafo de conocimiento) y escribe ``resultados.jsonl`` con el
formato de la Sección 9 del documento de lineamientos: una línea por consulta,
3 documentos y 10 fragmentos de máximo 250 palabras cada uno.

    python entrega/generador.py

Sin argumentos usa las rutas por defecto relativas a este archivo:

    consultas   -> entrega/consultas.txt
    índices     -> entrega/base_vectorial/encoder_*/
    grafo       -> entrega/grafo/grafo.graphml
    salida      -> entrega/resultados.jsonl


Cómo se recupera
----------------
El corpus quedó repartido entre dos encoders complementarios (Sección 4.4):

    BAAI/bge-m3                    xlsx del AI Index + mapas .pbf   (9.886 frags)
    intfloat/multilingual-e5-base  pdf de observatorios + json web  (79.178 frags)

Como cada fragmento vive en uno solo de los dos espacios vectoriales, las
puntuaciones de coseno de un índice y del otro no son directamente comparables
(cada encoder tiene su propia escala de similitud). En vez de fusionar rangos a
ciegas, el módulo usa los dos índices como fuentes de *recall* y después
**reproyecta todos los candidatos a un único espacio común** (el de e5) para
puntuarlos con la misma vara:

    1. Recuperación:  top-k de FAISS/e5 + top-k de FAISS/bge-m3 + evidencia del
       grafo de conocimiento (Sección 8.5). La unión forma el pool de candidatos.
    2. Calibración:   cada candidato se puntúa por similitud coseno contra el
       vector de la consulta en el espacio de e5. Para los fragmentos que ya
       están en ese índice el vector se lee del propio índice (exacto, sin
       recodificar); los que vienen de bge-m3 se codifican con e5.
    3. Refinamiento:  de los mejores candidatos se elige la ventana de <= 250
       palabras (cortada en límites de oración) que más se parece a la consulta.
       Ese es el texto que se entrega y la puntuación con la que se ordena, que
       es justo lo que evalúa NDCG@10 (la relevancia se juzga sobre el texto).
    4. Fusión:        puntuación final = z(coseno) + PESO_GRAFO * evidencia del
       grafo normalizada. El grafo aporta tanto candidatos nuevos como refuerzo
       a los que la vía vectorial ya había encontrado.
    5. Agregación:    los fragmentos se agrupan por documento (max-pooling más
       una bonificación por fragmentos adicionales) y se entregan los 3 mejores.

Todo el proceso opera únicamente sobre vectores, puntuaciones de similitud y
metadata. No interviene ningún modelo generativo/decoder en ninguna etapa, como
exige la Sección 8.3: los dos modelos usados son encoders (familia BERT/XLM-R) y
el grafo se consulta con búsqueda exacta sobre su vocabulario de entidades.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Orden de importación: torch/sentence-transformers ANTES que faiss.
#
# En macOS las ruedas de faiss-cpu y de torch traen cada una su propia copia de
# libomp; si faiss se carga primero, el proceso muere sin traza al construir el
# encoder. Importar torch primero deja una sola runtime de OpenMP activa. La
# variable de entorno es el cinturón de seguridad para el caso contrario.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

import numpy as np  # noqa: E402
import faiss  # noqa: E402

# Segunda mitad del mismo problema: aunque torch se cargue primero, la búsqueda
# multihilo de faiss revienta el proceso con SIGSEGV en macOS cuando la runtime
# de OpenMP de torch ya está inicializada. Con un solo hilo no ocurre, y el
# coste es irrelevante: la búsqueda exacta sobre 79.178 vectores de 768d tarda
# ~10 ms. En Linux se deja el paralelismo por defecto.
if sys.platform == "darwin":
    faiss.omp_set_num_threads(1)


# ===========================================================================
# Configuración
# ===========================================================================

RAIZ = Path(__file__).resolve().parent

# Un bloque por índice FAISS de la entrega. `prefijo_pasaje` y `prefijo_consulta`
# reproducen exactamente la convención con la que se codificó cada base: e5 fue
# entrenado con los prefijos "query: " / "passage: " y omitirlos degrada la
# recuperación; bge-m3 no lleva prefijo.
INDICES = [
    {
        "carpeta": "encoder_multilingual-e5-base",
        "modelo": "intfloat/multilingual-e5-base",
        "prefijo_consulta": "query: ",
        "prefijo_pasaje": "passage: ",
        "top_k": 200,
        "espacio_comun": True,  # este es el espacio en el que se calibra todo
    },
    {
        "carpeta": "encoder_bge-m3",
        "modelo": "BAAI/bge-m3",
        "prefijo_consulta": "",
        "prefijo_pasaje": "",
        "top_k": 100,
        "espacio_comun": False,
    },
]

N_DOCUMENTOS = 3        # documentos por consulta (Sección 9.2)
N_FRAGMENTOS = 10       # fragmentos por consulta (Sección 9.2)
MAX_PALABRAS = 250      # límite duro por fragmento (Sección 9.2.1)

N_REFINAR = 60          # candidatos a los que se les busca la mejor ventana
MAX_FRAG_POR_DOC = 5    # tope de fragmentos del mismo documento en el top-10

PESO_GRAFO = 0.5        # peso de la evidencia del grafo frente a z(coseno)
BONIF_DOC = 0.30        # peso de los fragmentos secundarios al puntuar un documento

# --- parámetros de la consulta al grafo -----------------------------------
MAX_NGRAMA = 6          # longitud máxima (en palabras) de una entidad buscada
MIN_CARACTERES = 4      # entidades más cortas que esto son ruido
DF_MAXIMA_REL = 0.02    # un nodo presente en >2% del corpus no discrimina nada
DF_SATURACION = 5       # un nodo detectado 1 o 2 veces no es "muy específico"
DF_UMBRAL_INFO = 2000   # una entidad debe ser al menos así de rara en el corpus
DF_MAX_VECINO = 200     # no se propaga desde vecinos que son "hubs"
PESO_VECINO = 0.15      # las menciones de un vecino valen menos que las propias
MAX_VECINOS = 25        # vecinos de primer orden por entidad de la consulta
BONIF_COBERTURA = 0.40  # premio por tocar varias entidades distintas

# Palabras funcionales de los tres idiomas del corpus. Se usan para descartar
# n-gramas que no nombran ninguna entidad ("de la", "how the", "para os").
VACIAS = {
    "a", "al", "ante", "como", "con", "contra", "cual", "cuales", "cuando", "cuanto",
    "de", "del", "desde", "donde", "dos", "e", "el", "ella", "ellas", "ellos", "en",
    "entre", "era", "es", "esa", "ese", "eso", "esta", "estan", "este", "esto", "estos",
    "ha", "han", "hasta", "hay", "la", "las", "le", "les", "lo", "los", "mas", "me",
    "mi", "mismo", "mucho", "muy", "no", "o", "os", "otra", "otro", "para", "pero",
    "poco", "por", "porque", "que", "quee", "quien", "se", "segun", "ser", "si", "sin",
    "sobre", "son", "su", "sus", "tal", "tan", "te", "tiene", "todo", "todos", "tras",
    "un", "una", "uno", "unos", "y", "ya",
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "them", "there", "these", "they", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with",
    "das", "dos", "em", "nao", "num", "numa", "ou", "pelo", "pela", "sao", "uma",
    # verbos y sustantivos de encuadre que aparecen en casi toda pregunta
    "cómo", "como", "qué", "que", "cuál", "cuáles", "manera", "forma", "principales",
    "recientes", "actualmente", "existen", "representan", "implica", "implican",
}

NS = "{http://graphml.graphdrawing.org/xmlns}"


# ===========================================================================
# Utilidades de texto
# ===========================================================================

_RE_ESPACIOS = re.compile(r"\s+")
_RE_NO_ALFANUM = re.compile(r"[^\w\s]", re.UNICODE)


def normalizar(texto: str) -> str:
    """Minúsculas, sin puntuación y con espacios colapsados.

    Se usa tanto sobre los identificadores de nodo del grafo como sobre las
    consultas, de modo que ambos lados se comparen en la misma forma. No se
    quitan las tildes: los nodos del grafo las conservan y quitarlas en un solo
    lado rompería la coincidencia.
    """
    texto = unicodedata.normalize("NFC", texto)
    texto = _RE_NO_ALFANUM.sub(" ", texto.lower())
    return _RE_ESPACIOS.sub(" ", texto).strip()


def palabras(texto: str) -> list[str]:
    """Tokenización por espacios: es la misma unidad con la que el documento
    de lineamientos cuenta el límite de 250 palabras por fragmento."""
    return texto.split()


_RE_FIN_ORACION = re.compile(
    r"""
    (?<=[.!?…])        # el corte va después de un signo de cierre
    ["'”’)\]]*         # comillas o paréntesis que cierren detrás del signo
    \s+                # el espacio que separa de la siguiente oración
    (?=[¿¡"'“(\[]*[A-ZÁÉÍÓÚÑÜ0-9])   # la siguiente empieza en mayúscula o cifra
    """,
    re.VERBOSE,
)

# Abreviaturas frecuentes en el corpus tras las que un punto NO cierra oración.
_ABREVIATURAS = re.compile(
    r"\b(?:ee\.?\s?uu|ee|uu|dr|dra|sr|sra|ing|prof|fig|no|núm|num|vol|pág|pag|art|"
    r"aprox|etc|ca|cf|vs|op|cit|u\.s|u\.k|st|mr|mrs|ms|jr|sr)\.$",
    re.IGNORECASE,
)


def dividir_en_oraciones(texto: str) -> list[str]:
    """
    Parte un texto en oraciones completas.

    El requisito de completitud lingüística (Sección 3.3) prohíbe entregar
    oraciones cortadas, así que este es el único punto por el que se permite
    cortar un fragmento largo. El corte se hace en signo de cierre seguido de
    mayúscula, y se rechaza si lo que precede al punto es una abreviatura
    conocida (ahí el punto no cierra nada).
    """
    piezas = _RE_FIN_ORACION.split(texto.strip())
    oraciones: list[str] = []
    for pieza in piezas:
        pieza = pieza.strip()
        if not pieza:
            continue
        if oraciones and _ABREVIATURAS.search(oraciones[-1]):
            oraciones[-1] = oraciones[-1] + " " + pieza
        else:
            oraciones.append(pieza)
    return oraciones or [texto.strip()]


_RE_INICIO_ORACION = re.compile(r'^["\'“¿¡(\[]*[A-ZÁÉÍÓÚÑÜ0-9]')
_RE_FINAL_ORACION = re.compile(r'[.!?…]["\'”’)\]]*$')

# Si recortar los bordes se comiera más de esta fracción del fragmento, se deja
# el texto tal cual: pasa con las filas de tablas y las fichas de municipio, que
# no tienen forma de oración y desaparecerían enteras.
MIN_CONSERVADO = 0.6


def recortar_a_oraciones(texto: str) -> str:
    """
    Quita del fragmento un arranque y un final que no sean oraciones completas.

    Los fragmentos de la partición de PDF se cortaron con una ventana deslizante
    de tamaño fijo, así que muchos empiezan a mitad de una frase y terminan a
    mitad de otra. La Sección 3.3 prohíbe entregar oraciones incompletas, y
    además el texto recortado se lee mejor —que no es un detalle menor cuando la
    relevancia se juzga precisamente sobre ese texto (Sección 10.2.1)—.
    """
    texto = texto.strip()
    oraciones = dividir_en_oraciones(texto)
    if len(oraciones) < 2:
        return texto

    inicio = 0 if _RE_INICIO_ORACION.match(oraciones[0]) else 1
    fin = len(oraciones)
    if not _RE_FINAL_ORACION.search(oraciones[-1]) and fin - inicio > 1:
        fin -= 1
    if inicio >= fin:
        return texto

    recorte = " ".join(oraciones[inicio:fin]).strip()
    if len(palabras(recorte)) < MIN_CONSERVADO * len(palabras(texto)):
        return texto
    return recorte


def ventanas_de_250(texto: str, maximo: int = MAX_PALABRAS) -> list[str]:
    """
    Devuelve las ventanas de como máximo ``maximo`` palabras en que se puede
    entregar un fragmento, cortando siempre entre oraciones (Sección 9.2.1).

    Si el fragmento ya cabe, devuelve una sola ventana con el texto intacto.
    Las ventanas se solapan en una oración para que ninguna idea quede partida
    justo en la frontera. El caso degenerado —una única oración de más de 250
    palabras, que ninguna segmentación puede respetar— se recorta por palabras;
    ocurre solo en tablas mal extraídas de un PDF.
    """
    if len(palabras(texto)) <= maximo:
        return [texto.strip()]

    oraciones = dividir_en_oraciones(texto)
    ventanas: list[str] = []
    i = 0
    while i < len(oraciones):
        acumulado: list[str] = []
        total = 0
        j = i
        while j < len(oraciones):
            n = len(palabras(oraciones[j]))
            if acumulado and total + n > maximo:
                break
            if not acumulado and n > maximo:  # oración monstruosa: se recorta
                acumulado.append(" ".join(palabras(oraciones[j])[:maximo]))
                total = maximo
                j += 1
                break
            acumulado.append(oraciones[j])
            total += n
            j += 1
        ventanas.append(" ".join(acumulado))
        if j >= len(oraciones):
            break
        i = j - 1 if j - 1 > i else j  # una oración de solape
    return ventanas


def recortar_a_250(texto: str) -> str:
    """Red de seguridad: garantiza el límite duro de 250 palabras justo antes
    de escribir el resultado, por si alguna ventana se coló más larga."""
    tokens = palabras(texto)
    if len(tokens) <= MAX_PALABRAS:
        return texto.strip()
    oraciones = dividir_en_oraciones(texto)
    acumulado: list[str] = []
    total = 0
    for oracion in oraciones:
        n = len(palabras(oracion))
        if acumulado and total + n > MAX_PALABRAS:
            break
        acumulado.append(oracion)
        total += n
    if not acumulado:
        return " ".join(tokens[:MAX_PALABRAS])
    return " ".join(acumulado)


# ===========================================================================
# Base vectorial
# ===========================================================================

class BaseVectorial:
    """Un índice FAISS con su almacén de metadata alineado línea a línea."""

    def __init__(self, carpeta: Path, config: dict):
        self.nombre = config["carpeta"]
        self.config = config
        self.indice = faiss.read_index(str(carpeta / "index.faiss"))
        self.metadata: list[dict] = []
        with (carpeta / "metadata.jsonl").open(encoding="utf-8") as f:
            for linea in f:
                self.metadata.append(json.loads(linea))
        if len(self.metadata) != self.indice.ntotal:
            raise SystemExit(
                f"{self.nombre}: {len(self.metadata)} líneas de metadata "
                f"para {self.indice.ntotal} vectores"
            )
        self.modelo: SentenceTransformer | None = None

    def cargar_modelo(self, dispositivo: str) -> SentenceTransformer:
        if self.modelo is None:
            self.modelo = SentenceTransformer(self.config["modelo"], device=dispositivo)
        return self.modelo

    def codificar(self, textos: list[str], prefijo: str, lote: int = 16) -> np.ndarray:
        """Codifica y normaliza a norma unitaria, para que el producto interno
        del índice sea exactamente la similitud coseno (Sección 8.2)."""
        vectores = self.modelo.encode(
            [prefijo + t for t in textos],
            batch_size=lote,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectores, dtype="float32")

    def vector_consulta(self, consulta: str) -> np.ndarray:
        return self.codificar([consulta], self.config["prefijo_consulta"])

    def vector_de(self, fila: int) -> np.ndarray:
        """Lee del índice el vector ya almacenado de un fragmento. Evita
        recodificar: es exactamente el mismo vector que se indexó."""
        return self.indice.reconstruct(int(fila)).astype("float32")

    def buscar(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.indice.ntotal)
        distancias, indices = self.indice.search(vector, k)
        return [(int(i), float(d)) for i, d in zip(indices[0], distancias[0]) if i >= 0]


# ===========================================================================
# Grafo de conocimiento (Sección 8.5)
# ===========================================================================

class GrafoConocimiento:
    """
    Consulta del grafo sin cargarlo entero en memoria.

    ``grafo.graphml`` pesa ~1 GB (226.821 entidades y 5,1 millones de
    relaciones); leerlo con ``networkx.read_graphml`` exige varios GB de RAM y
    minutos de espera. Como las 50 consultas se conocen de antemano, aquí se
    recorre el archivo **una sola vez** con un parser incremental y se acumula
    la evidencia de todas las consultas en esa misma pasada.

    El recorrido aprovecha que GraphML (tal como lo escribe NetworkX) emite
    primero todos los nodos y después todas las aristas: cuando aparece la
    primera arista ya se conoce el vocabulario completo de entidades, así que en
    ese punto se resuelve qué nodos menciona cada consulta y el resto del
    archivo solo tiene que acumular puntuaciones.

    Puntuación de un fragmento (Sección 8.5, punto 3):

        bruto(c) = Σ w(e)  sobre las entidades e de la consulta mencionadas en c
                 + 0.15 · Σ w(e)·w(v)  sobre los vecinos v de primer orden de e

                        bruto(c) · (1 + 0.4·(entidades_cubiertas(c) − 1))
        puntuación(c) = ────────────────────────────────────────────────────
                                    √(entidades_totales(c))

    con tres correcciones que se ganaron a base de mirar lo que devolvía la
    versión ingenua (sumar pesos y ordenar):

    * **w es un peso tipo IDF** y las entidades que aparecen en más del 2 % del
      corpus se descartan del todo. Una entidad como "colombia" o "inteligencia
      artificial" no discrimina nada y sí arrastra decenas de miles de
      fragmentos irrelevantes.
    * **el divisor √(entidades_totales)** compensa la densidad de entidades del
      fragmento. Sin él, los primeros puestos se los llevaban siempre las
      páginas de bibliografía y las tablas de datos: no son relevantes para
      nada, pero mencionan tantas entidades que acumulan puntuación por pura
      acumulación.
    * **el factor de cobertura** premia al fragmento que toca varias entidades
      distintas de la consulta frente al que repite una sola.
    """

    def __init__(self, ruta: Path, chunk_a_fila: dict[str, tuple[str, int]],
                 df_corpus: dict[str, int]):
        self.ruta = ruta
        self.chunk_a_fila = chunk_a_fila
        self.df_corpus = df_corpus
        self.total_chunks = max(len(chunk_a_fila), 1)
        self.umbral_peso = float(np.log1p(self.total_chunks / DF_UMBRAL_INFO))
        self._contenedor: ET.Element | None = None

    # -- pasada 1: vocabulario de entidades ---------------------------------

    def _leer_nodos(self, ctx) -> ET.Element | None:
        """Consume los nodos del archivo y deja montado el vocabulario.

        Devuelve el primer elemento ``edge`` encontrado: el iterador ya lo
        consumió y no puede volver a emitirlo, así que se procesa fuera.
        """
        claves: dict[tuple[str, str], str] = {}
        # nodo_id -> fila; las filas indexan chunks_ptr/chunks_dat (formato CSR)
        nodos: dict[str, int] = {}
        chunks_ptr: list[int] = [0]
        chunks_dat: list[int] = []
        # los chunk_id se internan a entero: hay 5,1 millones de referencias y
        # guardarlas como cadenas multiplicaría la memoria por veinte
        self.chunk_id_a_int: dict[str, int] = {}
        self.int_a_chunk_id: list[str] = []
        # forma normalizada (sin puntuación) -> filas de nodo que la comparten
        alias: dict[str, list[int]] = defaultdict(list)

        primera_arista = None
        for evento, el in ctx:
            if evento == "start":
                if el.tag == NS + "graph" and self._contenedor is None:
                    self._contenedor = el
                continue
            if el.tag == NS + "key":
                claves[(el.get("for"), el.get("attr.name"))] = el.get("id")
                continue
            if el.tag == NS + "edge":
                primera_arista = el
                break
            if el.tag != NS + "node":
                continue

            nodo_id = el.get("id")
            clave_chunks = claves.get(("node", "chunk_ids"))
            crudo = ""
            for dato in el:
                if dato.get("key") == clave_chunks:
                    crudo = dato.text or ""
                    break
            el.clear()

            if nodo_id in nodos:
                continue
            fila = len(nodos)
            nodos[nodo_id] = fila
            for chunk_id in crudo.split("|"):
                if chunk_id:
                    chunks_dat.append(self._interna(chunk_id))
            chunks_ptr.append(len(chunks_dat))
            alias[normalizar(nodo_id)].append(fila)

        self.claves = claves
        self.nodos = nodos
        self.alias = alias
        self.chunks_ptr = np.asarray(chunks_ptr, dtype=np.int64)
        self.chunks_dat = np.asarray(chunks_dat, dtype=np.int32)
        return primera_arista

    def _interna(self, chunk_id: str) -> int:
        entero = self.chunk_id_a_int.get(chunk_id)
        if entero is None:
            entero = len(self.int_a_chunk_id)
            self.chunk_id_a_int[chunk_id] = entero
            self.int_a_chunk_id.append(chunk_id)
        return entero

    def _df(self, fila: int) -> int:
        """Número de fragmentos en que aparece la entidad de esa fila."""
        return int(self.chunks_ptr[fila + 1] - self.chunks_ptr[fila])

    def _peso_nodo(self, fila: int) -> float:
        """
        Peso de una entidad a partir de en cuántos fragmentos la marcó el NER.

        El conteo se satura por abajo en ``DF_SATURACION``: sin ese tope, una
        entidad que el NER solo detectó una vez recibiría el peso máximo, que es
        justo al revés de lo que interesa. Una detección única no es señal de
        especificidad, es señal de que probablemente sea un artefacto de
        extracción.
        """
        df = self._df(fila)
        if df == 0 or df > DF_MAXIMA_REL * self.total_chunks:
            return 0.0
        return float(np.log1p(self.total_chunks / max(df, DF_SATURACION)))

    def _peso_consulta(self, frase: str, n_palabras: int) -> float:
        """
        Peso de una entidad de la consulta, medido por lo informativa que es su
        redacción **en el corpus**, no en el grafo.

        Esta es la corrección que más ruido quita. El vocabulario del grafo
        contiene sustantivos genéricos que el NER marcó como entidad en un par
        de fragmentos sueltos ("amenazas", "servicios", "pruebas", "impacto"):
        pesarlos por su frecuencia dentro del grafo les daba el peso máximo, y
        arrastraban a la respuesta fragmentos que no tenían nada que ver. Medida
        contra el corpus, su frecuencia real los delata: "servicios" aparece en
        miles de fragmentos y "spoofing" en unas decenas.

        Devuelve 0 si la entidad no supera el umbral de informatividad, en cuyo
        caso simplemente no se usa: para una consulta sin ninguna entidad
        distintiva, el grafo no aporta nada y la recuperación se queda con la
        vía vectorial, que es mejor que inventar evidencia.
        """
        fichas = frase.split()
        if not fichas:
            return 0.0
        idfs = [np.log1p(self.total_chunks / (1 + self.df_corpus.get(f, 0))) for f in fichas]
        peso = float(np.mean(idfs)) * (1.0 + 0.25 * (n_palabras - 1))
        return peso if peso >= self.umbral_peso else 0.0

    # -- emparejamiento consulta -> entidades -------------------------------

    def entidades_de(self, consulta: str) -> dict[int, float]:
        """
        Localiza en la consulta las entidades que existen como nodo del grafo.

        La búsqueda es por n-gramas contra el vocabulario de nodos, que es la
        forma exacta (y barata) de lo que hacía ``consultaGrafo.py`` recorriendo
        los 226.821 nodos con una expresión regular por nodo. El vocabulario
        contra el que se compara es el que produjo GLiNER al construir el grafo,
        de modo que el reconocimiento de entidades de la consulta y el del
        corpus comparten exactamente el mismo inventario (Sección 8.5, punto 1).

        Un n-grama de una sola palabra solo se acepta si la consulta lo escribe
        como nombre propio o como sigla ("Colombia", "GAOR", "LEO"). Es la
        diferencia entre las entidades que de verdad anclan una consulta y los
        sustantivos genéricos que el NER dejó en el grafo como nodos sueltos
        ("pruebas", "amenazas", "servicios", "impacto"): pesados por frecuencia
        son indistinguibles —el corpus es mayoritariamente inglés, así que
        cualquier palabra española resulta "rara"— pero la mayúscula del texto
        original sí los separa.
        """
        superficie = re.findall(r"\w+", unicodedata.normalize("NFC", consulta), re.UNICODE)
        fichas = [normalizar(f) for f in superficie]
        propias = [bool(re.search(r"[A-ZÁÉÍÓÚÑÜ0-9]", f)) for f in superficie]

        encontrados: dict[int, float] = {}
        for largo in range(min(MAX_NGRAMA, len(fichas)), 0, -1):
            for inicio in range(len(fichas) - largo + 1):
                grupo = fichas[inicio:inicio + largo]
                if not grupo[0] or not grupo[-1]:
                    continue
                if grupo[0] in VACIAS or grupo[-1] in VACIAS:
                    continue
                if largo == 1 and not propias[inicio]:
                    continue
                frase = " ".join(grupo)
                if len(frase) < MIN_CARACTERES:
                    continue
                filas = self.alias.get(frase)
                if not filas:
                    continue
                peso = self._peso_consulta(frase, largo)
                if peso <= 0:
                    continue
                for fila in filas:
                    if self._peso_nodo(fila) > 0:  # descarta los nodos-cajón
                        encontrados[fila] = max(encontrados.get(fila, 0.0), peso)
        return encontrados

    # -- pasada 2: aristas y acumulación ------------------------------------

    def evidencia(self, consultas: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
        """
        Devuelve, por consulta, un diccionario ``chunk_id -> puntuación`` con la
        evidencia que aporta el grafo.
        """
        inicio = time.time()
        ctx = ET.iterparse(str(self.ruta), events=("start", "end"))
        arista = self._leer_nodos(ctx)
        claves = self.claves
        print(f"  grafo: {len(self.nodos)} entidades leídas "
              f"({time.time() - inicio:.0f}s)", flush=True)

        # densidad de entidades de cada fragmento: cuántas entidades distintas
        # del grafo lo mencionan. Es el divisor que impide que una página de
        # bibliografía gane por acumulación.
        densidad = np.bincount(self.chunks_dat, minlength=len(self.int_a_chunk_id))
        densidad = np.maximum(densidad, 1)

        # entidades de cada consulta y semilla de puntuación por mención directa
        bruto: list[dict[int, float]] = []      # por consulta: chunk_int -> peso acumulado
        cobertura: list[dict[int, int]] = []    # por consulta: chunk_int -> entidades distintas
        entidades_por_consulta: list[dict[int, float]] = []
        pesos_por_nodo: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for q, (_, texto) in enumerate(consultas):
            acumulador: dict[int, float] = defaultdict(float)
            cuenta: dict[int, int] = defaultdict(int)
            entidades = self.entidades_de(texto)
            entidades_por_consulta.append(entidades)
            for fila, peso in entidades.items():
                pesos_por_nodo[fila].append((q, peso))
                ini, fin = self.chunks_ptr[fila], self.chunks_ptr[fila + 1]
                for chunk_int in self.chunks_dat[ini:fin]:
                    acumulador[int(chunk_int)] += peso
                    cuenta[int(chunk_int)] += 1
            bruto.append(acumulador)
            cobertura.append(cuenta)

        # Vecinos de primer orden. En vez de expandir hacia todos (hay entidades
        # con decenas de miles de aristas, y expandir a ciegas es justo lo que
        # llenaba el resultado de ruido), de cada entidad de la consulta se
        # guardan solo los MAX_VECINOS vecinos mejor respaldados: los que
        # comparten más fragmentos con ella y son a su vez específicos.
        mejores: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
        clave_chunks_arista = claves.get(("edge", "chunk_ids"))
        n_aristas = 0

        def procesar(el: ET.Element) -> None:
            nonlocal n_aristas
            n_aristas += 1
            fila_o = self.nodos.get(el.get("source"))
            fila_d = self.nodos.get(el.get("target"))
            if fila_o is None or fila_d is None:
                el.clear()
                return
            tocado_o = pesos_por_nodo.get(fila_o)
            tocado_d = pesos_por_nodo.get(fila_d)
            if not tocado_o and not tocado_d:
                el.clear()
                return

            chunks_arista = ""
            for dato in el:
                if dato.get("key") == clave_chunks_arista:
                    chunks_arista = dato.text or ""
                    break
            el.clear()
            apoyo = chunks_arista.count("|") + 1 if chunks_arista else 0

            for fila_propia, fila_vecina, tocados in (
                (fila_o, fila_d, tocado_o),
                (fila_d, fila_o, tocado_d),
            ):
                if not tocados or apoyo == 0:
                    continue
                if self._df(fila_vecina) > DF_MAX_VECINO:
                    continue
                peso_vecino = self._peso_nodo(fila_vecina)
                if peso_vecino <= 0:
                    continue
                fuerza = apoyo * peso_vecino
                for q, _peso in tocados:
                    monton = mejores[(q, fila_propia)]
                    if len(monton) < MAX_VECINOS:
                        heapq.heappush(monton, (fuerza, fila_vecina))
                    elif fuerza > monton[0][0]:
                        heapq.heapreplace(monton, (fuerza, fila_vecina))

        if arista is not None:
            procesar(arista)
        for evento, el in ctx:
            if evento == "start" or el.tag != NS + "edge":
                # los <data> no se tocan aquí: son hijos de la arista y hay que
                # leerlos antes de vaciarla, no antes de que llegue su cierre
                continue
            procesar(el)
            # el elemento contenedor sigue guardando una referencia a cada hijo
            # ya vaciado; sin esta poda periódica el árbol crece hasta ocupar
            # cientos de MB con 5,1 millones de aristas
            if self._contenedor is not None and n_aristas % 500_000 == 0:
                self._contenedor.clear()

        print(f"  grafo: {n_aristas} relaciones recorridas "
              f"({time.time() - inicio:.0f}s)", flush=True)

        # expansión por los vecinos que sobrevivieron a la criba
        for (q, fila_propia), monton in mejores.items():
            peso_propio = entidades_por_consulta[q][fila_propia]
            acumulador = bruto[q]
            for fuerza, fila_vecina in monton:
                aporte = PESO_VECINO * peso_propio * self._peso_nodo(fila_vecina)
                ini, fin = self.chunks_ptr[fila_vecina], self.chunks_ptr[fila_vecina + 1]
                for chunk_int in self.chunks_dat[ini:fin]:
                    acumulador[int(chunk_int)] += aporte

        # normalización y traducción a chunk_id de texto, quedándose solo con
        # los fragmentos que existen de verdad en la base vectorial
        salida: dict[str, dict[str, float]] = {}
        for q, (query_id, _) in enumerate(consultas):
            por_chunk: dict[str, float] = {}
            cuenta = cobertura[q]
            for chunk_int, valor in bruto[q].items():
                chunk_id = self.int_a_chunk_id[chunk_int]
                if chunk_id not in self.chunk_a_fila:
                    continue
                entidades_tocadas = cuenta.get(chunk_int, 0)
                factor = 1.0 + BONIF_COBERTURA * max(0, entidades_tocadas - 1)
                por_chunk[chunk_id] = valor * factor / float(np.sqrt(densidad[chunk_int]))
            salida[query_id] = por_chunk
        return salida


# ===========================================================================
# Lectura de consultas
# ===========================================================================

_RE_CONSULTA = re.compile(r"^\s*(q\d{3})\b[\s:.\-\t]*(.+)$", re.IGNORECASE)


def leer_consultas(ruta: Path) -> list[tuple[str, str]]:
    """
    Lee el archivo de consultas y devuelve [(query_id, texto), ...].

    Acepta tres formatos, para no depender de cómo se entreguen las preguntas:
      * texto plano con una consulta por línea, prefijada por su id (q001 ...);
      * JSON Lines con los campos ``query_id`` y ``query``/``pregunta``/``text``;
      * el PDF original, si ``pypdf`` está instalado.
    """
    if ruta.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit("Para leer las consultas desde PDF hace falta 'pypdf'")
        contenido = "\n".join((p.extract_text() or "") for p in PdfReader(str(ruta)).pages)
    else:
        contenido = ruta.read_text(encoding="utf-8")

    if ruta.suffix.lower() == ".jsonl":
        consultas = []
        for linea in contenido.splitlines():
            linea = linea.strip()
            if not linea:
                continue
            obj = json.loads(linea)
            texto = obj.get("query") or obj.get("pregunta") or obj.get("text") or ""
            consultas.append((obj["query_id"], _RE_ESPACIOS.sub(" ", texto).strip()))
        return consultas

    # Texto o PDF: las preguntas pueden venir partidas en varias líneas, así que
    # cada línea sin id se pega a la consulta que se está leyendo.
    consultas: list[tuple[str, str]] = []
    for linea in contenido.splitlines():
        coincidencia = _RE_CONSULTA.match(linea)
        if coincidencia:
            consultas.append([coincidencia.group(1).lower(), coincidencia.group(2).strip()])
        elif consultas and linea.strip():
            consultas[-1][1] += " " + linea.strip()
    return [(qid, _RE_ESPACIOS.sub(" ", texto).strip()) for qid, texto in consultas]


# ===========================================================================
# Recuperación
# ===========================================================================

def z(valores: np.ndarray) -> np.ndarray:
    """Estandariza las puntuaciones del pool de candidatos de una consulta.

    Trabajar en unidades de desviación típica hace que el peso del grafo
    (PESO_GRAFO) signifique lo mismo en una consulta donde los cosenos están
    muy juntos que en otra donde están muy separados."""
    if len(valores) == 0:
        return valores
    desviacion = float(valores.std())
    if desviacion < 1e-9:
        return np.zeros_like(valores)
    return (valores - float(valores.mean())) / desviacion


class Recuperador:
    def __init__(self, bases: dict[str, BaseVectorial], evidencia_grafo: dict[str, dict[str, float]]):
        self.bases = bases
        self.evidencia_grafo = evidencia_grafo
        self.comun = next(b for b in bases.values() if b.config["espacio_comun"])
        # chunk_id -> (nombre del índice, fila) para poder resolver los
        # candidatos que propone el grafo
        self.chunk_a_fila: dict[str, tuple[str, int]] = {}
        for nombre, base in bases.items():
            for fila, registro in enumerate(base.metadata):
                self.chunk_a_fila[registro["chunk_id"]] = (nombre, fila)

    def _vectores_en_espacio_comun(self, candidatos: list[tuple[str, int]]) -> np.ndarray:
        """
        Devuelve el vector e5 de cada candidato.

        Para los fragmentos que ya viven en el índice de e5 se lee el vector
        almacenado (es exactamente el que se indexó, sin coste ni deriva). Los
        que vienen de bge-m3 se codifican con e5 sobre su texto, con el mismo
        prefijo "passage: " con el que se construyó la base.
        """
        vectores = np.zeros((len(candidatos), self.comun.indice.d), dtype="float32")
        pendientes_idx: list[int] = []
        pendientes_txt: list[str] = []
        for i, (nombre, fila) in enumerate(candidatos):
            if nombre == self.comun.nombre:
                vectores[i] = self.comun.vector_de(fila)
            else:
                pendientes_idx.append(i)
                pendientes_txt.append(self.bases[nombre].metadata[fila]["texto"])
        if pendientes_txt:
            codificados = self.comun.codificar(pendientes_txt, self.comun.config["prefijo_pasaje"])
            for i, vector in zip(pendientes_idx, codificados):
                vectores[i] = vector
        return vectores

    def recuperar(self, query_id: str, consulta: str) -> dict:
        # --- 1. candidatos de cada índice FAISS ----------------------------
        # Cada índice se interroga con su propio encoder y su propio prefijo:
        # el vector de consulta tiene que habitar el mismo espacio semántico
        # que los vectores indexados (Sección 8.1, punto 1).
        candidatos: dict[str, tuple[str, int]] = {}
        vectores_consulta: dict[str, np.ndarray] = {}
        for base in self.bases.values():
            vector = base.vector_consulta(consulta)
            vectores_consulta[base.nombre] = vector
            for fila, _ in base.buscar(vector, base.config["top_k"]):
                candidatos[base.metadata[fila]["chunk_id"]] = (base.nombre, fila)

        # --- 2. candidatos del grafo (Sección 8.5, punto 3) ----------------
        grafo = self.evidencia_grafo.get(query_id, {})
        for chunk_id in sorted(grafo, key=grafo.get, reverse=True)[:100]:
            if chunk_id in self.chunk_a_fila:
                candidatos.setdefault(chunk_id, self.chunk_a_fila[chunk_id])

        chunk_ids = list(candidatos)
        referencias = [candidatos[c] for c in chunk_ids]

        # --- 3. calibración: todos contra la misma vara --------------------
        consulta_comun = vectores_consulta[self.comun.nombre][0]
        matriz = self._vectores_en_espacio_comun(referencias)
        cosenos = matriz @ consulta_comun

        maximo_grafo = max(grafo.values()) if grafo else 0.0
        aporte_grafo = np.array(
            [grafo.get(c, 0.0) / maximo_grafo if maximo_grafo > 0 else 0.0 for c in chunk_ids],
            dtype="float32",
        )
        puntuacion = z(cosenos) + PESO_GRAFO * aporte_grafo

        orden = np.argsort(-puntuacion)

        # --- 4. refinamiento: la mejor ventana de <=250 palabras ----------
        # Se recorta a oraciones completas y se trocea si hace falta; después se
        # vuelve a puntuar el texto que realmente se va a entregar. Puntuar el
        # chunk original y entregar otro texto sería ordenar por una cosa y
        # hacerse evaluar por otra.
        refinados: list[dict] = []
        pendientes: list[tuple[int, list[str]]] = []
        for posicion in orden[:N_REFINAR]:
            nombre, fila = referencias[posicion]
            registro = self.bases[nombre].metadata[fila]
            opciones = ventanas_de_250(recortar_a_oraciones(registro["texto"]))
            indice_refinado = len(refinados)
            refinados.append({
                "chunk_id": registro["chunk_id"],
                "doc_id": registro["doc_id"],
                "fuente": registro.get("fuente", ""),
                "texto": opciones[0],
                "coseno": float(cosenos[posicion]),
                "grafo": float(aporte_grafo[posicion]),
            })
            if len(opciones) > 1 or opciones[0] != registro["texto"]:
                pendientes.append((indice_refinado, opciones))

        # Una sola llamada al encoder para todas las ventanas de la consulta:
        # agrupar evita pagar el arranque del modelo una vez por candidato.
        if pendientes:
            planos = [texto for _, opciones in pendientes for texto in opciones]
            vectores = self.comun.codificar(planos, self.comun.config["prefijo_pasaje"])
            similitudes = vectores @ consulta_comun
            desplazamiento = 0
            for indice_refinado, opciones in pendientes:
                tramo = similitudes[desplazamiento:desplazamiento + len(opciones)]
                desplazamiento += len(opciones)
                elegida = int(np.argmax(tramo))
                refinados[indice_refinado]["texto"] = opciones[elegida]
                refinados[indice_refinado]["coseno"] = float(tramo[elegida])

        for registro in refinados:
            registro["texto"] = recortar_a_250(registro["texto"])

        cosenos_ref = z(np.array([r["coseno"] for r in refinados], dtype="float32"))
        for i, registro in enumerate(refinados):
            registro["puntuacion"] = float(cosenos_ref[i]) + PESO_GRAFO * registro["grafo"]
        refinados.sort(key=lambda r: -r["puntuacion"])

        # --- 5. los 10 fragmentos ------------------------------------------
        fragmentos: list[dict] = []
        por_documento: dict[str, int] = defaultdict(int)
        for registro in refinados:
            if len(fragmentos) >= N_FRAGMENTOS:
                break
            if por_documento[registro["doc_id"]] >= MAX_FRAG_POR_DOC:
                continue
            por_documento[registro["doc_id"]] += 1
            fragmentos.append({
                "rank": len(fragmentos) + 1,
                "chunk_id": registro["chunk_id"],
                "doc_id": registro["doc_id"],
                "text": registro["texto"],
            })
        # si el tope por documento dejó huecos, se rellenan por puntuación
        if len(fragmentos) < N_FRAGMENTOS:
            usados = {(f["chunk_id"], f["text"]) for f in fragmentos}
            for registro in refinados:
                if len(fragmentos) >= N_FRAGMENTOS:
                    break
                if (registro["chunk_id"], registro["texto"]) in usados:
                    continue
                fragmentos.append({
                    "rank": len(fragmentos) + 1,
                    "chunk_id": registro["chunk_id"],
                    "doc_id": registro["doc_id"],
                    "text": registro["texto"],
                })

        # --- 6. agregación a documento (Sección 8.6) -----------------------
        # Se agrupa por `fuente` y no por doc_id porque la evaluación a nivel de
        # documento empareja con el ground truth por el archivo original
        # (Sección 10.2.1): dos doc_id distintos del mismo archivo gastarían dos
        # de los tres cupos en el mismo documento.
        agrupado: dict[str, list[dict]] = defaultdict(list)
        for registro in refinados:
            agrupado[registro["fuente"] or registro["doc_id"]].append(registro)

        documentos: list[tuple[float, str]] = []
        for fuente, registros in agrupado.items():
            valores = sorted((r["puntuacion"] for r in registros), reverse=True)
            puntuacion_doc = valores[0] + BONIF_DOC * sum(max(0.0, v) for v in valores[1:])
            mejor = max(registros, key=lambda r: r["puntuacion"])
            documentos.append((puntuacion_doc, mejor["doc_id"]))
        documentos.sort(key=lambda par: -par[0])

        # El esquema exige exactamente 3 documentos. Con 60 candidatos
        # refinados siempre hay de sobra, pero si un corpus muy pequeño no
        # diera para tres se completa con el resto del pool de candidatos.
        if len(documentos) < N_DOCUMENTOS:
            ya = {doc_id for _, doc_id in documentos}
            for posicion in orden:
                nombre, fila = referencias[posicion]
                doc_id = self.bases[nombre].metadata[fila]["doc_id"]
                if doc_id not in ya:
                    ya.add(doc_id)
                    documentos.append((float(puntuacion[posicion]), doc_id))
                if len(documentos) >= N_DOCUMENTOS:
                    break

        return {
            "query_id": query_id,
            "documents": [
                {"rank": i + 1, "doc_id": doc_id}
                for i, (_, doc_id) in enumerate(documentos[:N_DOCUMENTOS])
            ],
            "fragments": fragmentos[:N_FRAGMENTOS],
        }


# ===========================================================================
# Programa principal
# ===========================================================================

def frecuencias_en_corpus(bases, consultas: list[tuple[str, str]]) -> dict[str, int]:
    """
    Cuenta en cuántos fragmentos del corpus aparece cada palabra de las
    consultas.

    Es el insumo con el que el grafo decide si una entidad de la consulta es
    informativa o es una palabra de relleno. Solo se cuentan las palabras que
    alguna consulta usa (unos pocos centenares), así que una única pasada por
    los 89.064 fragmentos basta y tarda pocos segundos.
    """
    interes = set()
    for _, texto in consultas:
        interes.update(palabras(normalizar(texto)))
    interes -= VACIAS

    df: dict[str, int] = defaultdict(int)
    for base in bases:
        for registro in base.metadata:
            for ficha in set(normalizar(registro["texto"]).split()) & interes:
                df[ficha] += 1
    return df


def elegir_dispositivo(preferencia: str) -> str:
    if preferencia != "auto":
        return preferencia
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera resultados.jsonl a partir de la base de conocimiento.",
    )
    parser.add_argument("--consultas", type=Path, default=RAIZ / "consultas.txt",
                        help="archivo de consultas (.txt, .jsonl o .pdf)")
    parser.add_argument("--salida", type=Path, default=RAIZ / "resultados.jsonl")
    parser.add_argument("--base-vectorial", type=Path, default=RAIZ / "base_vectorial")
    parser.add_argument("--grafo", type=Path, default=RAIZ / "grafo" / "grafo.graphml")
    parser.add_argument("--sin-grafo", action="store_true",
                        help="ignora el grafo y recupera solo con los índices FAISS")
    parser.add_argument("--cache-grafo", type=Path, default=None,
                        help="archivo donde guardar/reutilizar la evidencia del grafo")
    parser.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    consultas = leer_consultas(args.consultas)
    if not consultas:
        raise SystemExit(f"No se leyó ninguna consulta de {args.consultas}")
    print(f"Consultas: {len(consultas)} ({consultas[0][0]}–{consultas[-1][0]})")

    dispositivo = elegir_dispositivo(args.dispositivo)
    print(f"Dispositivo: {dispositivo}")

    bases: dict[str, BaseVectorial] = {}
    for config in INDICES:
        carpeta = args.base_vectorial / config["carpeta"]
        if not (carpeta / "index.faiss").exists():
            raise SystemExit(f"Falta el índice {carpeta / 'index.faiss'}")
        base = BaseVectorial(carpeta, config)
        base.cargar_modelo(dispositivo)
        bases[config["carpeta"]] = base
        print(f"  {config['carpeta']}: {base.indice.ntotal} fragmentos, {base.indice.d}d")

    # --- evidencia del grafo (una sola pasada para las 50 consultas) -------
    evidencia: dict[str, dict[str, float]] = {}
    if not args.sin_grafo:
        if args.cache_grafo and args.cache_grafo.exists():
            evidencia = json.loads(args.cache_grafo.read_text(encoding="utf-8"))
            print(f"  grafo: evidencia reutilizada de {args.cache_grafo}")
        elif args.grafo.exists():
            chunk_a_fila = {
                registro["chunk_id"]: (nombre, fila)
                for nombre, base in bases.items()
                for fila, registro in enumerate(base.metadata)
            }
            df_corpus = frecuencias_en_corpus(bases.values(), consultas)
            evidencia = GrafoConocimiento(args.grafo, chunk_a_fila, df_corpus).evidencia(consultas)
            if args.cache_grafo:
                args.cache_grafo.write_text(
                    json.dumps(evidencia, ensure_ascii=False), encoding="utf-8")
            con_evidencia = sum(1 for v in evidencia.values() if v)
            print(f"  grafo: evidencia para {con_evidencia}/{len(consultas)} consultas")
        else:
            print(f"  aviso: no se encontró {args.grafo}; se recupera solo con FAISS")

    recuperador = Recuperador(bases, evidencia)

    inicio = time.time()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8") as f:
        for i, (query_id, texto) in enumerate(consultas, 1):
            resultado = recuperador.recuperar(query_id, texto)
            f.write(json.dumps(resultado, ensure_ascii=False) + "\n")
            print(f"  [{i:2d}/{len(consultas)}] {query_id} "
                  f"docs={[d['doc_id'] for d in resultado['documents']]}", flush=True)

    print(f"\nEscrito {args.salida} ({time.time() - inicio:.0f}s)")


if __name__ == "__main__":
    main()
