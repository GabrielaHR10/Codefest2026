"""
consultaGrafo.py

Dada una pregunta en lenguaje natural, consulta el grafo de conocimiento
construido por construir_grafo.py (Sección 8.5 del documento CODEFEST):

    1. Extrae las entidades mencionadas en la pregunta con el MISMO
       componente NER usado al construir el grafo (GLiNER).
    2. Busca esas entidades entre los nodos del grafo (por coincidencia
       exacta o parcial de texto).
    3. Recupera los vecinos de primer orden de cada nodo encontrado
       (relaciones entrantes y salientes).
    4. Devuelve los chunk_id vinculados como evidencia, con una
       puntuación según cuántas relaciones relevantes los respaldan.

Este script NO genera una respuesta en lenguaje natural: la Sección
8.3 del documento prohíbe usar modelos generativos/decoder en
cualquier etapa de la recuperación. Por eso el resultado es evidencia
estructurada (entidades, nodos, chunk_id, puntuación, relaciones), no
una respuesta redactada -- la síntesis final es responsabilidad de
otra etapa del pipeline (fuera de este script).

Uso:
    python consulta_grafo.py
    (pide la ruta de grafo.graphml y luego la pregunta; imprime la
    evidencia encontrada en formato JSON)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx


# Categorías de entidad que reconoce GLiNER (zero-shot: no requieren
# entrenamiento previo, se le pasan como texto en cada consulta). Debe
# coincidir con la lista usada en construir_grafo.py, para que el
# vocabulario de tipos de la pregunta y el del grafo sea el mismo.
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

# Umbral de confianza mínimo para que GLiNER acepte una entidad de la
# pregunta como candidata. Es más bajo que el umbral usado al
# construir el grafo (0.5), porque las preguntas son cortas y con poco
# contexto: GLiNER da menos confianza aunque la entidad sea obvia para
# un humano. Confirmado empíricamente (ver debug_gliner.py): en
# preguntas de una sola línea, hasta la entidad "correcta" a veces no
# llega ni a 0.1 de score (ej. "coronavirus" en una pregunta en
# inglés, con las labels en español, dio score 0.098).
#
# Bajar el umbral aquí no afecta la calidad del grafo en sí -- ese se
# construyó aparte, con el umbral 0.5, sobre texto más largo y rico
# (los chunks). El NER es, además, solo UNA fuente de candidatos entre
# dos: buscar_nodos_por_texto() hace una búsqueda directa por texto
# como respaldo, para los casos en que GLiNER no proponga la entidad
# como candidata en absoluto.
UMBRAL_NER_PREGUNTA = 0.05


def cargar_ner():
    """Carga el mismo modelo NER usado en construir_grafo.py (GLiNER,
    multilingüe, zero-shot)."""
    from gliner import GLiNER
    return GLiNER.from_pretrained("urchade/gliner_multi-v2.1")


# Artículos/determinantes que GLiNER a veces incluye pegados al inicio
# del span que detecta (ej. devuelve "el coronavirus" en vez de
# "coronavirus"). Se usan en limpiar_entidad_pregunta() para quitarlos
# antes de comparar contra el grafo.
ARTICULOS_INICIALES = {"el", "la", "los", "las", "un", "una", "unos", "unas", "the", "a", "an"}
#puse artículos en inglés por si hacian preguntas en inglés pero creo que no 

def limpiar_entidad_pregunta(texto: str) -> str:
    """
    Quita artículos/determinantes sueltos al inicio de una entidad
    detectada por el NER.

    Es necesario porque GLiNER a veces devuelve el artículo pegado al
    sustantivo (ej. "el coronavirus" en vez de "coronavirus"). Sin
    esta limpieza, la comparación por substring exigiría que el nodo
    del grafo también trajera literalmente "el " justo antes de la
    palabra, lo que produce:
      - falsos positivos: coincide por accidente con palabras que
        terminan en "el" (ej. "novel coronavirus" parece contener
        "el coronavirus" solo porque "novel" termina en "el").
      - falsos negativos: nodos que sí hablan de "coronavirus" pero
        sin ese artículo pegado al lado quedan sin encontrarse.
    """
    palabras = texto.strip().split()
    while palabras and palabras[0].lower() in ARTICULOS_INICIALES:
        palabras.pop(0)
    return " ".join(palabras) if palabras else texto.strip()


def extraer_entidades_pregunta(pregunta: str, ner_modelo) -> list[str]:
    """
    Ejecuta GLiNER sobre la pregunta con las categorías de
    TIPOS_ENTIDAD y el umbral UMBRAL_NER_PREGUNTA, y devuelve el texto
    de cada entidad detectada, ya limpio de artículos iniciales.
    """
    entidades = ner_modelo.predict_entities(pregunta, TIPOS_ENTIDAD, threshold=UMBRAL_NER_PREGUNTA)
    return [limpiar_entidad_pregunta(e["text"]) for e in entidades]


def normalizar_entidad(texto: str) -> str:
    """Normaliza un texto para usarlo como id de nodo o para comparar
    contra uno: colapsa espacios repetidos y pasa a minúsculas."""
    return re.sub(r"\s+", " ", texto).strip().lower()


def _contiene_como_frase(cadena: str, subcadena: str) -> bool:
    """
    Verifica si 'subcadena' aparece dentro de 'cadena' respetando
    límites de palabra (\\b de regex), en vez de un substring crudo.

    Esto evita falsos positivos como "novel coronavirus" pareciendo
    contener "el coronavirus" solo porque la palabra "novel" termina
    justo en "el" -- un substring crudo (operador "in" de Python) no
    distingue eso de una coincidencia real de palabra completa.
    """
    if not subcadena:
        return False
    return re.search(r"\b" + re.escape(subcadena) + r"\b", cadena) is not None


def buscar_nodos_por_texto(pregunta: str, grafo: nx.DiGraph) -> list[str]:
    """
    Búsqueda directa (sin pasar por el NER) de nodos del grafo cuyo id
    aparece literalmente dentro de la pregunta.

    Complementa a extraer_entidades_pregunta() + encontrar_nodos():
    GLiNER es zero-shot sobre categorías amplias (país, organización,
    ...), y a veces no propone como candidata una palabra suelta que
    no encaja claramente en ninguna categoría (ej. "wuhan"), aunque
    esa palabra exista tal cual como nodo del grafo. Como el
    vocabulario de nodos ya se conoce de antemano (a diferencia del
    NER, que evalúa la pregunta "a ciegas"), comparar directo contra
    ese vocabulario es más confiable que depender solo del NER.
    """
    pregunta_norm = normalizar_entidad(pregunta)
    nodos_filtrados = []

    for nodo_id in grafo.nodes:
        # nodos de 1-2 caracteres se excluyen: un substring tan corto
        # coincidiría con casi cualquier pregunta, puro ruido
        if len(nodo_id) >= 3 and _contiene_como_frase(pregunta_norm, nodo_id):
            nodos_filtrados.append(nodo_id)

    return nodos_filtrados


def encontrar_nodos(grafo: nx.DiGraph, entidades: list[str]) -> list[str]:
    """
    Empareja las entidades detectadas en la pregunta (vía NER) con
    nodos del grafo: primero se intenta coincidencia exacta (texto
    normalizado); si no hay, se cae a coincidencia parcial por
    substring (con límites de palabra) como respaldo, para tolerar
    variaciones menores de redacción entre la pregunta y el corpus.

    Cuando cae al respaldo por substring, recoge TODOS los nodos que
    coincidan, no solo el primero -- si cortara en el primer match, se
    perderían nodos igual de válidos (ej. "coronavirus" aparece en
    varios títulos distintos del corpus), y el resultado dependería de
    forma arbitraria del orden de inserción de los nodos en el grafo.
    recuperar_evidencia() ya se encarga de puntuar y ordenar todos los
    candidatos después, así que aquí conviene traer de más y dejar que
    el scoring filtre lo relevante.
    """
    nodos_encontrados = []
    for ent in entidades:
        nodo_id = normalizar_entidad(ent)
        if nodo_id in grafo:
            nodos_encontrados.append(nodo_id)
            continue
        if len(nodo_id) < 3:
            # con el umbral de NER tan bajo, a veces se cuela una
            # palabra suelta corta ("el", "la"...); un substring así
            # de corto matchearía casi cualquier nodo
            continue
        for nodo in grafo.nodes:
            if _contiene_como_frase(nodo, nodo_id) or _contiene_como_frase(nodo_id, nodo):
                nodos_encontrados.append(nodo)
    return nodos_encontrados


# Máximo de chunks que devuelve recuperar_evidencia(), quedándose con
# los de mayor puntuación.
TOP_K_EVIDENCIA = 10


def recuperar_evidencia(grafo: nx.DiGraph, nodos: list[str]) -> list[dict]:
    """
    Para cada nodo encontrado, acumula los chunk_id donde se menciona
    la entidad y los de sus vecinos de primer orden (relaciones
    entrantes y salientes), con una puntuación = número de relaciones
    relevantes que respaldan ese chunk (Sección 8.5, punto 3).

    Devuelve como máximo TOP_K_EVIDENCIA chunks, ordenados de mayor a
    menor puntuación.
    """
    # defaultdict evita tener que inicializar manualmente cada clave
    # nueva (chunk_id) antes de sumarle o agregarle algo por primera vez.
    puntuacion_chunk = defaultdict(int)
    relaciones_por_chunk = defaultdict(list)

    for nodo in nodos:
        # 1) mención directa: chunks donde el nodo mismo fue detectado
        for chunk_id in grafo.nodes[nodo].get("chunk_ids", "").split("|"):
            if chunk_id:
                puntuacion_chunk[chunk_id] += 1

        # 2) vecinos de primer orden: grafo.succ[nodo] / grafo.pred[nodo]
        # dan (vecino, datos_de_la_arista) en un solo recorrido, cada
        # uno en su dirección real -- succ = aristas salientes
        # (nodo -> vecino), pred = aristas entrantes (vecino -> nodo).
        # Evita ambigüedad de dirección cuando existen aristas en
        # ambos sentidos entre los mismos dos nodos.
        for vecino, datos_arista in grafo.succ[nodo].items():  # nodo -> vecino
            relacion = datos_arista.get("relacion", "")
            for chunk_id in datos_arista.get("chunk_ids", "").split("|"):
                if chunk_id:
                    puntuacion_chunk[chunk_id] += 1
                    relaciones_por_chunk[chunk_id].append(f"{nodo} -[{relacion}]-> {vecino}")

        for vecino, datos_arista in grafo.pred[nodo].items():  # vecino -> nodo
            relacion = datos_arista.get("relacion", "")
            for chunk_id in datos_arista.get("chunk_ids", "").split("|"):
                if chunk_id:
                    puntuacion_chunk[chunk_id] += 1
                    relaciones_por_chunk[chunk_id].append(f"{vecino} -[{relacion}]-> {nodo}")

    chunks_ordenados = sorted(puntuacion_chunk.items(), key=lambda x: -x[1])[:TOP_K_EVIDENCIA]
    return [
        {
            "chunk_id": chunk_id,
            "puntuacion_grafo": puntuacion,
            "relaciones": relaciones_por_chunk.get(chunk_id, []),
        }
        for chunk_id, puntuacion in chunks_ordenados
    ]


def consultar(pregunta: str, grafo: nx.DiGraph, ner_modelo) -> dict:
    """
    Punto de entrada principal: dada una pregunta y el grafo ya
    cargado, ejecuta todo el proceso (NER -> búsqueda de nodos ->
    recuperación de evidencia) y devuelve un diccionario con el
    resultado estructurado, listo para serializar a JSON.
    """
    entidades = extraer_entidades_pregunta(pregunta, ner_modelo)
    nodos_ner = encontrar_nodos(grafo, entidades)
    nodos_directos = buscar_nodos_por_texto(pregunta, grafo)
    # unión sin duplicados, preservando el orden (primero lo que aportó el NER)
    nodos = list(dict.fromkeys(nodos_ner + nodos_directos))
    evidencia = recuperar_evidencia(grafo, nodos)
    return {
        "pregunta": pregunta,
        "entidades_detectadas": entidades,
        "nodos_encontrados": nodos,
        "evidencia": evidencia,
    }


def main():
    """CLI interactiva: pide la ruta del grafo y la pregunta por
    input(), y muestra el resultado de consultar() como JSON."""
    ruta_grafo = input("Ruta de grafo.graphml (Enter para 'grafo/grafo.graphml'): ").strip()
    ruta_grafo = Path(ruta_grafo) if ruta_grafo else Path("grafo/grafo.graphml")
    if not ruta_grafo.exists():
        print(f"Error: no se encontró '{ruta_grafo}'")
        return

    print("Cargando grafo...")
    grafo = nx.read_graphml(ruta_grafo)

    print("Cargando encoder NER (GLiNER)...")
    ner_modelo = cargar_ner()

    pregunta = input("\nEscribe tu pregunta: ").strip()
    resultado = consultar(pregunta, grafo, ner_modelo)

    print()
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()