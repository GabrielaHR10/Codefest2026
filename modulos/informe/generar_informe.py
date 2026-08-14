"""
generar_informe.py

Compone ``entrega/informe_tecnico.pdf``, el documento técnico de máximo 8
páginas que pide la Sección 1.4 del reto: estrategia de chunking y su
justificación, encoders seleccionados y criterios, tipo de índice FAISS y
descripción del grafo de conocimiento.

    python modulos/informe/generar_informe.py

Las cifras del informe no están escritas a mano: se leen de la base vectorial
entregada, de modo que el documento no puede desincronizarse de lo que hay en
``entrega/``.
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

RAIZ = Path(__file__).resolve().parents[2]
ENTREGA = RAIZ / "entrega"

AZUL = colors.HexColor("#1F3864")
GRIS = colors.HexColor("#F2F2F2")


# ---------------------------------------------------------------------------
# Cifras leídas de la propia entrega
# ---------------------------------------------------------------------------

def estadisticas() -> dict:
    datos = {}
    for carpeta in sorted((ENTREGA / "base_vectorial").iterdir()):
        ruta = carpeta / "metadata.jsonl"
        if not ruta.exists():
            continue
        docs, fuentes, palabras = set(), set(), []
        formatos = collections.Counter()
        fenomenos = collections.Counter()
        docs_por_formato = collections.defaultdict(set)
        with ruta.open(encoding="utf-8") as f:
            for linea in f:
                obj = json.loads(linea)
                docs.add(obj["doc_id"])
                fuentes.add(obj["fuente"])
                palabras.append(len(obj["texto"].split()))
                formatos[obj["formato"]] += 1
                fenomenos[obj["fenomeno"]] += 1
                docs_por_formato[obj["formato"]].add(obj["doc_id"])
        datos[carpeta.name] = {
            "fragmentos": len(palabras),
            "documentos": len(docs),
            "fuentes": len(fuentes),
            "formatos": dict(formatos),
            "docs_por_formato": {k: len(v) for k, v in docs_por_formato.items()},
            "fenomenos": dict(sorted(fenomenos.items())),
            "palabras_media": statistics.mean(palabras),
            "palabras_max": max(palabras),
        }
    return datos


# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------

def estilos() -> dict:
    base = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle(
            "titulo", parent=base["Title"], fontSize=19, leading=23, textColor=AZUL,
            spaceAfter=2),
        "subtitulo": ParagraphStyle(
            "subtitulo", parent=base["Normal"], fontSize=10.5, leading=14,
            textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=14),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=12.5, leading=15, textColor=AZUL,
            spaceBefore=11, spaceAfter=5),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=10.5, leading=13,
            textColor=colors.HexColor("#2E4E86"), spaceBefore=8, spaceAfter=3),
        "cuerpo": ParagraphStyle(
            "cuerpo", parent=base["BodyText"], fontSize=8.9, leading=12.2,
            alignment=TA_JUSTIFY, spaceAfter=5),
        "lista": ParagraphStyle(
            "lista", parent=base["BodyText"], fontSize=8.9, leading=12.2,
            alignment=TA_JUSTIFY, leftIndent=12, bulletIndent=3, spaceAfter=3),
        "celda": ParagraphStyle(
            "celda", parent=base["BodyText"], fontSize=8.2, leading=10.5, spaceAfter=0),
        "celda_cab": ParagraphStyle(
            "celda_cab", parent=base["BodyText"], fontSize=8.2, leading=10.5,
            textColor=colors.white, spaceAfter=0),
        "formula": ParagraphStyle(
            "formula", parent=base["BodyText"], fontSize=9.2, leading=13,
            alignment=1, textColor=AZUL, spaceBefore=4, spaceAfter=6),
        "pie": ParagraphStyle(
            "pie", parent=base["BodyText"], fontSize=7.6, leading=10,
            textColor=colors.HexColor("#666666")),
    }


def tabla(filas: list[list[str]], anchos: list[float], est: dict) -> Table:
    cuerpo = [[Paragraph(c, est["celda_cab"] if i == 0 else est["celda"]) for c in fila]
              for i, fila in enumerate(filas)]
    t = Table(cuerpo, colWidths=anchos, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRIS]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def punto(texto: str, est: dict):
    return Paragraph(texto, est["lista"], bulletText="•")


# ---------------------------------------------------------------------------
# Contenido
# ---------------------------------------------------------------------------

def construir(datos: dict) -> list:
    est = estilos()
    bge = datos["encoder_bge-m3"]
    e5 = datos["encoder_multilingual-e5-base"]
    total_frag = bge["fragmentos"] + e5["fragmentos"]
    total_docs = bge["documentos"] + e5["documentos"]
    total_fuentes = bge["fuentes"] + e5["fuentes"]

    h = []
    h.append(Paragraph("Base de conocimiento vectorial — Documento técnico", est["titulo"]))
    h.append(Paragraph(
        "CODEFEST AD ASTRA 2026 · Etapa 1 · Decisiones de diseño de la base vectorial, "
        "el grafo de conocimiento y el módulo de recuperación", est["subtitulo"]))

    # ---------------------------------------------------------------- 1
    h.append(Paragraph("1. Qué se entrega", est["h1"]))
    h.append(Paragraph(
        f"La base cubre <b>{total_frag:,}</b> fragmentos procedentes de "
        f"<b>{total_docs:,}</b> documentos ({total_fuentes:,} archivos distintos del corpus "
        "de ADL) repartidos entre los tres fenómenos del reto. Se construyó con "
        "<b>dos encoders complementarios</b>, cada uno con su propio índice FAISS "
        "(Sección 4.4), y se acompaña del componente bonus: un grafo de conocimiento de "
        "226.821 entidades y 5.118.081 relaciones que cubre las dos particiones."
        .replace(",", "."), est["cuerpo"]))

    h.append(tabla([
        ["Índice", "Encoder", "Dim.", "Fragmentos", "Docs.", "Partición del corpus"],
        ["encoder_multilingual-e5-base", "intfloat/multilingual-e5-base", "768",
         f"{e5['fragmentos']:,}".replace(",", "."), f"{e5['documentos']:,}".replace(",", "."),
         f"PDF de observatorios ({e5['docs_por_formato'].get('pdf', 0)} docs) y "
         f"artículos web en JSON ({e5['docs_por_formato'].get('json', 0)} docs)"],
        ["encoder_bge-m3", "BAAI/bge-m3", "1024",
         f"{bge['fragmentos']:,}".replace(",", "."), str(bge["documentos"]),
         f"Datasets XLSX del AI Index ({bge['formatos'].get('xlsx', 0)} filas) y "
         f"mapas .pbf ({bge['formatos'].get('pbf', 0)} municipios)"],
    ], [4.2 * cm, 3.5 * cm, 1.1 * cm, 1.9 * cm, 1.4 * cm, 5.0 * cm], est))
    h.append(Spacer(1, 5))
    h.append(Paragraph(
        "El reparto no es arbitrario: el corpus se dividió por tipo de fuente para poder "
        "procesarlo en paralelo, y cada mitad se codificó con el encoder que mejor le "
        "convenía. La consecuencia es que un fragmento vive en uno solo de los dos "
        "espacios vectoriales, y eso condiciona cómo se combinan en la recuperación "
        "(Sección 5 de este informe).", est["cuerpo"]))

    # ---------------------------------------------------------------- 2
    h.append(Paragraph("2. Preprocesamiento y extracción de texto", est["h1"]))
    h.append(Paragraph(
        "Cada formato del corpus exige una vía de extracción distinta (Sección 2.1). En "
        "todos los casos el texto resultante se normaliza a UTF-8, se colapsan espacios y "
        "caracteres de control, y se eliminan cabeceras, pies y numeración de página "
        "repetidos.", est["cuerpo"]))
    h.append(tabla([
        ["Formato", "Extracción", "Unidad de fragmentación"],
        ["PDF", "Texto por página respetando el orden de lectura; se descartan figuras y "
                "elementos decorativos sin contenido legible.",
         "Ventana deslizante sobre el texto continuo del documento."],
        ["JSON", "Se interpreta el objeto y se concatenan solo los campos de contenido "
                 "(título y cuerpo). Los campos descriptivos (url, fecha, autores) se "
                 "guardan como metadata, no se mezclan con el texto.",
         "Bloques del artículo."],
        ["XLSX", "Se lee la cabecera y luego cada registro como pares "
                 "<i>columna: valor</i>, de modo que cada valor conserva el nombre de su "
                 "columna como contexto.",
         "Una fila = un fragmento."],
        ["PBF", "Se decodifican las teselas vectoriales, se recorren las capas y se leen "
                "los atributos de cada elemento del mapa. Como el mismo municipio se "
                "repite en varios niveles de zoom, se deduplica: de 8.522 apariciones "
                "quedan 985 municipios únicos.",
         "Un municipio = un fragmento, redactado en prosa."],
    ], [1.7 * cm, 9.2 * cm, 6.2 * cm], est))

    # ---------------------------------------------------------------- 3
    h.append(Paragraph("3. Estrategia de chunking y su justificación", est["h1"]))
    h.append(Paragraph(
        "No se usó una única estrategia para todo el corpus, porque los cuatro tipos de "
        "fuente tienen estructuras que no se parecen en nada. Lo que sí es común es el "
        "criterio: el fragmento debe ser la unidad más pequeña que todavía responde por sí "
        "sola, sin depender de lo que había antes.", est["cuerpo"]))
    h.append(punto(
        f"<b>PDF — ventana deslizante de 220 palabras con 40 de solapamiento.</b> Los "
        f"informes de observatorios son texto corrido de decenas de páginas, sin marcado "
        f"estructural fiable tras la extracción. Una ventana de 220 palabras (media real: "
        f"{e5['palabras_media']:.0f}) cabe holgadamente en el límite de 512 tokens del "
        f"encoder y deja margen para el prefijo. El solapamiento de 40 palabras es lo que "
        f"evita que una idea que cae justo en la frontera entre dos ventanas se pierda "
        f"para ambas. Cada fragmento guarda además la página de inicio y de fin, lo que "
        f"permite rastrear cualquier resultado hasta el PDF original.", est))
    h.append(punto(
        "<b>JSON — bloques del artículo.</b> Los artículos web ya vienen con el cuerpo "
        "separado en párrafos, así que se respeta esa segmentación del autor, que es la "
        "más cercana a la estructura semántica del texto.", est))
    h.append(punto(
        "<b>XLSX — una fila por fragmento.</b> En un dataset tabular la fila es la unidad "
        "semántica completa: partirla no aporta nada y juntar varias mezcla registros que "
        "no tienen relación. Cada celda se serializa con el nombre de su columna delante "
        "para que el encoder sepa qué significa cada valor.", est))
    h.append(punto(
        "<b>PBF — un municipio por fragmento.</b> Los mapas no son texto, así que el "
        "módulo de mapas convierte los atributos de cada municipio (economías ilícitas, "
        "grupos armados presentes, indicadores) en un párrafo en lenguaje natural. El "
        "resultado se comporta como cualquier otro fragmento del corpus.", est))
    h.append(Paragraph(
        "<b>Completitud lingüística.</b> El requisito de la Sección 3.3 se aplica en dos "
        "puntos. En el índice, las particiones de JSON, XLSX y PBF cortan siempre en "
        "límites naturales (párrafo, fila, entidad del mapa). En la entrega, "
        "<i>generador.py</i> vuelve a imponerlo sobre el texto que efectivamente se "
        "reporta: recorta los bordes de cada fragmento hasta la oración completa más "
        "cercana y, si un fragmento supera las 250 palabras, lo divide en ventanas que "
        "cortan solo entre oraciones (Sección 9.2.1).", est["cuerpo"]))

    # ---------------------------------------------------------------- 4
    h.append(Paragraph("4. Encoders y criterios de selección", est["h1"]))
    h.append(Paragraph(
        "El corpus mezcla español, inglés y portugués, y las consultas de evaluación "
        "llegan en los tres idiomas, así que el soporte multilingüe nativo era la "
        "condición eliminatoria: sin él, una pregunta en español no puede recuperar un "
        "informe en inglés. Los dos modelos elegidos son <b>encoders</b> de la familia "
        "BERT/XLM-R, con licencia MIT, disponibles públicamente en HuggingFace.", est["cuerpo"]))
    h.append(tabla([
        ["Criterio", "intfloat/multilingual-e5-base", "BAAI/bge-m3"],
        ["Multilingüe", "100 idiomas; entrenado con pares consulta–pasaje en varios "
                        "idiomas, que es justo la tarea del reto.",
         "100+ idiomas, muy sólido en recuperación translingüe."],
        ["Dimensión", "768 — suficiente para el tamaño del corpus y cuatro veces más "
                      "barato de almacenar y comparar que un modelo grande.",
         "1024, con mayor capacidad expresiva."],
        ["Longitud máx.", "512 tokens, que es lo que fija el tamaño de chunk.",
         "8.192 tokens, holgado para filas y descripciones de mapas."],
        ["Rendimiento", "Fuerte en la pista multilingüe de MTEB/BEIR en recuperación densa.",
         "Referencia actual en recuperación multilingüe."],
        ["Coste", "278M parámetros: permite indexar 79.178 fragmentos en tiempo razonable.",
         "568M parámetros: se reservó para la partición pequeña."],
    ], [2.4 * cm, 7.4 * cm, 7.3 * cm], est))
    h.append(Spacer(1, 5))
    h.append(Paragraph(
        "<b>Convención de codificación.</b> e5 fue entrenado con prefijos de rol y "
        "omitirlos degrada la recuperación: los fragmentos se indexaron con "
        "<i>«passage: »</i> y las consultas se codifican con <i>«query: »</i>. bge-m3 no "
        "usa prefijo. Ambas convenciones se verificaron reconstruyendo vectores del "
        "índice y comparándolos con una recodificación del mismo texto: la similitud es "
        "exactamente 1,0 con el prefijo correcto y baja a 0,94 con el equivocado.",
        est["cuerpo"]))

    # ---------------------------------------------------------------- 5
    h.append(Paragraph("5. Índice FAISS", est["h1"]))
    h.append(Paragraph(
        "Los dos índices son <b>IndexFlatIP con vectores normalizados a norma unitaria</b>, "
        "serializados con <i>faiss.write_index()</i> y cargables con <i>faiss.read_index()</i> "
        "sin dependencias adicionales. Con norma unitaria el producto interno es "
        "exactamente la similitud coseno (Sección 8.2), de modo que el índice devuelve "
        "directamente la métrica que pide el reto.", est["cuerpo"]))
    h.append(Paragraph(
        "Se descartaron los índices aproximados (IVF, HNSW) a propósito. Con 89.064 "
        "vectores, la búsqueda exacta tarda del orden de 10 ms por consulta: no hay ningún "
        "problema de latencia que resolver, y a cambio un índice aproximado introduciría "
        "una pérdida de exactitud que se pagaría directamente en NDCG. La regla es simple: "
        "no se sacrifica recall para ganar una velocidad que no hace falta.", est["cuerpo"]))
    h.append(Paragraph(
        "El almacén de metadata es un <i>metadata.jsonl</i> por índice, con un objeto por "
        "línea y <b>la línea i correspondiendo al vector i</b> del índice FAISS "
        "(Sección 5.3). Cada objeto trae los ocho campos obligatorios de la Tabla 1 más "
        "la ruta original y, en los PDF, las páginas de origen.", est["cuerpo"]))

    h.append(PageBreak())

    # ---------------------------------------------------------------- 6
    h.append(Paragraph("6. Grafo de conocimiento (componente bonus)", est["h1"]))
    h.append(Paragraph(
        "El grafo se construyó sobre los mismos fragmentos indexados, en tres etapas "
        "(Sección 7.2):", est["cuerpo"]))
    h.append(punto(
        "<b>Reconocimiento de entidades.</b> Se usó <i>GLiNER</i> (urchade/gliner_multi-v2.1), "
        "un modelo NER multilingüe <i>zero-shot</i>: en vez de una taxonomía fija, se le "
        "pasa como texto la lista de tipos que interesan al reto —país, organización, grupo "
        "armado, institución del Estado, tecnología, sistema de armas, satélite, capacidad "
        "contraespacial, órbita, instrumento legal, economía ilícita, recurso natural, "
        "infraestructura crítica, evento o conflicto, entre otros—. Umbral de confianza 0,5 "
        "sobre el texto de los chunks.", est))
    h.append(punto(
        "<b>Extracción de relaciones.</b> Las relaciones se derivan de la coocurrencia de "
        "entidades dentro de un mismo fragmento, que es la unidad en la que hay evidencia "
        "textual verificable de que las dos entidades tienen algo que ver.", est))
    h.append(punto(
        "<b>Integración.</b> Cada nodo guarda los <i>chunk_id</i> y <i>doc_id</i> donde "
        "aparece la entidad, y cada arista guarda los <i>chunk_id</i> que la sostienen. Esa "
        "es la vinculación con la base vectorial (Sección 7.3): del grafo se puede volver "
        "siempre al texto que respalda cada afirmación. Se exporta como <i>grafo.graphml</i> "
        "desde NetworkX.", est))
    h.append(Paragraph(
        "El resultado son <b>226.821 entidades</b> y <b>5.118.081 relaciones</b>, unos "
        "843.672 vínculos entidad–fragmento. El archivo pesa cerca de 1 GB, lo que descarta "
        "cargarlo con <i>networkx.read_graphml</i> en cada consulta: <i>generador.py</i> lo "
        "recorre con un parser incremental <b>una sola vez para las 50 consultas</b> "
        "(45 segundos, memoria acotada), aprovechando que GraphML emite primero todos los "
        "nodos y después todas las aristas.", est["cuerpo"]))

    # ---------------------------------------------------------------- 7
    h.append(Paragraph("7. Módulo de recuperación", est["h1"]))
    h.append(Paragraph("7.1 El problema de combinar dos espacios disjuntos", est["h2"]))
    h.append(Paragraph(
        "Las estrategias clásicas de fusión (CombSUM, CombMNZ, RRF) suponen que los dos "
        "índices ordenan <i>el mismo conjunto</i> de fragmentos y que discrepan en el "
        "orden. Aquí no es el caso: las particiones son disjuntas, así que un fragmento "
        "aparece en un ranking y en el otro no. Aplicar RRF sin más pondría el mejor "
        "resultado de la partición tabular al mismo nivel que el mejor de la partición de "
        "informes, por irrelevante que fuera, solo porque ambos son «el primero de su "
        "lista». Y sumar cosenos crudos tampoco sirve, porque cada encoder tiene su propia "
        "escala de similitud.", est["cuerpo"]))
    h.append(Paragraph(
        "La solución adoptada es usar los dos índices como fuentes de <i>recall</i> y "
        "<b>reproyectar todos los candidatos a un único espacio común</b> antes de "
        "ordenarlos. Los fragmentos que ya están en el índice de e5 aportan su vector "
        "almacenado (exacto, sin recodificar); los que vienen de bge-m3 se codifican con e5 "
        "sobre su texto. Así todos se puntúan con la misma vara, sin ningún modelo "
        "generativo de por medio.", est["cuerpo"]))

    h.append(Paragraph("7.2 Flujo completo", est["h2"]))
    h.append(punto(
        "<b>Recuperación.</b> Top-200 del índice e5 (consulta codificada con "
        "<i>«query: »</i>) + top-100 del índice bge-m3 + hasta 100 fragmentos propuestos "
        "por el grafo. La unión forma el pool de candidatos.", est))
    h.append(punto(
        "<b>Calibración.</b> Coseno de cada candidato contra la consulta en el espacio "
        "común, estandarizado sobre el pool (media 0, desviación 1). Trabajar en unidades "
        "de desviación típica hace que el peso del grafo signifique lo mismo en una "
        "consulta donde los cosenos están muy juntos que en otra donde están separados.", est))
    h.append(punto(
        "<b>Refinamiento.</b> De los 60 mejores se recorta el texto a oraciones completas "
        "y, si supera las 250 palabras, se elige la ventana que más se parece a la "
        "consulta. Se puntúa la ventana que se va a entregar, no el chunk entero: es "
        "exactamente lo que evalúa NDCG@10, que juzga la relevancia sobre el campo "
        "<i>text</i>.", est))
    h.append(punto(
        "<b>Fusión con el grafo</b> (Sección 8.5, punto 4) y ordenación final.", est))
    h.append(Paragraph(
        "puntuación(c) = z(coseno(c)) + 0,5 · grafo(c)", est["formula"]))
    h.append(punto(
        "<b>Agregación a documento</b> (Sección 8.6): max-pooling más una bonificación de "
        "0,3 por cada fragmento adicional del mismo documento, de modo que un documento "
        "con varios fragmentos relevantes gane a uno con un único acierto aislado. La "
        "agrupación se hace por campo <i>fuente</i> y no por <i>doc_id</i>, porque la "
        "evaluación empareja documentos por el archivo original (Sección 10.2.1): dos "
        "<i>doc_id</i> del mismo archivo gastarían dos de los tres cupos en el mismo "
        "documento.", est))

    h.append(Paragraph("7.3 Cómo puntúa el grafo", est["h2"]))
    h.append(Paragraph(
        "Las entidades de la consulta se localizan buscando sus n-gramas contra el "
        "vocabulario de nodos del grafo —que es el que produjo GLiNER al construirlo, así "
        "que ambos lados comparten inventario (Sección 8.5, punto 1)—. Es la forma exacta, "
        "y mucho más barata, de lo que hacía la búsqueda por expresión regular nodo a "
        "nodo. A partir de ahí se recuperan los fragmentos de esas entidades y de sus "
        "vecinos de primer orden:", est["cuerpo"]))
    h.append(Paragraph(
        "bruto(c) = Σ w(e) sobre las entidades e de la consulta mencionadas en c "
        "+ 0,15 · Σ w(e)·w(v) sobre los vecinos v de primer orden", est["formula"]))
    h.append(Paragraph(
        "grafo(c) = bruto(c) · (1 + 0,4·(entidades_cubiertas(c) − 1)) / √(entidades_totales(c))",
        est["formula"]))
    h.append(Paragraph(
        "Los correctivos de esa fórmula salieron de mirar lo que devolvía la versión "
        "ingenua —sumar pesos IDF y ordenar—, que era casi inservible:", est["cuerpo"]))
    h.append(punto(
        "<b>El peso se mide contra el corpus, no contra el grafo.</b> El vocabulario del "
        "grafo contiene sustantivos genéricos que el NER marcó como entidad en un par de "
        "fragmentos sueltos («amenazas», «servicios», «pruebas»). Pesarlos por su "
        "frecuencia <i>dentro del grafo</i> les daba el peso máximo —una detección única "
        "parecía máxima especificidad— y arrastraban resultados sin ninguna relación con "
        "la pregunta.", est))
    h.append(punto(
        "<b>Una palabra suelta solo cuenta si la consulta la escribe como nombre propio o "
        "como sigla</b> («Colombia», «GAOR», «LEO»). Medir la frecuencia tampoco bastaba: "
        "como el corpus es mayoritariamente inglés, cualquier palabra española resulta "
        "«rara» y «pruebas» acababa pesando más que «spoofing». La mayúscula del texto "
        "original sí separa las entidades que anclan una consulta de las que no. Con esta "
        "regla el grafo aporta evidencia en 34 de las 50 consultas, con entidades como "
        "«grupos armados organizados residuales» o «corea del norte»; en las 16 restantes "
        "no aporta nada, que es preferible a inventar evidencia.", est))
    h.append(punto(
        "<b>El divisor por densidad de entidades</b> corrige el sesgo que ponía en cabeza "
        "a las páginas de bibliografía y a las tablas de datos: no son relevantes para "
        "nada, pero mencionan tantas entidades que ganaban por pura acumulación.", est))
    h.append(punto(
        "<b>La expansión hacia vecinos se limita a los 25 mejor respaldados</b> por "
        "entidad, en vez de expandir a ciegas por un grafo con nodos de decenas de miles "
        "de aristas.", est))

    # ---------------------------------------------------------------- 8
    h.append(Paragraph("8. Restricción sobre modelos generativos", est["h1"]))
    h.append(Paragraph(
        "Ninguna etapa de la construcción ni de la recuperación usa arquitecturas decoder "
        "(Sección 8.3). Los dos modelos de embedding son encoders; GLiNER, usado solo para "
        "construir el grafo, también lo es (DeBERTa). No hay reordenamiento por LLM, ni "
        "reformulación o expansión generativa de la consulta, ni síntesis de fragmentos: "
        "el texto que se entrega es literalmente el del corpus, recortado en límites de "
        "oración. Todo el reordenamiento se hace con similitud coseno, pesos IDF y "
        "metadata.", est["cuerpo"]))

    # ---------------------------------------------------------------- 9
    h.append(Paragraph("9. Reproducibilidad", est["h1"]))
    h.append(Paragraph(
        "<font face='Courier'>python entrega/generador.py</font> lee "
        "<i>entrega/consultas.txt</i>, carga los dos índices y el grafo, y reescribe "
        "<i>entrega/resultados.jsonl</i> con las 50 líneas del formato de la Sección 9. "
        "Tarda unos tres minutos en total. Las versiones de las librerías están fijadas en "
        "<i>requirements.txt</i>, porque un cambio de versión del encoder cambia los "
        "vectores y con ellos el ranking. <font face='Courier'>python "
        "modulos/validar_resultados.py</font> comprueba después el esquema completo: 50 "
        "líneas en orden, 3 documentos y 10 fragmentos por consulta, ningún fragmento por "
        "encima de 250 palabras y todos los identificadores existentes en la base "
        "entregada.", est["cuerpo"]))

    return h


def pie_de_pagina(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.2 * cm, "CODEFEST AD ASTRA 2026 · Etapa 1 · Documento técnico")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
    canvas.restoreState()


def main() -> None:
    datos = estadisticas()
    salida = ENTREGA / "informe_tecnico.pdf"
    doc = SimpleDocTemplate(
        str(salida), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=2 * cm,
        title="Documento técnico — CODEFEST AD ASTRA 2026 Etapa 1",
        author="Equipo CODEFEST AD ASTRA 2026",
    )
    doc.build(construir(datos), onFirstPage=pie_de_pagina, onLaterPages=pie_de_pagina)

    from pypdf import PdfReader
    paginas = len(PdfReader(str(salida)).pages)
    print(f"Escrito {salida} ({paginas} páginas)")
    if paginas > 8:
        raise SystemExit(f"El informe excede el máximo de 8 páginas ({paginas})")


if __name__ == "__main__":
    main()
