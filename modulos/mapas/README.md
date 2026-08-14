# Módulo de mapas — Amazon Underworld

Este módulo toma los 73 archivos de mapa (`.pbf`) que ADL entregó dentro del
corpus y los convierte en 985 fragmentos de texto que se pueden buscar. Cubre
el **Fenómeno 3** de CODEFEST AD ASTRA 2026, el de dinámicas territoriales en
América Latina.

La idea de fondo es sencilla. Los mapas no son texto, así que un buscador no
puede encontrar nada dentro de ellos tal como vienen. Lo que hace este módulo
es abrirlos, sacar la información que guardan sobre cada municipio y escribirla
en forma de párrafos en lenguaje natural. A partir de ahí ya son fragmentos
normales, como los que salen de un PDF, y el resto del sistema puede
procesarlos igual que a los demás.

Es un módulo **enchufable**. Produce un único archivo, `metadata_mapas.jsonl`,
que se junta con los fragmentos del resto del corpus. **No construye el índice
FAISS, ni el `generador.py`, ni el `resultados.jsonl`**: de eso se encarga el
pipeline central, que llevan otras personas del equipo.

**Contenido:** [Cómo ejecutar](#cómo-ejecutar) · [Qué produce](#qué-produce) ·
[Diseño y decisiones](#diseño-y-decisiones) ·
[Configuración](#configuración) · [Problemas frecuentes](#problemas-frecuentes)

---

## Cómo ejecutar

### 1. Instalar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r modulos/mapas/requirements.txt
```

En `requirements.txt` están todas las versiones fijadas exactamente. Eso no es
manía: el reto exige poder reproducir los resultados, y basta con que una
librería cambie de comportamiento para que el índice deje de ser el mismo.

Estas son las que hacen falta para generar los fragmentos:

| Paquete | Versión | Para qué se usa |
|---|---|---|
| `python` | 3.14.6 | |
| `mapbox-vector-tile` | 2.2.0 | abrir los archivos de mapa |
| `pandas` | 3.0.5 | manejar la tabla de municipios |
| `pyarrow` | 25.0.1 | guardar esa tabla en disco |
| `openpyxl` | 3.1.5 | leer el inventario de ADL, que es un Excel |
| `transformers` | 5.15.0 | contar los tokens de cada fragmento |

`transformers` es **opcional**. Si no está instalado, el módulo funciona igual;
lo único que cambia es que el campo `num_tokens` se calcula contando palabras
en vez de tokens reales, y el script avisa de ello por pantalla. Conviene
instalarlo antes de la entrega final para que ese número sea exacto. La primera
vez que se usa descarga el tokenizador de `BAAI/bge-m3`, así que hace falta
conexión a internet una sola vez.

### 2. Correr los tres pasos, en orden

```bash
CORPUS="/ruta/a/CORPUS CODEFEST AD ASTRA 2026"
INDICE="/ruta/a/Indice_Datos_Codefest.xlsx"

python decodificador_mvt.py "$CORPUS" municipios.parquet "$INDICE"
python generador_chunks.py  municipios.parquet "$INDICE"
python verificar_mapas.py   municipios.parquet metadata_mapas.jsonl
```

El proceso entero tarda unos **4 segundos**, así que se puede repetir tantas
veces como haga falta mientras se ajusta algo.

Cada paso alimenta al siguiente:

| Paso | Script | Qué recibe | Qué deja |
|---|---|---|---|
| 1 | `decodificador_mvt.py` | los `.pbf` y el inventario | `municipios.parquet`, una tabla con un municipio por fila |
| 2 | `generador_chunks.py` | esa tabla y el inventario | **`metadata_mapas.jsonl`** ← esto es lo que se entrega |
| 3 | `verificar_mapas.py` | las dos cosas anteriores | un informe en pantalla y `muestra_revision.txt` |

Al primer paso se le puede pasar directamente **la carpeta raíz del corpus
completo**, sin separar los mapas antes. El script busca archivos `.pbf` en
todas las subcarpetas, y como los mapas de Amazon Underworld son los únicos
`.pbf` del corpus, no hay riesgo de que recoja otra cosa por error.

El segundo argumento de cada comando es simplemente el nombre del archivo que
quieres que genere; puedes cambiarlo si prefieres otra ruta.

### 3. Comprobar que salió bien

El tercer script es el que dice si el resultado sirve o no. **Termina con código
de error 1 si encuentra algún fallo**, así que se puede usar tal cual en una
verificación automática:

```bash
python verificar_mapas.py municipios.parquet metadata_mapas.jsonl || echo "REVISAR"
```

Cuando todo va bien, sobre el corpus completo, verás algo así:

```
Teselas decodificadas correctamente: 73
Apariciones de municipio, con duplicados entre zooms: 8522
Deduplicacion: 8522 apariciones, quedan 985 municipios unicos
...
Resultado: sin fallos, 2 avisos por revisar
```

Estas son las cifras que deberían salir. Si alguna se desvía mucho, algo va
mal y merece la pena mirarlo antes de seguir:

- **73 de 73** archivos de mapa leídos
- **8.522** apariciones en bruto, que tras quitar repeticiones quedan en
  **985 municipios** distintos (cada municipio sale unas 8,7 veces)
- **6 países**: Brasil 771, Colombia 87, Ecuador 40, Bolivia 33, Perú 32,
  Venezuela 22
- **3 documentos** distintos, todos del nivel de zoom más alejado
- **662 municipios** (el 67%) con presencia de algún grupo armado registrada
- Fragmentos de entre 75 y 133 palabras, con una media de 86. El reto pone un
  máximo de 250, así que hay margen de sobra

Los **2 avisos** que aparecen son normales y no impiden entregar. Uno señala
que el campo `formato` vale `"pbf"`, que no es uno de los tres valores que
enumera la tabla del reto. El otro es una lista informativa de nombres de
lugares que podrían llevar tilde y no la llevan; ya se revisó a mano una por
una y están bien como están.

En total el verificador hace 18 comprobaciones, repartidas en cuatro bloques:
que la lectura de los mapas cuadre, que el resultado tenga todos los campos
obligatorios del reto, que el texto esté bien escrito (dentro del límite de
palabras, sin frases cortadas, sin contradicciones internas) y una selección de
fragmentos para leer con calma.

Ese último bloque importa más de lo que parece. **Ninguna comprobación
automática sustituye a leer `muestra_revision.txt`.** Los scripts detectan
problemas mecánicos, pero un texto puede ser correcto y aun así no tener
sentido, y eso solo lo ve una persona.

### 4. Probar sin el corpus completo

Si quieres tocar el código sin tener el corpus entero a mano, basta con recrear
la estructura de carpetas y copiar dos archivos de niveles de zoom distintos:

```bash
mkdir -p mini/tiles/3/2 mini/tiles/5/10
cp "$CORPUS"/F3_*/Amazon_Underworld/tiles/3/2/AMAZONUW_3.pbf   mini/tiles/3/2/
cp "$CORPUS"/F3_*/Amazon_Underworld/tiles/5/10/AMAZONUW_15.pbf mini/tiles/5/10/
python decodificador_mvt.py mini/ mini.parquet "$INDICE"
```

Que sean de dos zooms distintos no es un capricho: es lo que permite comprobar
que el módulo detecta correctamente cuando un mismo municipio aparece
repetido en varios archivos y se queda con una sola copia.

La estructura de carpetas sí importa. El script deduce la posición de cada
mapa a partir de las dos carpetas que lo contienen y del número que lleva el
nombre del archivo, así que si lo sacas de ahí deja de reconocerlo.

---

## Qué produce

El resultado es `metadata_mapas.jsonl`. Es un archivo de texto donde **cada
línea es un fragmento independiente**, escrito en formato JSON. Una línea por
municipio, 985 en total.

Cada línea lleva los ocho campos que el reto exige para todos los fragmentos,
más algunos añadidos que pueden ser útiles después:

```json
{
  "doc_id": "F3-AMAZONUW-014",
  "chunk_id": "F3-AMAZONUW-014-chunk-0000",
  "fuente": "F3_Dinamicas_Territoriales/Amazon_Underworld/tiles/3/2/AMAZONUW_3.pbf",
  "formato": "pbf",
  "fenomeno": 3,
  "posicion": 0,
  "num_tokens": 166,
  "texto": "Amapá (Amapa) é um município do estado do Amapá (Amapa), Brasil, ...",
  "pais": "Brasil", "nivel1": "Amapá", "nivel2": "Amapá",
  "id_municipio": "amapa-amapa", "n_grupos": 2, "n_facciones": 3,
  "idiomas": ["es","en","pt"], "zoom": 3, "n_palabras": 108,
  "observatorio": "Amazon_Underworld", "nivel": "municipio",
  "encoder_tokens": "BAAI/bge-m3"
}
```

Los dos campos de conteo conviene explicarlos, porque cuentan cosas distintas.
`n_grupos` son las organizaciones armadas grandes, las que el mapa marca con
una casilla propia: el ELN, el Comando Vermelho, el PCC. `n_facciones` son los
frentes y bandas locales con nombre propio, que el mapa solo menciona en un
texto suelto: *Frente Rodrigo Cadete*, *Los Villanos del Tahuamanu*.

Ambos números los calcula la misma función que redacta el texto, y eso es
deliberado: así es imposible que el texto diga una cosa y los campos digan
otra. Antes se calculaban por separado y llegaron a contradecirse.

Así es como queda un fragmento ya escrito:

> Amapá (Amapa) é um município do estado do Amapá (Amapa), Brasil, com uma
> população de 8.440 habitantes e uma extensão de 8.454 km² na Amazônia
> brasileira. Según el observatorio Amazon Underworld, en este municipio de la
> Amazonía brasileña se documenta la presencia de dos estructuras armadas:
> Comando Vermelho (CV) y Primeiro Comando da Capital (PCC). También se
> identifican las estructuras locales Família Terror do Amapá (FTA), Amigos
> para Sempre (APS) y União do Crime Amapá (UCA). El municipio fue objeto de
> investigación directa en terreno por parte del observatorio. Armed group
> presence in Amapá, Amapá State, Brazil, in the Brazilian Amazon, is
> documented by the Amazon Underworld observatory.

**Nada de esto se versiona.** Ni `metadata_mapas.jsonl` ni los archivos
auxiliares (`municipios.parquet`, `municipios.muestra.csv`,
`muestra_revision.txt`) están en el repositorio, y es a propósito: se regeneran
en segundos, y lo que se entrega es solo la estructura de `entrega/` que define
la Sección 1.4 del documento técnico. Su contenido ya vive dentro de
`entrega/base_vectorial/encoder_bge-m3/metadata.jsonl`; tener dos copias del
mismo texto solo lleva a que acaben diciendo cosas distintas.

El paso que mete estos fragmentos en la base vectorial está en
[`INTEGRACION.md`](INTEGRACION.md). **No es un `cat`**: hay que codificarlos y
añadirlos al índice FAISS en el mismo orden, o el sistema empieza a devolver
el texto equivocado sin avisar.

---

## Diseño y decisiones

### Qué contienen estos mapas en realidad

Conviene aclararlo desde el principio, porque el nombre engaña. **Esto no es
"un mapa" en el sentido de un dibujo con carreteras y ciudades: es una base de
datos sobre presencia de grupos armados**, municipio a municipio, en seis
países de la cuenca amazónica.

De cada municipio se sabe su población, su superficie, once casillas de sí/no
que indican qué organizaciones armadas operan allí (ELN, las dos ramas de las
disidencias de las FARC, Clan del Golfo, Segunda Marquetalia, Comando Vermelho,
PCC, Los Choneros, Los Lobos…) y un texto libre con los frentes concretos.

Ese texto libre es la parte más valiosa. Ahí aparecen nombres como *Frente
Rodrigo Cadete*, *Família Terror do Amapá*, *Los Colochos de Pucca* o *Los
Villanos del Tahuamanu*, que **no están en ninguna de las casillas** y que
tampoco aparecen en ningún otro documento del corpus. Son términos muy poco
frecuentes, y eso los hace especialmente fáciles de encontrar para un buscador
cuando alguien pregunta por ellos.

### Las decisiones que se tomaron, y cómo dar marcha atrás

**1. Un fragmento por municipio.** Salen 985 fragmentos de entre 75 y 133
palabras. Se eligió así porque coincide con la forma en que la gente pregunta
por estos datos, y porque cada fragmento se entiende solo, sin necesitar el
anterior ni el siguiente. La pega es que son cortos comparados con el resto del
corpus. También se valoró agrupar por departamento (daría unos 60 fragmentos,
más largos) o uno por archivo de mapa (solo 73, pero superarían diez veces el
límite de 250 palabras).

**2. La información se saca de los mapas, no del Excel que venía con ellos.**
ADL incluyó un archivo `AMAZONUW_amazonunderworld-data.xlsx` que parece contener
lo mismo, pero está incompleto: solo algunas filas traen los datos de grupos
armados. Además tiene los acentos corrompidos (`BogotÃ¡`, `MamorÃ©`), un
problema clásico de codificación de caracteres. Leyendo los mapas directamente
los acentos salen bien. El Excel se usó solo para comprobar que el número total
de municipios cuadraba.

**3. Cuando un municipio aparece repetido, se conserva una sola copia.** Los
mapas web se organizan por niveles de zoom: el mismo territorio se guarda
varias veces, con más o menos detalle según cuánto te acerques. Eso hace que
cada municipio aparezca unas 8,7 veces repartido entre los distintos niveles.

De todas esas copias se conserva **la del nivel más alejado**, el que abarca
más territorio. Resulta que con ese criterio los 985 municipios caben en solo
3 archivos, que entre los tres cubren el dataset entero. Si hiciera falta
cambiar de criterio, la tabla intermedia guarda en la columna `fuentes_todas`
todos los archivos donde apareció cada municipio, así que se puede rehacer sin
volver a procesar los 73 originales.

**4. El campo `fuente` lleva la ruta completa, no solo el nombre del archivo.**
Esto tiene su motivo. ADL repite los nombres: los 73 documentos usan entre
todos solo 13 nombres distintos, porque `AMAZONUW_15.pbf` existe a la vez en
cuatro carpetas diferentes. Como la corrección del reto identifica los
documentos por este campo, poner solo el nombre haría que 73 documentos
distintos parecieran 13. Se controla con `FUENTE_CON_RUTA`.

**5. Los fragmentos combinan tres idiomas.** El cuerpo va en español; los
municipios brasileños abren en portugués; y todos terminan con una frase en
inglés que resume dónde está el municipio y qué se documenta allí. Los nombres
se escriben con tilde y sin ella cuando difieren —`Bolívar (Bolivar)`— porque
la gente suele buscar sin tildes y para un buscador son palabras distintas.

Se descartó hacer tres versiones separadas del mismo fragmento, una por idioma:
triplicaría el tamaño del índice y las tres podrían acabar ocupando puestos del
resultado diciendo exactamente lo mismo.

Conviene saber que **la razón por la que se hizo así podría no sostenerse**: el
reto anunciaba consultas en tres idiomas, pero las 50 que ADL entregó en
`Extracto_Preguntas_50_v2.pdf` están **todas en español** (las 50 empiezan por
`¿` y ninguna contiene *what*, *how* ni *which*).

Si eso se confirma, el inglés y el portugués ocupan una cuarta parte de cada
fragmento en idiomas que ninguna consulta usa. Por eso existen los
interruptores `ANCLAJE_EN` y `APERTURA_PT`: ponerlos en `False` deja los
fragmentos solo en español (bajan de 86 a 65 palabras de media) y permite
medir si el resultado mejora. **No se ha cambiado el valor por defecto**,
porque si el conjunto definitivo de evaluación sí trae inglés o portugués,
quitarlos sería peor.

**6. Se incluyen también los municipios de los que no se sabe nada** (323 de
985). En esos casos el texto dice claramente que la falta de datos no significa
que no pase nada allí, sino que el observatorio no llegó a cubrirlo. Sirve por
si alguien pregunta justamente por los vacíos de información.

**7. No se repite un grupo que ya se ha nombrado.** Los frentes del texto libre
suelen coincidir con las casillas de sí/no, así que se comparan ignorando
tildes, paréntesis y mayúsculas, y solo se escriben los que aportan algo nuevo.
La etiqueta genérica "Otros grupos armados" se descarta si el texto libre ya
nombra grupos concretos, pero se conserva si no hay nada más, porque entonces
es la única señal de que allí ocurre algo.

**8. Todo el texto se arma con plantillas.** No interviene ningún modelo de
lenguaje generativo, que es justo lo que el reto prohíbe. Las traducciones que
hacen falta son un puñado de palabras fijas —seis países y cinco términos— y
las siglas (ELN, CV, PCC) son iguales en todos los idiomas. Es una tabla de
constantes, no traducción automática.

**9. Los tokens se cuentan con `BAAI/bge-m3`.** Es solo para rellenar el campo
`num_tokens`; aquí no se generan vectores. Si el equipo acaba usando otro
modelo, hay que cambiar `MODELO_ENCODER` y repetir el paso 2.

---

## Configuración

Estas son las constantes que se pueden ajustar, todas al principio de
`generador_chunks.py`:

| Constante | Por defecto | Qué controla |
|---|---|---|
| `MODELO_ENCODER` | `BAAI/bge-m3` | con qué modelo se cuentan los tokens. **Tiene que ser el mismo que use el equipo** |
| `FUENTE_CON_RUTA` | `True` | si `fuente` lleva la ruta completa o solo el nombre del archivo |
| `ANCLAJE_EN` | `True` | si se añade la frase final en inglés |
| `APERTURA_PT` | `True` | si los municipios de Brasil abren en portugués |
| `FORMATO` | `"pbf"` | el valor del campo `formato` |
| `CORRECCIONES` | — | tabla de nombres a los que les falta la tilde (`Sucumbios → Sucumbíos`) |

Después de cambiar cualquiera de ellas hay que **volver a ejecutar el paso 2**,
pero no el 1: la lectura de los mapas no cambia. Son unos segundos.

Dos avisos sobre estas constantes. `ANCLAJE_EN` y `APERTURA_PT` no son
cuestión de estilo: entre las dos mueven una cuarta parte del texto de cada
fragmento, así que conviene medir el efecto antes de tocarlas. Y con
`CORRECCIONES` hay que ir con cuidado en sentido contrario: añadir una tilde
donde no toca estropea el nombre en lugar de arreglarlo, así que cada entrada
se comprobó a mano.

---

## Problemas frecuentes

**`No se extrajo ningun municipio. Revisa la ruta.`**
La carpeta que le pasaste no contiene archivos `.pbf`, o están colocados de una
forma que el script no reconoce. Espera encontrarlos dentro de dos carpetas
numeradas, así: `tiles/3/2/AMAZONUW_3.pbf`. El script avisa uno por uno de los
archivos que no consigue situar. Para ver cómo están organizados realmente:
`find "$CORPUS" -name "*.pbf" | head`.

**`Sin inventario de ADL: doc_id y fuente se derivaran de la ruta.`**
No le pasaste el Excel de ADL, o no lo encontró. El módulo funciona igual, pero
**no conviene entregarlo así**: sin ese archivo, los identificadores de
documento se los inventa a partir de la ruta de tu ordenador, y son justamente
los que se comparan al corregir. Pásalo siempre como tercer argumento.

**`No se pudo cargar el tokenizer de BAAI/bge-m3`**
Falta instalar `transformers` o no hay conexión. No es grave para trabajar: el
campo `num_tokens` pasa a contar palabras y todo lo demás sigue igual. Pero hay
que regenerarlo con el tokenizador de verdad antes de la entrega.

**El número de municipios sale muy distinto de 985.**
Si salen bastantes menos, el filtro que descarta los polígonos sin datos está
tirando cosas que sí valían. Si salen bastantes más, el módulo no está
reconociendo que dos copias son el mismo municipio; lo más probable en ese caso
es que haya acentos corrompidos en los identificadores, lo que parte un
municipio en dos.

**Algún archivo de mapa no se puede leer.**
Está contemplado y se informa por pantalla. Bastantes de los mapas originales
no llegaron a descargarse y en su lugar hay un archivo de error de 28 bytes; se
descartan por tamaño antes de intentar abrirlos. Otros pueden venir
comprimidos, y esos se descomprimen solos.

---

## Para quien integra esto en el pipeline central

- El orden de las líneas del archivo de fragmentos **tiene que coincidir con el
  orden en que se insertan los vectores en FAISS**. Si alguien reordena, filtra
  o elimina duplicados en una de las dos listas sin hacer lo mismo en la otra,
  el sistema devolverá el texto equivocado para cada resultado aunque los
  vectores sean perfectos. Es el error más común y el más difícil de detectar,
  porque no falla: simplemente responde mal.
- **Los prefijos que exigen algunos modelos los pone quien codifica**, no este
  módulo. Si estos fragmentos van sin prefijo y los del resto del equipo sí lo
  llevan, quedan en desventaja y no será evidente por qué.
- Estos fragmentos **ya se entienden por sí solos**. No hace falta añadirles
  título ni sección antes de codificarlos, como sí conviene hacer con los que
  salen de un PDF largo.
- `municipios.parquet` es un buen punto de partida para el grafo de
  conocimiento opcional: las relaciones ya están explícitas en las casillas de
  sí/no, así que no hace falta extraerlas del texto.
