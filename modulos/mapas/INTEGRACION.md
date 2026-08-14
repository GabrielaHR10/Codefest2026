# Cómo entran los mapas en la base vectorial

Este documento explica el único paso que conecta el módulo de mapas con el
resto del sistema. Si solo quieres saber cómo se generan los fragmentos, está
en [`README.md`](README.md).

## El problema que resuelve

FAISS **no guarda los textos**. Guarda vectores numerados del 0 al N−1 y nada
más. El archivo `metadata.jsonl` es lo único que dice a qué fragmento
corresponde cada número, y la correspondencia es puramente posicional: la línea
42 describe al vector 42.

Por eso **no vale juntar los archivos con `cat`**. Si añades 985 líneas de
metadata sin añadir sus 985 vectores, esas líneas se quedan sin nada detrás. El
sistema no falla ni avisa: simplemente empieza a devolver el texto equivocado
para cada resultado. Es el error más común en este tipo de pipeline y el más
difícil de detectar, porque todo parece funcionar.

## El paso

Primero se generan los fragmentos (los tres pasos del [`README.md`](README.md))
y después se integran:

```bash
pip install -r modulos/mapas/requirements.txt

python modulos/mapas/decodificador_mvt.py "$CORPUS" municipios.parquet "$INDICE"
python modulos/mapas/generador_chunks.py  municipios.parquet "$INDICE"

python modulos/mapas/integrar_mapas.py entrega/base_vectorial/encoder_bge-m3
```

El último comando busca `metadata_mapas.jsonl` en el directorio actual, que es
donde lo deja el paso anterior. La primera vez descarga el modelo
`BAAI/bge-m3`, unos 2 GB.

**Ninguno de esos archivos intermedios se versiona**, y es a propósito: se
regeneran en segundos, y su contenido ya está dentro del `metadata.jsonl` que
sí se entrega. Tener dos copias del mismo texto en el repositorio solo lleva a
que acaben diciendo cosas distintas.

Lo que hace, en orden:

1. Comprueba que el índice y la metadata **ya estaban alineados**. Si no lo
   estaban, se detiene: no tiene sentido añadir nada encima de un desajuste.
2. Comprueba si los mapas ya estaban integrados. Si lo estaban, no hace nada,
   así que se puede ejecutar dos veces sin estropear nada.
3. Codifica los 985 fragmentos con el mismo modelo del índice y verifica que
   los vectores salen normalizados.
4. Guarda una copia de seguridad (`.bak`) de los dos archivos.
5. Añade los vectores al final del índice y las líneas al final de la
   metadata, **en el mismo orden**.
6. Verifica el resultado.

## Cómo sabemos que quedó bien

Al terminar hace tres comprobaciones, y la tercera es la que de verdad importa:

```
OK: 9886 vectores y 9886 lineas
OK: las 8901 lineas previas conservan su posicion
OK: 20 vectores al azar recuperan su propia linea
```

La última coge vectores al azar del índice, los busca dentro del propio índice
y comprueba que el resultado más parecido a cada uno es él mismo. Es la única
forma de detectar un desfase de posiciones: si la metadata estuviera corrida
aunque fuera una línea, esta prueba lo cantaría.

Si algo falla, el script termina con error y quedan los `.bak` para volver
atrás.

## Estado actual

Ya ejecutado. La base vectorial pasó de **8.901 a 9.886 fragmentos**; los 985
nuevos ocupan las posiciones 8901 a 9885.

Comprobado además que la recuperación funciona de extremo a extremo: al
codificar la consulta *"¿Qué grupos armados operan en los municipios de la
Amazonía brasileña?"* los cinco primeros resultados son municipios amazónicos
brasileños, con puntuaciones entre 0,63 y 0,65.

## Dos cosas que hay que acordar con el equipo

**Los prefijos del encoder.** Algunos modelos exigen que los fragmentos vayan
precedidos de una etiqueta (`passage: ` en la familia E5). BGE-m3 no lo
necesita, así que aquí se codifica el texto tal cual, igual que se hizo con los
fragmentos que ya estaban. Si en algún momento se cambia de modelo, esto hay
que revisarlo: codificar unos fragmentos con prefijo y otros sin él los manda a
zonas distintas del espacio y hunde su recuperación sin que se note por qué.
Está en la constante `PREFIJO_PASAJE`.

**Quién es dueño del orden final.** Este script añade siempre al final y nunca
reordena, que es la forma segura de hacerlo. Pero si otra persona regenera el
índice desde cero, o reordena la metadata, hay que volver a ejecutarlo. La
regla es que **una sola persona controle ese paso**.
