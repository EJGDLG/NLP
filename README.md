# NLP
================================================================================
# 6. Preguntas de análisis

> **Nota:** Las afirmaciones cuantitativas concretas, como qué token recibe mayor atención o con qué peso, deben interpretarse a partir de las tablas generadas después de ejecutar el notebook. A continuación se presenta la interpretación esperada de los resultados y su razonamiento.

## 1. ¿Qué tokens reciben mayor atención desde cada token seleccionado?

En la mayoría de las cabezas de atención, los tokens utilizados como consulta concentran una parte importante de su atención en los siguientes elementos:

* **`[SEP]` y `[CLS]`**: pueden funcionar como *attention sinks* o sumideros de atención. Cuando una cabeza no encuentra una relación especialmente relevante para una posición determinada, puede concentrar parte de su peso en estos tokens especiales.
* **Tokens adyacentes**: es frecuente observar atención hacia la palabra inmediatamente anterior o posterior, especialmente en las capas tempranas del modelo.
* **Núcleos sintácticos relacionados**: algunas cabezas muestran relaciones lingüísticamente plausibles. Por ejemplo:

  * Desde un verbo como `despertó` o `llamaban` hacia su sujeto, como `Gregorio` o `Samsa`.
  * Desde un sustantivo hacia su artículo o modificadores.
  * `ventana` puede atender a `la`.
  * `insecto` puede relacionarse con `monstruoso` o `un`.

Estos patrones muestran que ciertas cabezas capturan relaciones locales o sintácticas, aunque no todas tienen una interpretación lingüística directa.

---

## 2. ¿Cambian los patrones entre capas?

Sí. Los patrones de atención cambian de forma sistemática conforme aumenta la profundidad de la red.

### Capas tempranas, por ejemplo la capa 3

La atención suele ser principalmente **local y posicional**. Es común observar:

* Una diagonal marcada en las matrices de atención.
* Atención hacia tokens vecinos.
* Relaciones entre subpalabras pertenecientes a una misma palabra.
* Una distancia media de atención relativamente baja.

Estas capas parecen concentrarse principalmente en información local y en la estructura inmediata de la secuencia.

### Capas profundas, por ejemplo la capa 8

La atención suele ser más **dispersa y de largo alcance**. Pueden aparecer relaciones entre palabras alejadas dentro de la oración, como:

* Concordancia.
* Correferencia.
* Dependencias entre verbos y sus argumentos.
* Relaciones entre elementos separados por varias palabras.

También puede aumentar la concentración de atención sobre tokens especiales como `[SEP]` y `[CLS]`.

---

## 3. ¿Cambian los patrones entre cabezas?

Sí. Incluso dentro de una misma capa, las distintas cabezas de atención pueden presentar comportamientos diferentes.

Por ejemplo:

* Una cabeza puede concentrarse en el **token siguiente**.
* Otra puede prestar mayor atención al **token anterior**.
* Algunas pueden capturar relaciones como **artículo–sustantivo**.
* Otras pueden representar relaciones **preposición–término**.
* Algunas cabezas pueden dirigir gran parte de su atención hacia `[SEP]`, funcionando prácticamente como cabezas poco activas para determinados tokens.

En el experimento, las cabezas seleccionadas, como las cabezas **2 y 9**, pueden mostrar diferentes conjuntos de los cinco tokens con mayor atención para un mismo token de consulta.

Esto evidencia que las cabezas no realizan exactamente la misma función y pueden especializarse en diferentes tipos de relaciones.

---

## 4. ¿Las palabras con mayor atención son lingüísticamente relevantes?

**En algunos casos sí, pero no siempre.**

Existen cabezas cuyos tokens con mayor atención coinciden con relaciones lingüísticas razonables, por ejemplo:

* Verbo → sujeto.
* Sustantivo → adjetivo.
* Sustantivo → artículo.
* Preposición → complemento.

Sin embargo, también puede observarse una cantidad considerable de atención dirigida hacia tokens sin contenido léxico directo, como:

* `[CLS]`
* `[SEP]`
* Signos de puntuación.

Por esta razón, los pesos de atención no deben interpretarse directamente como un análisis sintáctico completo de la oración.

La atención puede reflejar relaciones útiles para el procesamiento interno del modelo sin que estas correspondan necesariamente con una relación lingüística interpretable por una persona.

---

## 5. ¿Qué diferencias aparecen entre oraciones simples y complejas?

En una **oración simple**, como `O1`, la atención suele encontrarse más concentrada debido a que las relaciones entre los elementos de la oración se producen a distancias relativamente cortas.

En las **oraciones complejas**, como `O2` y `O3`, que pueden incluir subordinadas y múltiples sintagmas:

* La atención se distribuye entre una mayor cantidad de tokens.
* Aparecen más relaciones de larga distancia.
* Un verbo principal puede atender a un argumento separado por una oración subordinada.
* La distancia media de atención tiende a aumentar.

Por lo tanto, la longitud y la complejidad sintáctica de la oración influyen en la forma en que el modelo distribuye su atención.

---

## 6. ¿Qué ocurre cuando una palabra se divide en subpalabras?

Los modelos basados en tokenización por subpalabras no necesariamente representan cada palabra mediante un único token.

Por ejemplo:

```text
parduzco → par ##du ##zco
```

Aunque para una persona `parduzco` representa una sola palabra, el modelo la procesa como varios tokens independientes.

Esto produce varias consecuencias:

* Los subtokens pertenecientes a una misma palabra pueden prestar **mucha atención entre sí**.
* Este comportamiento puede observarse como pequeños bloques cercanos a la diagonal de la matriz de atención.
* La atención correspondiente a una palabra completa ya no se encuentra en una sola fila de la matriz.
* Para analizar la palabra como una unidad, es necesario **agregar la atención de sus subtokens**.

En este análisis, la agregación puede realizarse mediante el promedio de las filas correspondientes a los subtokens.

Además, en muchos casos el primer subtoken puede conservar una mayor parte de la información sintáctica asociada a la palabra completa.

La tokenización también dificulta las comparaciones directas entre palabras de diferentes oraciones, ya que dos palabras pueden producir cantidades distintas de tokens.

---

## 7. ¿Qué NO puedes concluir observando únicamente los pesos de atención?

Los pesos de atención ofrecen información útil sobre cómo se relacionan los tokens dentro del modelo, pero presentan importantes limitaciones.

### Causalidad

Que un token reciba un peso alto de atención **no demuestra que sea la causa directa** de una predicción o de la representación final generada por el modelo.

### Importancia real

Los pesos de atención no consideran otros elementos importantes del mecanismo Transformer, entre ellos:

* Los *value vectors*.
* Las conexiones residuales.
* Las capas MLP.
* `LayerNorm`.
* Las transformaciones realizadas por las capas posteriores.

Por lo tanto, un peso de atención elevado no necesariamente significa que un token tenga la mayor influencia sobre la salida final.

### Explicación única

Es posible obtener resultados similares utilizando diferentes distribuciones de atención.

Esto significa que los pesos de atención no representan necesariamente una explicación única del comportamiento del modelo.

### Semántica y desambiguación

Si una palabra polisémica como `banco` presenta diferentes patrones de atención dependiendo de la oración, esto no permite determinar directamente qué significado seleccionó el modelo.

La información semántica se encuentra principalmente en los **estados ocultos y representaciones internas**, no exclusivamente en los pesos de atención.

### Sintaxis formal

Algunas cabezas pueden coincidir parcialmente con relaciones sintácticas conocidas, pero esto no significa que el modelo esté construyendo explícitamente un árbol sintáctico formal.

Las relaciones observadas:

* Dependen de la cabeza analizada.
* Pueden variar entre capas.
* No garantizan una estructura sintáctica coherente.

---

## Conclusión

Los pesos de atención son una herramienta útil para observar posibles relaciones entre los tokens y estudiar cómo cambia el procesamiento de información entre capas y cabezas del Transformer.

Sin embargo, deben interpretarse con precaución.

> **La atención puede considerarse una pista sobre el flujo de información dentro del modelo, pero no constituye por sí sola una explicación completa ni necesariamente fiel de las decisiones realizadas por la red neuronal.**
