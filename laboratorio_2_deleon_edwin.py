"""
CC3103 - Procesamiento de Lenguaje Natural
Laboratorio 2: Motor de Busqueda Semantica

Entrega: Edwin De Leon (Giovanni Alejandro)
Dominio del corpus: libros de ficcion (descripciones, recomendaciones y tramas)

Objetivos:
- Generar embeddings de oraciones usando un modelo local de Hugging Face.
- Calcular similitud coseno entre una consulta y un corpus.
- Devolver resultados top-k.
- Comparar busqueda semantica contra busqueda por palabras clave.

Instalacion requerida:
    pip install sentence-transformers numpy scikit-learn

Ejecutar:
    python laboratorio_2_deleon_edwin.py
"""

from __future__ import annotations

from typing import Iterable
import re

import numpy as np


MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# -----------------------------------------------------------------------------
# 1. Corpus De Prueba (dominio: libros de ficcion)
# -----------------------------------------------------------------------------

CORPUS = [
    "Una joven maga descubre que es la unica capaz de derrotar a un senor oscuro.",
    "Un detective privado investiga el asesinato de un empresario en una ciudad lluviosa.",
    "Una nave espacial se pierde en una galaxia desconocida mientras la tripulacion busca el camino a casa.",
    "Dos familias enemigas se enfrentan mientras sus hijos se enamoran en secreto.",
    "Un grupo de sobrevivientes lucha contra una plaga de no muertos en una ciudad abandonada.",
    "Un ladron experto planea el robo perfecto en el banco mas vigilado del pais.",
    "Una nina descubre un mundo magico al atravesar un armario antiguo.",
    "Un soldado regresa de la guerra y debe reconstruir su vida junto a su familia.",
    "Una bruja es exiliada de su aldea y jura vengarse de quienes la traicionaron.",
    "Un cientifico crea una inteligencia artificial que empieza a cuestionar su propia existencia.",
    "Un pirata busca un tesoro legendario escondido en una isla maldita.",
    "Una reportera investiga una conspiracion que involucra al gobierno de la ciudad.",
    "Un vampiro centenario se enamora de una humana mortal en la epoca moderna.",
    "Un principe debe elegir entre el trono y el amor de una plebeya.",
    "Un grupo de amigos queda atrapado en un bucle temporal durante sus vacaciones.",
    "Un escritor sin inspiracion descubre que sus historias empiezan a hacerse realidad.",
    "Una expedicion cientifica encuentra criaturas desconocidas en lo profundo del oceano.",
    "Un huerfano descubre que es el heredero de un reino olvidado.",
    "Un asesino a sueldo decide retirarse despues de un ultimo trabajo que sale mal.",
    "Una familia se muda a una casa embrujada con un oscuro secreto.",
    "Un viajero del tiempo intenta evitar una catastrofe que ya ocurrio en el pasado.",
    "Dos hermanas compiten por el trono de un reino en guerra civil.",
    "Un robot abandonado en un planeta desierto busca reunirse con su creador.",
    "Un joven granjero descubre poderes magicos ocultos que heredo de su madre.",
]


CONSULTAS = [
    # Coincidencia directa de palabras (comparte terminos con el corpus)
    "libro sobre una maga que enfrenta a un senor oscuro",
    # Sin coincidencia directa, pero con relacion semantica clara
    "historia de amor prohibido entre dos familias rivales",
    # Consulta ambigua (puede relacionarse con varios libros del corpus)
    "una historia sobre secretos ocultos",
    # Caso donde keyword search deberia fallar por vocabulario distinto
    "un robo que sale perfecto en un lugar muy vigilado",
    # Caso adicional: sinonimos y parafraseo (sin coincidencia literal)
    "un romance entre un ser inmortal y una persona comun",
    # Dominio ciencia ficcion con vocabulario distinto al corpus
    "maquina pensante que se pregunta si es consciente",
]


# -----------------------------------------------------------------------------
# 2. Utilidades De Instalacion
# -----------------------------------------------------------------------------

def cargar_modelo(nombre_modelo: str):
    """Carga un modelo de sentence-transformers.

    Si la dependencia no esta instalada, muestra una instruccion clara.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        mensaje = (
            "No se encontro la libreria 'sentence-transformers'.\n"
            "Instale las dependencias con:\n\n"
            "    pip install sentence-transformers numpy scikit-learn\n\n"
            "Si usa Google Colab:\n\n"
            "    !pip install sentence-transformers numpy scikit-learn\n"
        )
        raise SystemExit(mensaje) from exc

    return SentenceTransformer(nombre_modelo)


# -----------------------------------------------------------------------------
# 3. Embeddings
# -----------------------------------------------------------------------------

def generar_embeddings(modelo, textos: list[str]) -> np.ndarray:
    """Genera embeddings para una lista de textos.

    normalize_embeddings=True permite que el producto punto sea equivalente
    a similitud coseno para muchos usos practicos.
    """
    embeddings = modelo.encode(textos, normalize_embeddings=True)
    return np.asarray(embeddings)


# -----------------------------------------------------------------------------
# 4. Similitud Coseno Y Ranking
# -----------------------------------------------------------------------------

def similitud_coseno(embedding_consulta: np.ndarray, embeddings_corpus: np.ndarray) -> np.ndarray:
    """Calcula similitud coseno entre una consulta y todos los documentos.

    Como los embeddings estan normalizados, el producto punto equivale a la
    similitud coseno.
    """
    return embeddings_corpus @ embedding_consulta


def buscar_semanticamente(
    consulta: str,
    corpus: list[str],
    embeddings_corpus: np.ndarray,
    modelo,
    top_k: int = 3,
) -> list[dict]:
    """Devuelve los top-k textos semanticamente mas similares a la consulta."""
    embedding_consulta = generar_embeddings(modelo, [consulta])[0]
    puntajes = similitud_coseno(embedding_consulta, embeddings_corpus)

    indices_ordenados = np.argsort(puntajes)[::-1][:top_k]

    resultados = []
    for posicion, indice in enumerate(indices_ordenados, start=1):
        resultados.append({
            "rank": posicion,
            "indice": int(indice),
            "texto": corpus[indice],
            "score": float(puntajes[indice]),
        })

    return resultados


# -----------------------------------------------------------------------------
# 5. Busqueda Por Palabras Clave
# -----------------------------------------------------------------------------

def tokenizar_simple(texto: str) -> set[str]:
    """Tokeniza texto para una busqueda por palabras clave simple."""
    return set(re.findall(r"\b\w+\b", texto.lower()))


def buscar_por_palabras_clave(consulta: str, corpus: list[str], top_k: int = 3) -> list[dict]:
    """Busca textos por cantidad de palabras compartidas con la consulta.

    Esta implementacion es intencionalmente simple para compararla contra
    busqueda semantica.
    """
    tokens_consulta = tokenizar_simple(consulta)
    resultados = []

    for indice, texto in enumerate(corpus):
        tokens_texto = tokenizar_simple(texto)
        coincidencias = tokens_consulta.intersection(tokens_texto)
        score = len(coincidencias)

        resultados.append({
            "indice": indice,
            "texto": texto,
            "score": score,
            "coincidencias": sorted(coincidencias),
        })

    resultados.sort(key=lambda item: item["score"], reverse=True)
    return resultados[:top_k]


# -----------------------------------------------------------------------------
# 6. Impresion De Resultados
# -----------------------------------------------------------------------------

def imprimir_resultados_semanticos(consulta: str, resultados: Iterable[dict]) -> None:
    print("\nBusqueda semantica")
    print(f"Consulta: {consulta}")

    for resultado in resultados:
        print(
            f"  {resultado['rank']}. "
            f"score={resultado['score']:.4f} | "
            f"{resultado['texto']}"
        )


def imprimir_resultados_keyword(consulta: str, resultados: Iterable[dict]) -> None:
    print("\nBusqueda por palabras clave")
    print(f"Consulta: {consulta}")

    for posicion, resultado in enumerate(resultados, start=1):
        coincidencias = ", ".join(resultado["coincidencias"]) or "sin coincidencias"
        print(
            f"  {posicion}. "
            f"score={resultado['score']} | "
            f"coincidencias={coincidencias} | "
            f"{resultado['texto']}"
        )


def comparar_busquedas(
    consulta: str,
    corpus: list[str],
    embeddings_corpus: np.ndarray,
    modelo,
    top_k: int = 3,
) -> None:
    """Imprime busqueda semantica y keyword search para una consulta."""
    print("=" * 100)
    print(f"CONSULTA: {consulta}")
    print("=" * 100)

    resultados_semanticos = buscar_semanticamente(
        consulta=consulta,
        corpus=corpus,
        embeddings_corpus=embeddings_corpus,
        modelo=modelo,
        top_k=top_k,
    )
    resultados_keyword = buscar_por_palabras_clave(consulta, corpus, top_k=top_k)

    imprimir_resultados_semanticos(consulta, resultados_semanticos)
    imprimir_resultados_keyword(consulta, resultados_keyword)
    print()


# -----------------------------------------------------------------------------
# 7. Analisis (respuestas del estudiante)
# -----------------------------------------------------------------------------

def ejercicio_3_analizar_resultados():
    """Analisis de resultados tras ejecutar el programa.

    (Respuestas basadas en la ejecucion real del programa con este corpus.)

    1. En que consulta funciono mejor la busqueda semantica?
       En "un romance entre un ser inmortal y una persona comun" (top-1:
       "Un vampiro centenario se enamora de una humana mortal...", score
       0.6383) y en "maquina pensante que se pregunta si es consciente"
       (top-1: "Un cientifico crea una inteligencia artificial que
       empieza a cuestionar su propia existencia...", score 0.6057).
       Ninguna de las dos consultas comparte palabras de contenido con la
       oracion correcta, pero la busqueda semantica igual la ubica en el
       primer lugar. Keyword search, en cambio, fallo por completo en
       ambos casos: solo encontro coincidencias en palabras funcionales
       ("un", "una", "que", "es"), sin relacion con el tema real.

    2. En que consulta funciono mejor keyword search?
       En "un robo que sale perfecto en un lugar muy vigilado", donde hay
       coincidencia literal fuerte ("robo", "perfecto", "vigilado") y
       ambos metodos coinciden en el mismo resultado top-1 con score alto
       (keyword=5 coincidencias, semantico=0.7473).

    3. Que resultado fue inesperado?
       En "libro sobre una maga que enfrenta a un senor oscuro", keyword
       search ubico correctamente en el top-1 la oracion sobre la maga y
       el senor oscuro (score=7, coincidencia casi literal). La busqueda
       semantica, en cambio, puso en el top-1 "Una nina descubre un mundo
       magico al atravesar un armario antiguo" (score 0.6318) y dejo la
       oracion correcta en el puesto 2 (score 0.5022): el modelo asocio
       fuertemente el concepto general de "mundo magico/fantasia" con la
       consulta, por encima de la coincidencia especifica de "senor
       oscuro". Es un caso donde la busqueda semantica, pese a ser mas
       robusta en general, fue superada por keyword search.

    4. Que cambiaria del corpus para mejorar la busqueda?
       Varias oraciones del corpus comparten estructura y vocabulario muy
       similar ("Un/Una ... descubre/busca/enfrenta un mundo magico/
       poderes/reino"), lo que acerca demasiado sus embeddings entre si y
       genera confusiones como la del punto 3. Agregaria mas detalles
       distintivos por oracion (nombres de subgeneros, elementos de trama
       unicos) para separar mejor los vectores de oraciones tematicamente
       parecidas pero distintas.
    """
    pass


# -----------------------------------------------------------------------------
# 8. Programa Principal
# -----------------------------------------------------------------------------

def main() -> None:
    print("Cargando modelo de embeddings...")
    modelo = cargar_modelo(MODELO_EMBEDDINGS)

    print("Generando embeddings del corpus...")
    embeddings_corpus = generar_embeddings(modelo, CORPUS)

    print(f"Corpus: {len(CORPUS)} oraciones")
    print(f"Dimension de embeddings: {embeddings_corpus.shape[1]}")

    for consulta in CONSULTAS:
        comparar_busquedas(
            consulta=consulta,
            corpus=CORPUS,
            embeddings_corpus=embeddings_corpus,
            modelo=modelo,
            top_k=3,
        )


if __name__ == "__main__":
    main()
