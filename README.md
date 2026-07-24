# NLP
================================================================================
# REFLEXIÓN CRÍTICA
================================================================================
- Durante las pruebas observé que las expresiones regulares ofrecen una primera
capa de protección útil, pero imperfecta, un falso positivo importante ocurre
con el DPI: por su longitud y sus guiones, el patrón general de teléfono también
puede reconocerlo como PHONE. 

- Por eso es necesario aplicar primero la redacción
específica de DPI. También pueden confundirse números de referencia, cuentas o
códigos extensos con datos sensibles, entre los falsos negativos, un correo
escrito con espacios, un teléfono expresado con palabras o una contraseña que
no incluya términos como “clave” o “token” podrían pasar sin ser detectados.

- Al redactar información se protege la identidad del usuario, aunque se pierde
contexto que podría ser útil para validar formatos, distinguir países o resolver
la solicitud. Por ello, la política debe equilibrar privacidad y utilidad.

- Regex no sería suficiente para una empresa real, porque los datos aparecen en muchos
formatos, idiomas y contextos, y los atacantes pueden ofuscarlos fácilmente, además, 
una coincidencia no permite determinar por sí sola la intención del
mensaje, agregaría una capa de clasificación contextual, validadores específicos
para cada tipo de dato y reglas según el nivel de riesgo. 

- También usaría control de acceso, cifrado, registros de auditoría y revisión humana para incidentes
críticos, finalmente, probaría el sistema de forma periódica con casos límite y
actualizaría los patrones a partir de errores reales.
