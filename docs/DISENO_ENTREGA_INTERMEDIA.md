# Diseño de la entrega intermedia

## Pregunta que responde

El experimento busca responder tres preguntas relacionadas, pero distintas:

1. ¿Qué diferencia existe entre Regresión Logística, XGBoost y Qwen3-8B cuando los tres reciben las mismas veintinueve variables estructuradas?
2. ¿Los ocho ejemplos mejoran a Qwen frente a la modalidad sin ejemplos?
3. ¿Qué cambia dentro de Qwen3-8B cuando a esas variables se agrega la descripción libre del negocio?

La Regresión Logística se conserva como referencia lineal. Recibe el mismo bloque
estructurado que XGBoost y Qwen, pero no recibe texto.

## Comparaciones autorizadas

| Configuración | TransUnion | Formulario directo | Descripción libre |
|---|---:|---:|---:|
| Regresión Logística — estructurado | 20 | 9 | 0 |
| XGBoost — estructurado | 20 | 9 | 0 |
| Qwen3-8B — estructurado, sin ejemplos | 20 | 9 | 0 |
| Qwen3-8B — estructurado y descripción, sin ejemplos | 20 | 9 | 1 |
| Qwen3-8B — estructurado, ocho ejemplos | 20 | 9 | 0 |
| Qwen3-8B — estructurado y descripción, ocho ejemplos | 20 | 9 | 1 |

Esto produce seis filas de resultados. No se admite ninguna otra combinación en
la tabla de la entrega.

## Por qué se eligió este diseño

Los tres modelos reciben los veinte atributos del buró y nueve declaraciones
directas del formulario. El contraste permite observar la diferencia entre una
combinación lineal, un modelo capaz de representar relaciones no lineales y un
modelo de lenguaje sin cambiar las columnas disponibles.

La descripción libre se agrega solamente a Qwen. En esta entrega no se transforma
el texto para incorporarlo a Regresión Logística o XGBoost. Hacerlo correctamente
exigiría definir una representación numérica de significado, por ejemplo un
*embedding*, ajustar cualquier reducción con entrenamiento y validación, y evaluar
en una nueva prueba temporal. Esa extensión queda para un trabajo posterior.

## Campos almacenados como texto

La fuente contiene cuatro columnas de tipo texto, pero solo
`descripcion_negocio` se utiliza como narración libre. `subcategoria_texto`
funciona como categoría, `tipo_credito` está casi completamente concentrado en un
único valor y `otra_categoria_negocio` mezcla categorías con respuestas breves y
repite parte de la descripción en numerosos casos. No se concatenan esos campos ni
se los presenta como cuatro relatos equivalentes.

Un trabajo posterior podrá normalizar esas etiquetas por significado o producir
vectores semánticos para incorporarlos a los métodos clásicos. La transformación
deberá fijarse con entrenamiento y validación y medirse en una nueva prueba temporal.

## Definición del desenlace

`target=1` significa mora mayor de 60 días dentro de los primeros 150 días de
observación. `target=0` significa pago normal, con atraso máximo de hasta 30 días.
Los casos con atrasos entre 31 y 60 días no integran la comparación principal.

La cohorte contiene 4.201 créditos: 3.360 de entrenamiento, 420 de validación y
421 de prueba. La separación respeta el orden temporal.

## Qwen3-8B

Qwen recibe las veintinueve variables como una secuencia de nombres legibles y
valores. Los códigos negativos de TransUnion se traducen como “sin historial”. El
perfil textual añade al final solamente la descripción del negocio.

Se evalúan dos modalidades. La primera no contiene ejemplos resueltos. La segunda
incluye ocho ejemplos del entrenamiento, cuatro morosos y cuatro normales,
seleccionados por semejanza usando las mismas veintinueve variables. Los ejemplos
no cambian los parámetros del modelo y ningún caso de prueba puede aparecer entre
ellos.
