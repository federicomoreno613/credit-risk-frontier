# Comparación de la predicción del riesgo crediticio entre un modelo de lenguaje con razonamiento y métodos clásicos de aprendizaje automático

*Comparación con datos del buró, variables directas del formulario y descripción libre del negocio*

**Universidad de Buenos Aires — Facultad de Ciencias Exactas y Naturales**  
**Maestría en Explotación de Datos y Descubrimiento de Conocimiento**  
Tesista: Lic. Federico Nicolás Moreno  
Director: Mgs. Boris Dorian Da Silva  
Codirector: Dr. Cristian Bravo  
Buenos Aires, 15 de julio de 2026

## 1. Introducción

El crédito es el monto que una institución financiera presta a una persona o a un negocio con el compromiso de que sea devuelto, generalmente con intereses y en cuotas. Como el pago futuro no se conoce al momento de la solicitud, las entidades utilizan modelos que asignan un puntaje relacionado con la probabilidad de incumplimiento. El **riesgo crediticio** es el fenómeno que se intenta medir; el **modelo de riesgo crediticio** es la herramienta utilizada para estimarlo. Su puntaje se combina con la política de riesgo y otras reglas de la institución, por lo que no decide por sí solo si el crédito se otorga o se rechaza.

Una parte importante de esa evaluación se apoya en el historial de pago registrado en el buró de crédito. Cuando esos antecedentes son escasos, existe menos información formal para ordenar el riesgo. En este trabajo, **historial escaso** significa que varios atributos del buró no tienen información. No equivale necesariamente a no estar bancarizado ni a solicitar un crédito por primera vez. El Global Findex 2025 informa que el 79 % de la población adulta mundial tiene una cuenta financiera y que en América Latina y el Caribe la proporción se acerca al 70 % (Demirgüç-Kunt et al., 2025). Tener una cuenta, sin embargo, no implica haber recibido un crédito.

Las empresas de tecnología financiera, conocidas como *fintech*, utilizan herramientas digitales para ofrecer servicios como pagos, ahorro o crédito (Financial Stability Board, 2017). Cada entidad define su propia política de aprobación. Por eso, en una base de créditos otorgados solo puede observarse qué ocurrió con las operaciones aprobadas y desembolsadas. Los resultados describen ese conjunto de créditos, no a todas las personas que solicitaron financiamiento.

Los modelos de lenguaje ofrecen una posibilidad adicional frente a los métodos tabulares: pueden leer datos no estructurados, como el texto libre de una solicitud. También pueden recibir datos estructurados si cada fila se convierte en una secuencia ordenada de texto, procedimiento conocido como serialización (Hegselmann et al., 2023). Sin embargo, comprender una descripción no implica que esa descripción mejore la predicción. El texto puede ser breve, repetitivo o contener información que ya aparece en otras variables.

Las solicitudes analizadas contienen veinte atributos del buró TransUnion, nueve variables numéricas directas del formulario y cuatro campos almacenados como texto. Solo uno de estos últimos, `descripcion_negocio`, funciona como una descripción libre escrita por la persona solicitante. Los otros tres se comportan principalmente como categorías o etiquetas.

La comparación utiliza la Regresión Logística, XGBoost y Qwen3-8B. Los tres reciben los mismos veinte atributos de TransUnion y las mismas nueve variables directas del formulario. De este modo, cuando se comparan los modelos con datos estructurados, cambia el método pero no la información disponible. En una segunda comparación, Qwen mantiene esas veintinueve variables y agrega únicamente `descripcion_negocio`. Los métodos clásicos no reciben el texto en esta entrega porque requieren una transformación numérica previa, que queda planteada como un paso posterior.

**El objetivo de este trabajo es comparar la capacidad de ordenar el riesgo de mora de la Regresión Logística, XGBoost y Qwen3-8B cuando reciben los mismos datos estructurados, y medir dentro de Qwen qué cambia al agregar la descripción libre del negocio.** La evaluación utiliza créditos antiguos para preparar los métodos y créditos más recientes para probarlos.

A partir de este objetivo se plantean tres preguntas:

1. ¿Qué diferencia existe entre la Regresión Logística, XGBoost y Qwen3-8B cuando reciben las mismas veintinueve variables estructuradas?
2. ¿La descripción libre del negocio agrega información útil dentro de Qwen?
3. ¿Ocho ejemplos resueltos del conjunto de entrenamiento ayudan a Qwen a ordenar mejor el riesgo que la modalidad sin ejemplos?

La medida principal es el área bajo la curva ROC (AUC). Un valor de 0,50 equivale a ordenar al azar y un valor de 1 representa un ordenamiento perfecto. El aporte de la descripción se resume mediante el cambio de AUC (ΔAUC) y se calcula sobre los mismos créditos con respuesta válida en ambas configuraciones. La definición del resultado también se mantiene fija: `target=1` significa mora mayor de 60 días dentro de los primeros 150 días de observación.

## 2. Marco teórico

Antes de describir el experimento hace falta ubicar el problema. La evaluación de un crédito parte de información incompleta y esa dificultad aumenta cuando el buró tiene pocos antecedentes. Sobre esa base se presentan los dos modelos clásicos y se explica qué cambia cuando un modelo de lenguaje puede leer la descripción del negocio. La capacidad de leer texto no evita los problemas propios de una evaluación crediticia, en especial la separación temporal y la selección previa de los créditos aprobados.

### 2.1 El problema del riesgo de crédito y la inclusión financiera

La puntuación de crédito, conocida también como *credit scoring*, consiste en estimar la probabilidad de que un solicitante no cumpla con sus obligaciones de pago. Se decide con la información incompleta disponible al momento de la solicitud, sin saber lo que ocurrirá después (Hand y Henley, 1997). La variable objetivo es binaria: vale 1 si el crédito entra en mora, es decir, si deja de pagarse, y 0 si se paga con normalidad; los datos de entrada son características del solicitante y de su negocio, y la probabilidad estimada alimenta la decisión de aprobar o rechazar. Ahora bien, el umbral que define la mora no es único, y para fijarlo existen acuerdos internacionales de regulación bancaria. Uno de los más destacados es el del Comité de Supervisión Bancaria de Basilea (BCBS), que establece criterios comunes para medir y gestionar el riesgo de crédito de las entidades financieras; según sus definiciones, se considera en incumplimiento a un deudor cuando la institución estima improbable el pago sin ejecutar garantías, o cuando registra un atraso superior a 90 días (BCBS, 2006).

En microfinanzas, donde las cuotas son pequeñas y frecuentes y los montos reducidos, son habituales los umbrales de 60 y 90 días (Cornelli et al., 2022; Chioda et al., 2024). Este trabajo adopta el umbral de 60 días y excluye la zona intermedia (entre 31 y 60 días de atraso) para separar con nitidez a quienes pagan de quienes incumplen, de modo que los casos ambiguos —atrasos transitorios que luego se regularizan— no contaminen la señal de aprendizaje.

El buró de crédito es la información que ciertas entidades centralizan sobre el historial de endeudamiento y pago de las personas, a partir de lo que reportan múltiples instituciones (Thomas et al., 2002). El problema estructural de las microfinanzas es que el buró presupone que el cliente ya tuvo crédito antes, de modo que para quien nunca accedió al sistema formal aporta poca o ninguna señal. Cuando el buró reúne pocos antecedentes se habla de historial escaso. La literatura en inglés utiliza *thin-file*, una expresión que alude a las antiguas carpetas físicas del buró: si había pocos registros, la carpeta era delgada. Este concepto describe la cantidad de información disponible; no indica por sí solo que la persona nunca haya tenido un crédito ni que esté fuera del sistema bancario.

### 2.2 Modelos clásicos para datos tabulares

Los datos de una solicitud de crédito son tabulares: filas (clientes) y columnas (variables numéricas y categóricas), como una planilla. Para este tipo de datos se utilizan habitualmente dos familias de modelos, que en este trabajo sirven como referencias.

La Regresión Logística es el modelo histórico de la industria. Estima la probabilidad de mora como una combinación lineal de las variables, transformada por una función logística que comprime el resultado al intervalo [0, 1]. Cada coeficiente tiene una interpretación directa —cuánto sube o baja el riesgo al mover una variable— y el modelo resultante es transparente y auditable, una propiedad decisiva en un dominio regulado. Es la base de las tarjetas de puntuación que describe Siddiqi (2017) y que siguen en uso en gran parte de la banca. Su límite es la rigidez: solo captura relaciones lineales y, salvo que se las construya a mano, no aprovecha las interacciones entre variables.

Los métodos de árboles potenciados por gradiente (*gradient boosting*) levantan esa restricción. La idea es construir cientos de árboles de decisión pequeños de forma secuencial, donde cada árbol nuevo se especializa en corregir los errores que cometieron los anteriores; la suma de todos ellos forma un predictor que captura interacciones complejas sin que haya que especificarlas de antemano. Su implementación más difundida es XGBoost (Chen y Guestrin, 2016). Su límite aparece con las variables categóricas de alta cardinalidad y con el texto: XGBoost puede aprender que la etiqueta "panadería" se asocia a cierto nivel de mora dentro del entrenamiento, pero no relaciona por sí solo ese rubro con otros semánticamente cercanos.

Lessmann et al. (2015) evaluaron 41 clasificadores sobre ocho conjuntos de datos de crédito y encontraron que los árboles potenciados por gradiente se ubicaban entre los métodos con mejor desempeño. En conjuntos tabulares de tamaño moderado, trabajos más recientes también muestran ventajas de los métodos basados en árboles frente a redes neuronales (Grinsztajn et al., 2022). Por esa razón XGBoost funciona aquí como una referencia exigente, no como un ganador supuesto de antemano.

### 2.3 Dos sesgos que condicionan la lectura de los datos

Dos problemas condicionan la lectura de los resultados: mezclar información de distintos momentos y observar solamente los créditos que fueron aprobados. En la literatura se los conoce como filtración temporal y sesgo de selección.

Un modelo de crédito se usa siempre para predecir el futuro con información del pasado (Hand y Henley, 1997), de modo que una evaluación realista debe entrenar con créditos antiguos y probar con posteriores. La razón no es sólo que un mismo cliente aparezca en ambos conjuntos (los créditos pueden ser de personas distintas), sino que todos los créditos otorgados en un mismo período comparten un contexto común: el nivel de inflación y actividad económica de ese momento, la política de aprobación que la empresa tecnológica financiera aplicaba entonces y los patrones estacionales del negocio (por ejemplo, la temporada alta de ciertos rubros). Si se mezclan al azar créditos de todas las fechas, el modelo aprende esas regularidades propias de cada período y luego se lo evalúa sobre créditos del mismo período, una información que no estará disponible al puntuar un cliente futuro. El desempeño medido resulta así optimista. Esta fuga de información temporal es el motivo por el que la comparación de este trabajo se hace respetando el orden cronológico. La validación cruzada aleatoria produce estimaciones sesgadas (Bergmeir y Benítez, 2012).

El segundo problema es que la cartera observada contiene solamente créditos aprobados y desembolsados. Para las solicitudes rechazadas no existe un resultado de pago y, por lo tanto, la base no permite saber cómo se habrían comportado. Las asociaciones estimadas describen a quienes atravesaron la política de aprobación, no a todas las personas que solicitaron financiamiento. La literatura denomina a esta restricción sesgo de selección muestral y propone técnicas de inferencia de rechazados (*reject inference*) para trabajar bajo supuestos adicionales (Hand y Henley, 1997). En esta investigación no se aplican esas técnicas ni se atribuye de manera causal una asociación concreta al proceso de selección. La consecuencia práctica es más acotada: los resultados no deben generalizarse fuera del conjunto de créditos aprobados analizado.

### 2.4 Modelos de lenguaje aplicados a datos tabulares

Los modelos de lenguaje de gran escala son redes entrenadas sobre grandes volúmenes de texto para anticipar la palabra siguiente. Esa tarea de entrenamiento les permite reconocer significados y relaciones del lenguaje. A diferencia de los modelos tabulares clásicos, pueden leer de manera directa una descripción como “venta de alimentos preparados” y relacionarla con conceptos aprendidos previamente.

La hipótesis que motiva su uso es que la descripción del negocio puede contener información que no aparece en las columnas numéricas. Esta expectativa es especialmente relevante cuando el buró tiene pocos antecedentes. Sin embargo, el conocimiento general del lenguaje no reemplaza el aprendizaje sobre la cartera concreta. Una relación que parece razonable en términos generales puede no repetirse entre los créditos que una entidad decidió aprobar.

Para que el modelo lea también las variables estructuradas, cada fila se convierte en una descripción ordenada. La técnica se conoce como TabLLM (Hegselmann et al., 2023). En lugar de enviar una planilla, se presentan frases como “ingresos mensuales del negocio: ...” o “atributo del buró: sin historial disponible”. Esta conversión permite mantener las mismas veintinueve variables y comparar una consulta sin la descripción del negocio con otra que la incorpora al final.

El modelo puede recibir la solicitud sin antecedentes resueltos dentro de la consulta. Esta modalidad se denomina *zero-shot* y en el documento se presenta como “sin ejemplos”. También puede recibir algunos créditos de entrenamiento con su resultado conocido antes del caso que debe evaluar. Esta segunda modalidad se denomina *few-shot* y aquí se presenta como “con ocho ejemplos”. Los ejemplos no modifican los parámetros del modelo: funcionan como referencias dentro de cada consulta.

Qwen3-8B (Qwen Team, 2025) es un modelo de pesos abiertos que puede ejecutarse de manera local. El número 8B indica aproximadamente ocho mil millones de parámetros. Su inclusión permite estudiar si un modelo que comprende texto, pero que no fue entrenado específicamente con toda la cartera, puede ordenar el riesgo a partir de las variables estructuradas y aprovechar además la descripción del negocio. Se lo evalúa sin ejemplos y con ocho ejemplos seleccionados únicamente desde el conjunto de entrenamiento.

### 2.5 Antecedentes en puntuación de crédito con modelos de lenguaje

Feng et al. (2023) evaluaron modelos de lenguaje generalistas sobre nueve conjuntos de datos de puntuación crediticia y los compararon con métodos clásicos. Sus resultados fueron mixtos y utilizaron métricas basadas principalmente en decisiones ya convertidas a clases. El presente estudio conserva una probabilidad continua para medir el ordenamiento mediante AUC.

Hegselmann et al. (2023) mostraron que la serialización de filas y el uso de pocos ejemplos pueden resultar útiles en distintos conjuntos tabulares. Ese antecedente no permite anticipar el resultado en una cartera de microfinanzas latinoamericana: el idioma, la definición de mora, el tamaño de la muestra y la selección previa de créditos aprobados son diferentes.

La revisión de Golec y AlabdulJalil (2026) muestra que el uso de modelos de lenguaje en riesgo de crédito es un campo activo. Aun así, sigue siendo necesario evaluar modelos locales sobre datos reales y con una separación temporal. Para estudiar el texto, además, conviene mantener fijos el modelo, los créditos y las variables estructuradas, de modo que el único cambio sea la incorporación de la descripción.

Los antecedentes no anticipan un único resultado. Los árboles potenciados suelen funcionar bien cuando existen miles de registros tabulares etiquetados, mientras que los modelos de lenguaje pueden reconocer significados que una columna numérica no representa de manera directa. En este trabajo, las familias se comparan con la misma base de veintinueve variables. La capacidad de leer lenguaje natural se estudia luego dentro de Qwen, agregando solamente la descripción libre.

## 3. Metodología

El estudio busca responder una pregunta acotada sin perder de vista cómo se formó el conjunto de créditos analizado. Por eso la metodología se presenta en el mismo orden en que se tomaron las decisiones: unión de fuentes, definición del desenlace, separación temporal, descripción exploratoria, selección de variables y comparación de modelos.

### 3.1 Diseño del estudio y alcance de la población

El estudio es retrospectivo y observacional: utiliza información reunida durante el funcionamiento habitual de la entidad y no modifica las condiciones bajo las cuales se otorgaron los créditos. La unidad de análisis es un crédito desembolsado. Por lo tanto, las conclusiones describen la cartera aprobada y no a todas las personas que solicitaron financiamiento.

Esta distinción tiene consecuencias importantes. Para los créditos rechazados no existe un resultado de pago, porque nunca fueron otorgados. No es posible saber cómo se habrían comportado. En consecuencia, el estudio no evalúa la política de aprobación completa ni permite afirmar que una asociación observada entre los aprobados se mantenga en la población general. La población efectiva está formada por los créditos que pudieron vincularse de manera exacta, completaron el tiempo de observación y quedaron fuera de la zona ambigua de mora.

La fuente de características contiene 4.897 operaciones. Reúne información declarada en el formulario, datos provenientes del buró TransUnion y cuatro campos almacenados como texto. Una segunda fuente contiene los pagos posteriores. Como sus identificadores no eran directamente compatibles, fue necesario reconstruir qué registro de pagos correspondía a cada crédito sin utilizar el desenlace para hacer la unión.

### 3.2 Vinculación de las fuentes y cobertura

La vinculación se realizó mediante una firma formada por 22 variables que estaban presentes en ambas fuentes: veinte atributos del buró, el indicador de género y la fecha. Una coincidencia se aceptó únicamente cuando la combinación aparecía una sola vez en cada archivo. Si había duplicados o dudas, el caso quedaba afuera. No se utilizaron aproximaciones por similitud ni correcciones manuales.

Este procedimiento vinculó 4.443 de los 4.897 créditos, es decir, el 90,7 %. Después de exigir una observación completa de 150 días y excluir los atrasos intermedios, el conjunto final quedó compuesto por 4.201 operaciones. La regla exacta reduce el tamaño, pero evita que un pago se asigne al crédito equivocado y permite reconstruir el proceso de manera automática.

Los 454 casos sin coincidencia exacta no se distribuyen de la misma manera que los incluidos. Entre los vinculados, el 74,9 % pertenece al segmento de historial esparso; entre los excluidos, el 46,0 %. También existen diferencias en una medida de mora ponderada del buró y en las fechas: los excluidos se concentran especialmente entre mayo y septiembre de 2023.

Estas diferencias no explican por sí mismas por qué faltan los vínculos y tampoco prueban un mecanismo causal. Sí muestran que el conjunto final no debe presentarse como una copia perfecta de la base original. Las conclusiones se refieren a las 4.201 operaciones elegibles. La comparación entre incluidos y excluidos se conserva como una limitación de cobertura.

**Cuadro 1. Formación del conjunto de créditos analizados.**

| Etapa | Créditos | Observación |
|---|---:|---|
| Fuente de solicitudes | 4.897 | Operaciones disponibles para el cruce |
| Vinculación exacta con pagos | 4.443 | 90,7 % de la fuente |
| Conjunto final | 4.201 | Ventana completa y sin zona intermedia |
| Entrenamiento | 3.360 | Período inicial |
| Validación | 420 | Período intermedio |
| Prueba | 421 | Período más reciente |

### 3.3 Definición de mora y ventana de observación

Para todos los modelos se adoptó una única convención: **1 significa mora y 0 significa pago normal**. Un crédito se considera moroso cuando el atraso máximo observable dentro de los primeros 150 días supera 60 días. Se considera normal cuando no supera 30 días. Los casos con atrasos entre 31 y 60 días se excluyen para no asignar a uno de los extremos situaciones intermedias que pueden evolucionar de manera distinta.

También se exige que cada crédito haya tenido la misma oportunidad de mostrar el desenlace. La ventana uniforme de 150 días evita comparar un préstamo observado durante pocas semanas con otro seguido durante cinco meses. El corte de observación es el 11 de mayo de 2024; los créditos que no completaron la ventana antes de esa fecha no ingresan en el conjunto principal.

**Figura 1. Distribución del desenlace en el conjunto final.** De los 4.201 créditos, 2.810 presentan mora mayor de 60 días dentro de 150 días y 1.391 corresponden a pago normal.

![Distribución del desenlace](analisis_E4_review/figuras_analisis/fig1_distribucion_simple.png)

La elección de 150 días quedó fijada antes de evaluar la prueba. Como comprobación adicional, se reconstruyó el desenlace con ventanas de 120, 180 y 210 días. El Cuadro 2 incorpora también la fila principal de 150 días para que la comparación utilice la misma estructura.

**Cuadro 2. Sensibilidad del conjunto de créditos y de la partición temporal al modificar la ventana de observación.** El asterisco identifica el horizonte principal del estudio.

| Horizonte (días) | Cantidad de créditos | % mora | Entrenamiento | Validación | Prueba |
|---:|---:|---:|---:|---:|---:|
| 120 | 3.954 | 61,2 | 3.163 | 395 | 396 |
| 150* | 4.201 | 66,9 | 3.360 | 420 | 421 |
| 180 | 4.165 | 70,3 | 3.332 | 416 | 417 |
| 210 | 3.896 | 72,6 | 3.116 | 389 | 391 |

El porcentaje observado de mora aumenta al ampliar la ventana. Este patrón es compatible con una mayor oportunidad para acumular atraso, aunque los conjuntos no son idénticos porque también cambian el seguimiento completo y la exclusión de la zona intermedia. Esta sensibilidad describe las consecuencias de cambiar el horizonte, pero no selecciona la definición que produce el mejor AUC.

### 3.4 Separación temporal de entrenamiento, validación y prueba

Los 4.201 créditos se ordenaron por fecha de desembolso. Los primeros 3.360 se destinaron al entrenamiento, los 420 siguientes a la validación y los últimos 421 a la prueba. En términos proporcionales, la división es 80 %, 10 % y 10 %. La lista de casos de cada parte quedó congelada en un manifiesto para impedir cambios posteriores.

El entrenamiento sirve para que los modelos aprendan. La validación permite escoger configuraciones sin tocar la prueba. La prueba se abre una sola vez, cuando las decisiones ya están tomadas, y representa el período más reciente. Este orden se parece al uso real: se aprende del pasado para estimar el futuro. Una división aleatoria habría mezclado períodos y podría haber dado una imagen demasiado optimista.

La tasa de mora cambia con fuerza a través del tiempo: es 70,1 % en entrenamiento, 79,0 % en validación y 29,2 % en prueba. Esto no vuelve inútil la evaluación. La hace más exigente y revela que el contexto de la cartera cambió. El AUC todavía permite evaluar si los casos se ordenan correctamente, pero las probabilidades pueden quedar demasiado altas si el modelo aprendió en un período con mucha más mora.

**Figura 2. Evolución mensual y frecuencia de mora por partición temporal.** `target=1` representa mora mayor de 60 días dentro de 150 días. La figura se calcula sobre los 4.201 casos del conjunto final.

![Evolución temporal de la mora](analisis_E4_review/figuras_analisis/fig2_temporal_fix.png)

La línea mensual permite ver que el cambio no es gradual ni uniforme. Existen meses con frecuencias muy altas y una caída pronunciada hacia el final. La figura describe el patrón observado; no permite atribuirlo a una modificación específica de la política, de la población o del contexto económico.

### 3.5 Datos utilizados en cada comparación

Solo se utilizaron datos disponibles antes de la decisión. Se excluyeron la antigüedad calculada desde el desembolso, las condiciones finales del crédito concedido, dos puntajes internos de la empresa, la fecha, la etiqueta de mora, la partición, los identificadores y el segmento de historial. También quedaron afuera tres relaciones económicas calculadas a partir de otras columnas. Esta decisión evita que una transformación derivada sea presentada como si fuera una declaración original.

La tabla analítica contiene 4.201 filas y 96 columnas, pero el experimento no entrega todas esas columnas a todos los modelos. La selección responde a la procedencia y a la función de cada dato. El Cuadro 3 resume los datos que recibe cada modelo.

**Cuadro 3. Datos estructurados y no estructurados utilizados en cada comparación.** La columna “Texto libre” identifica si se incorpora `descripcion_negocio`; ningún método clásico recibe ese campo en esta entrega.

| Configuración | Variables de TransUnion | Variables directas del formulario | Texto libre | Comparación que permite |
|---|---:|---:|---|---|
| Regresión Logística — datos estructurados | 20 | 9 | No | Referencia lineal sobre las mismas 29 variables |
| XGBoost — datos estructurados | 20 | 9 | No | Relaciones no lineales sobre las mismas 29 variables |
| Qwen3-8B — datos estructurados | 20 | 9 | No | Modelo de lenguaje sobre los mismos datos escritos como texto |
| Qwen3-8B — datos estructurados y descripción | 20 | 9 | Sí | Cambio dentro de Qwen al agregar datos no estructurados |

La Regresión Logística, XGBoost y Qwen reciben las mismas veintinueve variables estructuradas. Las veinte primeras provienen de una fuente externa a la declaración del cliente y las nueve restantes son respuestas numéricas del formulario. Esta igualdad permite que la comparación principal cambie el método y no la información de entrada.

Qwen recibe los mismos valores, pero no en forma de matriz numérica. Cada fila se convierte en una secuencia ordenada con el significado de la variable y su valor. La segunda configuración de Qwen mantiene esa secuencia y agrega solamente la descripción del negocio.

No se incluyen categorías codificadas, canales, alianzas, objetivos del crédito, nivel educativo ni género. Tampoco se incluyen ingresos estimados, flujo de caja libre ni la relación entre costos e ingresos, porque son cálculos derivados y no respuestas directas. El objetivo no es agotar todas las combinaciones posibles, sino construir una comparación que pueda explicarse columna por columna.

#### 3.5.1 Denominación oficial de los atributos de TransUnion

Los códigos de TransUnion no se interpretaron a partir de su abreviatura. El Cuadro 4 transcribe la columna “Descripción” de la hoja `Variables_CreditVision` del diccionario *Experto CreditVision V3.8* y conserva también la unidad o escala informada en la columna “Tipo Valor”. Los códigos se presentan en mayúsculas, como aparecen en el diccionario, aunque en la tabla analítica estén almacenados en minúsculas. Se mantiene la terminología de la fuente —incluidas expresiones como “instalamentos”, “Pago Inferido” y “transactor”— para no reemplazarla por un significado deducido.

**Cuadro 4. Denominación oficial completa de los veinte atributos de TransUnion utilizados.** La columna central reproduce la clasificación de uso y la unidad o escala del diccionario; la última conserva sin abreviar su descripción oficial.

| Código | Uso y unidad o escala | Denominación oficial completa |
|---|---|---|
| `AGG308` | Mora · $ Miles | Monto en mora agregado de obligaciones no hipotecarias en créditos financieros al mes M = 08 |
| `WD81` | Mora · Porcentaje | Mora ponderada en créditos financieros en el mes M = 01 |
| `AGG2503` | Plazo Inferido · Cantidad | Plazo Inferido (relación de saldo sobre la cuota minima) agregado en el mes M=03 |
| `UTLMAG04` | Utilización · Índice | Magnitud de utilización de obligaciones retail en los últimos 24 meses (índice que mide la tendencia de 0 a 600) |
| `DUEMAG01` | Otro · Índice | Magnitud total de todas las obligaciones en los últimos 24 meses (índice que mide la tendencia de 0 a 600) |
| `AEPMAG01` | Pago Inferido · Índice | Magnitud del exceso de Pago Inferido agregado no hipotecario en los últimos 24 meses (índice que mide la tendencia de 0 a 600) |
| `BI21S` | Apertura · Meses | Meses desde la más reciente apertura bancaria en instalamentos |
| `LMD34S` | Utilización · Porcentaje | Utilización de obligaciones bancarias vigentes sin garantía de mediano plazo reportadas en los últimos 12 meses |
| `RI27S` | Cuentas · Cantidad | Número de obligaciones actualmente vigentes y al día de retail instalamentos con 24 meses o más de antigüedad |
| `RLE904` | Pago Inferido · $ Miles con 3 decimales | Exceso de Pago Inferido en cuentas de hipotecario en los últimos 6 meses |
| `TEL32S` | Saldo · $ Miles con 3 decimales | Saldo máximo en obligaciones vigentes de telecomunicaciones reportadas en los últimos 12 meses |
| `TRANBAL09` | Saldo · $ Miles con 3 decimales | Saldo asignado a obligaciones identificadas como transactor al mes 9 |
| `AT104S` | Apertura · Porcentaje | Porcentaje de obligaciones aperturadas en los últimos 24 meses sobre el total de obligaciones |
| `SA21S` | Apertura · Meses | Meses desde la más reciente cuenta de ahorros aperturada |
| `AT103S` | Cuentas · Porcentaje | Porcentaje de obligaciones vigentes y al día del total de obligaciones |
| `TEL03S` | Cuentas · Cantidad | Número de obligaciones vigentes al día de telecomunicaciones |
| `AT34AF` | Utilización · Porcentaje | Utilización de obligaciones vigentes reportadas en los últimos 12 meses en créditos financieros |
| `G051S` | Mora · Porcentaje | Porcentaje de obligaciones que alguna vez estuvo en mora |
| `AGG9316` | Mora · $ Miles | Monto agregado en mora al mes M = 16 |
| `WD03` | Mora · Porcentaje | Mora ponderada en las obligaciones en el mes M = 06 |

Los valores negativos de estos atributos no representan montos negativos ni una disminución del riesgo. Son códigos de ausencia definidos por la fuente. Para los modelos clásicos se tratan como datos faltantes. En la serialización de Qwen se escriben como “sin historial”. Esta equivalencia conserva el significado real del dato aunque la representación sea distinta.

#### 3.5.2 Variables directas seleccionadas del formulario

Las nueve variables del formulario se eligieron por representar respuestas directas disponibles antes de la decisión y por tener una interpretación económica sencilla. No se eligieron mirando su relación con el desenlace de prueba. Todas son numéricas y se presentan en el Cuadro 5.

**Cuadro 5. Variables directas del formulario incluidas en los tres modelos.** Los montos se encuentran expresados en pesos colombianos según la fuente.

| Nombre en la tabla | Significado | Naturaleza del dato |
|---|---|---|
| `appusers_age` | Edad de la persona solicitante, en años | Declaración personal |
| `credits_dependants_amount` | Cantidad de personas económicamente a cargo | Declaración del hogar |
| `credits_family_expenses` | Gastos familiares mensuales | Monto declarado |
| `shops_monthly_incomes` | Ingresos mensuales del negocio | Monto declarado |
| `shops_monthly_outcomes` | Egresos mensuales del negocio | Monto declarado |
| `shops_daily_incomes` | Ingresos diarios del negocio | Monto declarado |
| `shops_initial_capital` | Capital inicial del negocio | Monto declarado |
| `shops_rent_amount` | Arriendo mensual del negocio | Monto declarado |
| `shops_shop_age` | Antigüedad del negocio, en años | Declaración sobre la actividad |

Estas variables no equivalen a estados contables auditados. La exploración muestra faltantes y valores extremos, por lo que deben interpretarse como información declarada durante la solicitud. Se incorporan a los tres modelos para sostener una base común. En la Regresión Logística se completan y estandarizan; XGBoost las recibe como valores numéricos con faltantes; Qwen las recibe escritas en la serialización.

#### 3.5.3 Campos almacenados como texto y elección de la descripción libre

La fuente contiene cuatro columnas de tipo texto: `subcategoria_texto`, `descripcion_negocio`, `otra_categoria_negocio` y `tipo_credito`. La inspección de su contenido mostró que no corresponden a cuatro narraciones equivalentes. La descripción del negocio es el campo libre de mayor variedad. La subcategoría funciona como una etiqueta con 37 valores, el tipo de crédito tiene solamente dos valores y está casi completamente concentrado en “Primer Crédito”, y el rubro declarado mezcla etiquetas con respuestas breves.

**Figura 3. Cobertura y extensión de los cuatro campos almacenados como texto.** La cobertura indica la proporción de los 4.201 casos con un valor no vacío; la extensión se resume por la mediana de caracteres entre los casos con dato.

![Cobertura y extensión de los campos textuales](credit-risk-frontier/figures/intermedia_20260714_redesign/03_campos_textuales.png)

La descripción del negocio está presente en el 82,8 % de los casos y tiene una mediana de 46 caracteres. La subcategoría y el tipo de crédito tienen cobertura completa, pero funcionan principalmente como etiquetas. El rubro declarado aparece en el 29,5 % de los casos y, cuando coincide con la descripción, repite con frecuencia el mismo contenido o una formulación cercana.

Por estas razones, el experimento textual utiliza solamente `descripcion_negocio`. Esta decisión evita presentar como “texto libre” un conjunto compuesto en gran parte por categorías, y evita unir campos distintos en una sola cadena. La escritura se conserva tal como fue registrada: no se corrigen errores ortográficos, no se resumen respuestas y no se asignan categorías con otro modelo de lenguaje.

Los tres campos restantes siguen siendo descriptos en el análisis exploratorio porque ayudan a comprender la base, pero no ingresan a la consulta evaluada. Una investigación posterior podría normalizarlos por significado o transformar la descripción mediante representaciones numéricas aprendidas. Esa alternativa requeriría un experimento nuevo y no se incorpora después de observar la prueba.

### 3.6 Lectura exploratoria del conjunto de datos

Antes de comparar modelos se realiza una lectura exploratoria del conjunto. Su función no es adelantar cuál será el mejor predictor, sino mostrar qué información existe, cómo se distribuye y qué problemas deben tenerse presentes al leer los resultados. Las asociaciones con la mora se calculan solo sobre los 3.360 créditos de entrenamiento. De este modo, la prueba temporal no se utiliza para elegir variables ni para construir explicaciones previas al resultado final.

El recorrido comienza por las etiquetas textuales. Se busca saber si las subcategorías reúnen negocios con comportamientos distintos, pero sin apoyar una tasa en grupos demasiado pequeños. Por eso se muestran solamente las que tienen al menos 40 créditos en entrenamiento.

**Figura 4. Frecuencia de mora por subcategoría del negocio en entrenamiento.** La línea punteada representa el promedio del conjunto de entrenamiento. Las diferencias son descriptivas y pueden reflejar otras características de los créditos o del período.

![Frecuencia de mora por subcategoría](credit-risk-frontier/figures/intermedia_20260714_redesign/04_subcategorias.png)

La frecuencia observada va desde 40,7 % en “venta de accesorios, bolsos y/o bisutería” hasta 79,7 % en “estéticas y spa”. Esta amplitud indica que la subcategoría contiene diferencias entre grupos. No demuestra que la actividad económica cause la mora ni que la relación se mantenga en meses posteriores. También puede estar asociada con ingresos, canal de adquisición, política de aprobación u otras variables no representadas en la figura.

Después se revisan los atributos del buró. Allí aparece un problema distinto: algunos valores negativos son códigos de ausencia y no cantidades, de modo que se separan como “sin información”. Para el “Porcentaje de obligaciones que alguna vez estuvo en mora” (`G051S`) se utilizan intervalos fijos. La “Mora ponderada en créditos financieros en el mes M = 01” (`WD81`) está concentrada en pocos valores y tiene una cola larga; entre los casos con información se forman cinco grupos del mismo tamaño.

**Figura 5. Asociación descriptiva del “Porcentaje de obligaciones que alguna vez estuvo en mora” (G051S) y la “Mora ponderada en créditos financieros en el mes M = 01” (WD81) con la mora observada en entrenamiento.** Las barras muestran la frecuencia de mora dentro de cada grupo y la línea punteada el promedio de entrenamiento. Los grupos no representan efectos causales.

![Asociaciones bivariadas del buró](credit-risk-frontier/figures/intermedia_20260714_redesign/05_bivariado_buro.png)

En ambos atributos aparecen diferencias ordenadas entre grupos. La frecuencia aumenta al pasar hacia valores mayores, aunque los códigos sin información no siguen necesariamente la misma secuencia. El gráfico permite comprender que los códigos especiales deben tratarse de manera explícita y que una única regla verbal no resume toda la relación.

Después se calculó, para cada variable numérica o binaria permitida, su correlación individual con `target=1` dentro del entrenamiento. Una correlación positiva indica que los valores mayores aparecen asociados con más mora; una negativa indica la dirección contraria. La medida describe una relación aislada y no reemplaza a un modelo que combina varias variables.

**Figura 6. Variables con mayor asociación individual con la mora en entrenamiento.** Se muestran las doce correlaciones de mayor magnitud entre las variables admitidas. Rojo indica asociación positiva y azul, negativa.

![Asociaciones individuales en entrenamiento](credit-risk-frontier/figures/intermedia_20260714_redesign/06_asociaciones_entrenamiento.png)

El “Porcentaje de obligaciones que alguna vez estuvo en mora” (`G051S`), la “Mora ponderada en créditos financieros en el mes M = 01” (`WD81`) y la “Mora ponderada en las obligaciones en el mes M = 06” (`WD03`) aparecen entre las asociaciones positivas de mayor magnitud. La cantidad de personas a cargo es la única declaración directa del formulario que ingresa en este grupo de doce variables. La dirección observada corresponde a la cartera aprobada y al período de entrenamiento; no debe trasladarse automáticamente a todas las personas que solicitan crédito.

Las variables monetarias requieren una lectura adicional porque provienen de declaraciones y contienen valores muy alejados del centro de la distribución. La escala logarítmica permite mostrar montos pequeños y grandes sin que unos oculten a los otros. La línea punteada utiliza el criterio descriptivo de cuartil superior más 1,5 veces el recorrido intercuartílico.

**Figura 7. Distribución de dos montos declarados y presencia de valores extremos en entrenamiento.** Las curvas se separan por desenlace solo para describir su superposición. Superar la línea punteada no implica que el dato sea erróneo.

![Valores extremos en montos declarados](credit-risk-frontier/figures/intermedia_20260714_redesign/07_valores_extremos.png)

Los ingresos mensuales del negocio y los gastos familiares presentan colas extensas. Hay 329 ingresos y 319 gastos por encima del criterio descriptivo. Esos casos pueden corresponder a negocios de otra escala, errores de carga o unidades interpretadas de manera distinta. Como los archivos disponibles no permiten decidir entre esas posibilidades, se conservan y se señalan como una limitación de calidad.

El EDA termina contando cuántos de los veinte atributos seleccionados del buró carecen de información. En las tablas y figuras se denomina “historial esparso” a los casos en los que seis o más muestran códigos de ausencia. Es la forma operativa de aproximar el historial escaso en este estudio, no una definición universal.

**Figura 8. Distribución de la cantidad de atributos del buró sin información.** La línea punteada marca el corte principal de seis atributos. En el conjunto hay 3.175 créditos con historial esparso y 1.026 con historial denso.

![Distribución del historial del buró](credit-risk-frontier/figures/intermedia_20260714_redesign/08_historial_buro.png)

Esta clasificación no equivale a falta de bancarización ni a primer crédito. Una persona puede tener seis atributos sin información y aun así registrar otros productos financieros. Del mismo modo, un primer crédito con la entidad puede coexistir con antecedentes en otras instituciones.

El conteo por persona se complementa con una revisión variable por variable. Dos columnas pueden pertenecer al mismo buró y tener coberturas muy distintas. En el formulario ocurre algo similar: algunos datos están casi completos, mientras que otros faltan para una parte considerable de las solicitudes.

**Figura 9. Proporción de valores faltantes en las veintinueve variables estructuradas compartidas por los tres modelos.** En TransUnion, los códigos negativos se cuentan como ausencia de información. En el formulario se consideran los valores vacíos. La figura utiliza los 4.201 casos del conjunto final.

![Faltantes en las variables estructuradas](credit-risk-frontier/figures/intermedia_20260714_redesign/09_faltantes_variables.png)

La cobertura es muy desigual. `RLE904` carece de información en el 99,1 % de los casos, `LMD34S` en el 94,1 % y `AGG2503` en el 84,6 %. Entre las declaraciones del formulario, el arriendo mensual falta en el 62,7 % y los ingresos diarios en el 36,7 %. En cambio, la edad y los ingresos mensuales del negocio tienen una cobertura superior al 97 %.

Estos faltantes no se interpretan como ceros. En la Regresión Logística se completan con medianas aprendidas en entrenamiento; XGBoost conserva la ausencia y puede asignarle una rama propia; Qwen la recibe mediante una expresión explícita. La comparación no supone que esas tres formas sean equivalentes, sino que documenta cómo cada método puede procesar el mismo problema de disponibilidad.

Por último, se cruza la definición de historial con la partición temporal. Este gráfico cumple una función distinta de la Figura 8: no cuenta faltantes, sino que muestra cuántos casos hay y cuál es su frecuencia de mora dentro de cada combinación.

**Figura 10. Frecuencia de mora por disponibilidad del historial y partición temporal.** Las barras comparan historial esparso y denso dentro de entrenamiento, validación y prueba. `target=1` mantiene en todos los casos el significado de mora mayor de 60 días dentro de 150 días.

![Mora por historial y partición](credit-risk-frontier/figures/intermedia_20260714_redesign/10_segmentos_por_particion.png)

En entrenamiento, la mora alcanza 74,3 % entre los 2.808 casos con historial esparso y 48,7 % entre los 552 casos con historial denso. En prueba, las frecuencias bajan a 35,3 % y 25,7 %, respectivamente. La dirección de la diferencia entre segmentos se mantiene, pero el descenso temporal aparece en ambos. Esto impide explicar la caída general de la prueba solamente por el cambio en la composición del historial.

La figura no demuestra que la falta de antecedentes cause mora. El segmento puede estar relacionado con otras características de los solicitantes, con el momento de originación y con la política de aprobación. Su función es mostrar que la disponibilidad de información y el tiempo son dos dimensiones simultáneas del problema.

### 3.7 Modelos clásicos

#### 3.7.1 Regresión Logística

La Regresión Logística se utiliza como referencia lineal. Estima el logaritmo de las chances de mora como una suma de las veintinueve variables y sus coeficientes: `log[p / (1 − p)] = β₀ + β₁x₁ + … + β₂₉x₂₉`. Un coeficiente positivo se asocia con mayores chances de mora y uno negativo con menores chances, manteniendo fijas las demás variables.

Los códigos negativos de TransUnion se tratan como faltantes. Las ausencias se completan con la mediana de entrenamiento y las variables se estandarizan con el promedio y el desvío estándar de ese mismo conjunto. Estas transformaciones se aplican luego, sin volver a estimarlas, a validación y prueba.

La intensidad de la regularización se elige entre `C=0,01`, `0,1`, `1` y `10` mediante el AUC de validación. La prueba no participa en esa decisión. Como las variables están estandarizadas, los coeficientes de resultados pueden compararse en una escala común; describen asociaciones dentro del modelo y no efectos causales.

#### 3.7.2 XGBoost

XGBoost combina árboles de decisión construidos de manera secuencial y puede representar relaciones no lineales e interacciones. Recibe las mismas veintinueve variables. Los faltantes no se completan con una mediana: el algoritmo puede aprender hacia qué rama dirigirlos en cada regla.

Se evalúan veinte configuraciones de hiperparámetros propuestas por Optuna (Akiba et al., 2019). Cada alternativa se mide en validación y el entrenamiento se detiene cuando agregar árboles deja de mejorar; la prueba permanece separada.

Para describir el modelo se utilizan valores SHAP (Lundberg y Lee, 2017). Cada valor distribuye entre las variables la diferencia entre una predicción y un valor de referencia. La figura de resultados resume el promedio de su valor absoluto: muestra cuánto movió cada variable las predicciones, pero omite la dirección. Por eso esa magnitud no debe compararse directamente con un coeficiente β, que es global y conserva el signo.

### 3.8 Modelo de lenguaje

#### 3.8.1 Qué es Qwen3-8B

Qwen3-8B es un modelo de pesos abiertos con aproximadamente 8,2 mil millones de parámetros (Qwen Team, 2025). En este estudio se ejecuta de manera local mediante Ollama, con el modo de razonamiento habilitado y sin ajustar sus parámetros con estos datos. Por lo tanto, no existe un entrenamiento supervisado de Qwen sobre los 3.360 créditos.

La copia utilizada ocupa aproximadamente 5,2 GB y emplea la cuantización `Q4_K_M`. La ventana de contexto se fija en 40.960 tokens, se reservan hasta 1.024 para la respuesta y la temperatura es cero. Estas decisiones favorecen una ejecución local reproducible, pero delimitan el alcance a esta versión y esta precisión numérica.

Los resultados conservan la probabilidad final, la validez del formato y los metadatos de la consulta, pero no el razonamiento intermedio ni la respuesta verbal completa. En consecuencia, no se reconstruyen explicaciones caso por caso a partir del puntaje. Qwen tampoco aprueba ni rechaza créditos: produce una estimación que aquí se evalúa de manera retrospectiva.

#### 3.8.2 De una fila tabular a una consulta

Qwen no recibe una planilla. Para que pueda leerla, cada nombre de columna se reemplaza por una descripción y se escribe junto con su valor. La serialización conserva el mismo orden en todos los casos: primero los veinte atributos de TransUnion y después las nueve variables directas del formulario. Los códigos negativos del buró se expresan como “sin historial”. Los faltantes del formulario también se presentan como ausencia, no como cero.

La entrada con datos estructurados contiene exactamente esas veintinueve variables. Las primeras veinte provienen de TransUnion y las nueve restantes son respuestas directas del formulario. No se agregan categorías de negocio, canal, alianza, género, objetivos del crédito, identificadores, fecha, partición ni segmento. La entrada con datos estructurados y no estructurados conserva todo ese bloque y añade al final únicamente `descripcion_negocio` cuando tiene contenido.

**Cuadro 6. Fragmento de una solicitud real anonimizada del conjunto de entrenamiento y su representación.** El identificador y el desenlace se omiten. Los valores negativos del buró que aparecen en la tabla fuente se traducen como ausencia en la consulta.

| Elemento | Valor real del registro | Forma presentada a Qwen |
|---|---:|---|
| `AGG308` | 0 | Monto en mora agregado ... al mes M = 08: 0,00 |
| `WD81` | 6,25 | Mora ponderada ... en el mes M = 01: 6,25 |
| `AGG2503` | -1 | Plazo inferido ... en el mes M = 03: sin historial |
| `appusers_age` | 43 | Edad de la persona solicitante: 43 |
| `shops_monthly_incomes` | 1.600.000 | Ingresos mensuales del negocio: 1.600.000 |
| `shops_monthly_outcomes` | 600.000 | Egresos mensuales del negocio: 600.000 |
| `shops_shop_age` | 4 | Antigüedad del negocio: 4 años |
| `descripcion_negocio` | `Comidas preparadas` | Descripción del negocio: Comidas preparadas |

La entrada sin descripción del registro del Cuadro 6 ocupa 2.199 caracteres. La entrada con descripción ocupa 2.244 y termina exactamente de la siguiente manera:

```text
la antigüedad del negocio en años: 4.00, descripción del negocio: Comidas preparadas
```

El Anexo 7.4 reproduce el mensaje completo regenerado desde ese mismo registro. La instrucción indica de manera explícita que `target=1` corresponde a una mora mayor de 60 días dentro de los primeros 150 días de observación. El identificador, la fecha, la partición y el desenlace real no forman parte de la consulta.

#### 3.8.3 Consulta sin ejemplos y con ocho ejemplos

La respuesta debe terminar con `PROBABILIDAD_DE_MORA:` y un número entero entre 0 y 100. Pedir una probabilidad permite ordenar casos sin fijar un punto de corte. Si la salida no respeta ese formato, se registra como inválida y no se reemplaza por un valor arbitrario.

Qwen se evalúa sin ejemplos y con ocho créditos resueltos antes del caso que debe puntuar. La cantidad de ocho se fijó como una decisión práctica: permite mostrar cuatro casos de cada desenlace y mantener un conjunto pequeño de referencias, sin hacer que los ejemplos ocupen la mayor parte de la consulta ni aumenten innecesariamente su costo. No surge de una optimización estadística y no se probaron distintas cantidades sobre el conjunto de prueba; por eso no se presenta como un número óptimo.

Los ocho ejemplos cambian para cada crédito evaluado. El conjunto candidato contiene únicamente los 3.360 créditos de entrenamiento. Para medir semejanza se utilizan las mismas veintinueve variables estructuradas: los códigos negativos de TransUnion se convierten en faltantes y cada valor observado se estandariza con el promedio y el desvío del entrenamiento. A los faltantes se les asigna cero en esa escala transformada, lo que corresponde al promedio de entrenamiento.

Luego se calcula la distancia euclídea entre el caso y cada candidato. Se recorren los vecinos desde el más cercano hasta reunir cuatro con `target=1` y cuatro con `target=0`, sin repetir identificadores. El desenlace se usa solamente para asegurar ese equilibrio; no participa en la distancia. El orden final se mezcla con una semilla fija de 42. Así se combinan relevancia local, presencia de ambas clases y separación estricta de validación y prueba.

Cada ejemplo muestra sus variables en una forma compacta y el resultado observado como `PROBABILIDAD_DE_MORA: 100` o `PROBABILIDAD_DE_MORA: 0`. El caso final conserva las denominaciones completas. Incluir estas referencias no modifica los parámetros de Qwen: desaparecen al terminar la consulta.

Las cuatro configuraciones combinan las dos modalidades —sin ejemplos y con ocho ejemplos— con los dos tipos de entrada —datos estructurados y datos estructurados más la descripción—. La cantidad, el procedimiento de selección y la consigna quedaron fijados antes de ejecutar la prueba.

### 3.9 Métricas y forma de estimar la incertidumbre

El AUC es la medida principal porque evalúa el ordenamiento sin fijar un punto de corte. Un AUC de 0,78 no significa que el modelo “acierte el 78 % de los casos”: significa que, al tomar un crédito moroso y uno normal al azar, existe aproximadamente un 78 % de probabilidad de que el primero reciba un puntaje mayor.

El aporte de la descripción se calcula dentro de Qwen, restando el AUC obtenido con los datos estructurados al AUC obtenido con los datos estructurados más la descripción. La resta utiliza solamente créditos con una predicción válida en ambas entradas. Los intervalos del 95 % se estiman con 2.000 remuestreos estratificados de la prueba; las predicciones con y sin descripción se mantienen emparejadas en cada remuestreo.

La calibración se estudia en los métodos clásicos mediante diagramas de fiabilidad por deciles, el puntaje de Brier, el error de calibración esperado (ECE) y la pendiente de calibración. Estas medidas comparan la magnitud de las probabilidades con la frecuencia observada; un modelo puede ordenar bien y, al mismo tiempo, asignar riesgos demasiado altos.

También se informa el AUC por disponibilidad del historial y la distribución de los puntajes de Qwen. En este último caso se cuentan las probabilidades distintas y la concentración en los extremos 0 y 1. Por último, la proporción de respuestas válidas permite separar la estabilidad del formato de la capacidad de ordenar el riesgo.

### 3.10 Reproducibilidad

El procesamiento quedó dividido en pasos que pueden repetirse en el mismo orden: reconstrucción del conjunto de créditos, preparación de los datos de entrada, entrenamiento de los modelos clásicos, inferencia local de Qwen y cálculo de métricas y figuras. Las inferencias de Qwen guardan avances porque pueden tardar varias horas. Cada archivo de avance posee una huella que identifica el conjunto de datos, las veintinueve variables, la definición de mora, el tipo de entrada, la cantidad de ejemplos y la consigna. Una huella distinta no puede incorporarse al mismo resultado.

Todas las predicciones llegan a una tabla común con el modelo, la información recibida, la modalidad, la partición, el segmento, la etiqueta observada y la probabilidad estimada. Kedro se utiliza solamente como mecanismo de organización, trazabilidad y reproducción. No decide qué variables usar ni reemplaza las decisiones metodológicas.

### 3.11 Privacidad y uso responsable

Los datos describen personas y pequeños negocios en una decisión financiera sensible. Por esa razón, los textos originales, los pagos crudos y los identificadores no forman parte de los materiales públicos. Las respuestas completas de Qwen tampoco se conservaron en las cachés del experimento; solo se guardaron sus probabilidades y metadatos. El ejemplo incluido fue revisado para evitar nombres, direcciones y datos de contacto, y su clave y desenlace se omiten.

La ejecución local de Qwen mantiene los datos dentro del entorno de trabajo. Este estudio no aprueba ni rechaza créditos individuales: evalúa modelos sobre hechos pasados. Tampoco analiza equidad entre grupos, por lo que una mejora de AUC no debe interpretarse como habilitación automática para operar. Antes de una aplicación serían necesarios controles específicos de trato desigual, explicabilidad, apelación y seguimiento temporal.

## 4. Resultados

Los resultados se concentran en el conjunto de prueba, la lectura de los métodos clásicos, la tabla comparativa y las figuras que permiten interpretar el ordenamiento y las probabilidades. El ΔAUC de la descripción se presenta al final.

### 4.1 Conjunto de prueba

El conjunto final contiene 4.201 créditos con desenlace observable a 150 días. El entrenamiento reúne 3.360 casos y una frecuencia de mora de 70,1 %; la validación contiene 420 y alcanza 79,0 %. Los 421 créditos más recientes se reservan para la prueba y presentan una frecuencia de mora de 29,2 %.

La prueba también contiene una proporción mayor de casos con historial denso que el entrenamiento. Todos los modelos se comparan sobre este mismo período, pero sus resultados corresponden a un escenario en el que cambiaron tanto la frecuencia de mora como la disponibilidad de información del buró. Los datos no permiten atribuir ese cambio a una causa única.

### 4.2 Coeficientes y contribuciones de los métodos clásicos

La Regresión Logística seleccionada en validación utiliza `C=10`. Como las veintinueve variables fueron estandarizadas, sus coeficientes pueden compararse en una escala común. Un valor positivo se asocia con mayores chances de mora y uno negativo con menores chances, manteniendo fijas las demás variables. Esta lectura no representa un efecto causal.

**Figura 11. Coeficientes de la Regresión Logística con veintinueve variables estructuradas.** Las ausencias se completan con la mediana y las variables se estandarizan utilizando solamente el entrenamiento. Rojo representa coeficientes positivos y azul, negativos.

![Coeficientes de la Regresión Logística](credit-risk-frontier/figures/intermedia_20260714_redesign/11_coeficientes_logistica.png)

La “Mora ponderada en créditos financieros en el mes M = 01” (`WD81`) presenta el coeficiente positivo de mayor magnitud, β=1,477. Le siguen el “Monto en mora agregado de obligaciones no hipotecarias en créditos financieros al mes M = 08” (`AGG308`), con β=0,804, y el “Porcentaje de obligaciones que alguna vez estuvo en mora” (`G051S`), con β=0,631. La edad presenta el coeficiente negativo de mayor magnitud, β=-0,204. Los mayores pesos positivos se concentran en antecedentes de atraso.

**Figura 12. Variables con mayor contribución en XGBoost sobre la prueba temporal.** Se muestran las quince mayores medias del valor SHAP absoluto. La medida resume contribuciones locales y se expresa en la escala del logaritmo de las chances.

![Contribuciones SHAP de XGBoost](credit-risk-frontier/figures/intermedia_20260714_redesign/12_shap_xgboost.png)

`G051S` y `WD81` presentan las mayores contribuciones medias absolutas, 0,638 y 0,627. Luego aparecen `DUEMAG01`, con 0,400, y `WD03`, con 0,371. Entre las variables del formulario, la edad, el arriendo mensual y el capital inicial del negocio también se ubican entre las quince primeras. El coeficiente β conserva un signo global; el valor SHAP depende del caso y su promedio absoluto omite la dirección. Por eso sus magnitudes no son equivalentes.

### 4.3 Resultados finales: AUC, calibración y segmentos

La comparación reúne dos métodos clásicos y cuatro configuraciones de Qwen. Las configuraciones con datos estructurados reciben las mismas veinte variables de TransUnion y nueve variables directas del formulario. Las otras dos agregan los datos no estructurados, representados únicamente por la descripción del negocio.

**Cuadro 7. Resultados finales en la prueba temporal.** El AUC y su intervalo del 95 % se calculan sobre las probabilidades válidas. Los métodos clásicos producen una predicción para los 421 créditos; en Qwen también se informa cuántas respuestas respetan el formato solicitado.

| Configuración | Información recibida | Casos válidos | AUC | Intervalo del 95 % |
|---|---|---:|---:|---:|
| Regresión Logística | TransUnion más 9 variables directas | 421 de 421 | 0,757 | [0,707; 0,804] |
| XGBoost | TransUnion más 9 variables directas | 421 de 421 | 0,761 | [0,710; 0,808] |
| Qwen3-8B sin ejemplos | TransUnion más 9 variables directas | 404 de 421 | 0,654 | [0,601; 0,706] |
| Qwen3-8B sin ejemplos | TransUnion, 9 variables directas y descripción | 409 de 421 | 0,632 | [0,578; 0,684] |
| Qwen3-8B con 8 ejemplos | TransUnion más 9 variables directas | 407 de 421 | 0,620 | [0,564; 0,673] |
| Qwen3-8B con 8 ejemplos | TransUnion, 9 variables directas y descripción | 414 de 421 | 0,560 | [0,506; 0,612] |

XGBoost obtiene el mayor AUC, 0,761, y la Regresión Logística queda muy cerca, con 0,757. La diferencia puntual es 0,003 y sus intervalos se superponen ampliamente. El mejor resultado de Qwen es 0,654 con los datos estructurados, sin ejemplos y sin descripción. Incluir ocho ejemplos no mejora el AUC con ninguno de los dos tipos de entrada.

**Figura 13. ROC-AUC e intervalos del 95 % de las seis configuraciones evaluadas.** La línea vertical en 0,50 representa un ordenamiento equivalente al azar. Los intervalos se obtienen mediante 2.000 remuestreos de la prueba.

![Comparación de AUC entre modelos y datos de entrada](credit-risk-frontier/figures/intermedia_20260714_redesign/15_auc_modelos.png)

La figura confirma la lectura del cuadro: los métodos clásicos presentan los valores más altos y resultados muy cercanos entre sí. Qwen queda por debajo en las cuatro configuraciones, aun cuando recibe las mismas variables estructuradas.

**Figura 14. Calibración de la Regresión Logística y XGBoost en la prueba temporal.** Los puntos agrupan los créditos en diez deciles de probabilidad. La diagonal representa coincidencia entre la probabilidad estimada y la frecuencia observada; se mantienen los mismos colores de la Figura 13.

![Calibración de los modelos clásicos](analisis_E4_review/figuras_analisis/fig_calibracion.png)

Ambos modelos se curvan por debajo de la diagonal en los deciles altos: asignan probabilidades mayores que la frecuencia de mora observada. La pendiente es 0,699 en la Regresión Logística y 0,652 en XGBoost, por debajo del valor ideal de 1. XGBoost presenta menor Brier (0,179 frente a 0,198) y menor ECE (0,055 frente a 0,148), aunque su pendiente también muestra probabilidades demasiado extremas. La sobrestimación es coherente con la caída de la mora entre entrenamiento (70,1 %) y prueba (29,2 %), pero la figura no permite atribuir ese cambio a una causa única.

**Figura 15. AUC por disponibilidad del historial del buró.** Los puntos muestran el AUC y las líneas su intervalo del 95 %. Se incluyen los dos métodos clásicos y la mejor configuración de Qwen: datos estructurados sin ejemplos.

![AUC por segmento de historial](analisis_E4_review/figuras_analisis/fig_auc_segmento.png)

La Regresión Logística baja de 0,776 con historial denso a 0,718 con historial esparso; XGBoost baja de 0,786 a 0,695. Qwen obtiene 0,641 y 0,666, respectivamente. Sus intervalos se superponen y el segmento esparso contiene cerca de 150 casos, por lo que no se observa una diferencia clara entre segmentos para Qwen. Los dos métodos clásicos, por su parte, obtienen AUC puntualmente menores en el segmento con historial esparso.

### 4.4 Puntajes de Qwen y cambio al agregar la descripción

**Figura 16. Distribución de las probabilidades asignadas por Qwen3-8B.** Los cuatro paneles corresponden a la combinación entre modalidad sin ejemplos o con ocho ejemplos y datos estructurados solos o acompañados por la descripción.

![Histograma de puntajes de Qwen](analisis_E4_review/figuras_analisis/fig_qwen_hist.png)

Sin ejemplos, Qwen reparte sus respuestas válidas entre 18 valores con datos estructurados y 20 cuando también recibe la descripción. Con ocho ejemplos, la distribución se concentra en los extremos. Con datos estructurados, 203 de 407 créditos reciben exactamente 0 y 136 reciben 1; al agregar la descripción, 235 de 414 reciben 0 y 75 reciben 1.

En estas configuraciones, con ocho ejemplos se observa un patrón casi binario y una menor cantidad de niveles disponibles para ordenar casos intermedios. Esta concentración es coherente con la caída del AUC, pero no demuestra por sí sola una relación causal ni una limitación general de todos los modelos de lenguaje.

El aporte de la descripción se estima finalmente sobre pares comparables, es decir, sobre créditos con una probabilidad válida tanto sin descripción como con descripción. Por eso estos AUC se recalculan sobre la intersección de respuestas y pueden diferir levemente de los valores generales del Cuadro 7.

Sin ejemplos se obtienen 393 pares y el cambio es ΔAUC=-0,021, con un intervalo del 95 % entre -0,063 y 0,018. El intervalo incluye cero. Con ocho ejemplos se obtienen 400 pares y el cambio es ΔAUC=-0,065, con un intervalo entre -0,122 y -0,005; en este caso todo el intervalo queda por debajo de cero.

**Figura 17. Diferencia pareada de AUC al agregar la descripción del negocio a Qwen.** El punto muestra el cambio de AUC y la línea horizontal su intervalo del 95 %. La línea vertical en cero representa ausencia de cambio.

![Cambio pareado de AUC al agregar la descripción](credit-risk-frontier/figures/intermedia_20260714_redesign/16_delta_descripcion_qwen.png)

En las dos modalidades, agregar la descripción no mejora el ordenamiento. Sin ejemplos, el resultado es compatible tanto con una disminución pequeña como con ausencia de cambio. Con ocho ejemplos, la disminución se mantiene en todo el intervalo estimado.

## 5. Pasos a seguir

Los resultados dejan tres tareas concretas para la siguiente etapa de la tesis:

1. Construir una representación semántica de los campos textuales utilizando solamente entrenamiento y validación, e incorporarla a la Regresión Logística y XGBoost antes de evaluarla en una nueva ventana temporal.
2. Repetir Qwen después de corregir la unidad informada para `TEL32S` y conservar, bajo reglas explícitas de privacidad, las salidas necesarias para estudiar sus explicaciones. También se deberá evaluar si una extracción de probabilidad menos discreta evita la concentración observada en 0 y 1.
3. Replicar la comparación temporal con más casos de historial esparso y definir una estrategia de recalibración sobre datos representativos del período de aplicación.

Antes de utilizar los puntajes en una decisión también será necesario definir costos, umbrales, controles de equidad y seguimiento temporal. En esta etapa los modelos ordenan casos históricos: no aprueban ni rechazan solicitudes individuales.

## 6. Conclusiones

Este trabajo formó un conjunto temporal de 4.201 créditos y mantuvo una definición uniforme: `target=1` representa mora mayor de 60 días dentro de los primeros 150 días. La Regresión Logística, XGBoost y Qwen3-8B recibieron la misma base de veinte atributos de TransUnion y nueve variables directas del formulario.

En la prueba temporal, XGBoost obtiene el mayor AUC, 0,761, y la Regresión Logística queda muy cerca, con 0,757. Qwen3-8B no iguala a los métodos clásicos; su mejor AUC es 0,654 con datos estructurados y sin ejemplos. Los métodos clásicos obtienen AUC puntualmente menores en el segmento con historial esparso y sus probabilidades sobrestiman el riesgo en los deciles altos, por lo que una aplicación requeriría recalibración.

Los ocho ejemplos no mejoran el AUC de Qwen. Con ocho ejemplos, gran parte de sus puntajes se concentra en 0 y 1. La descripción del negocio tampoco agrega capacidad de ordenamiento: sin ejemplos, el ΔAUC es -0,021 y su intervalo incluye cero; con ocho ejemplos, es -0,065 y el intervalo queda por debajo de cero.

Las conclusiones se limitan a créditos aprobados, una partición temporal con un cambio marcado en la frecuencia de mora y una configuración concreta de Qwen. Dentro de ese alcance, el resultado es directo: los métodos clásicos ordenan mejor el riesgo y la lectura del texto libre no produce una mejora en las condiciones evaluadas.

## 7. Anexos y trazabilidad

### 7.1 Decisiones metodológicas fijadas antes de la prueba

- La clase positiva representa mora mayor de 60 días dentro de 150 días.
- Los atrasos entre 31 y 60 días se excluyen.
- La separación 80/10/10 respeta el orden temporal.
- La Regresión Logística, XGBoost y Qwen reciben las mismas veintinueve variables estructuradas.
- El bloque común contiene veinte atributos de TransUnion y nueve variables directas del formulario.
- La entrada con datos no estructurados agrega únicamente `descripcion_negocio`.
- Los ocho ejemplos de Qwen provienen solo del entrenamiento y se buscan con las mismas veintinueve variables.
- Ningún modelo recibe identificadores, fecha, etiqueta, segmento, variables posteriores, puntajes internos ni relaciones económicas derivadas.

### 7.2 Variables excluidas

Se excluyen la antigüedad calculada desde el desembolso, las condiciones finales del crédito, los puntajes internos de la empresa, la fecha, la etiqueta, la partición, los identificadores y el segmento. También se excluyen categorías codificadas, canal, alianza, género, educación, objetivo del crédito y las relaciones derivadas `estimated_income`, `free_cash_flow`, `cost_ingress_ratio`, `debts_savings` y `relacion_edad_deuda`. La exclusión se aplica antes de entrenar los modelos y antes de buscar ejemplos semejantes.

Los campos `subcategoria_texto`, `otra_categoria_negocio` y `tipo_credito` se conservan en la tabla analítica para describir la fuente, pero no ingresan a los modelos de esta entrega. El texto libre evaluado es únicamente `descripcion_negocio`.

### 7.3 Recorrido de los resultados

La secuencia de procesamiento reconstruye la unión exacta, calcula los desenlaces a 150 días y genera el manifiesto temporal. Un subflujo entrena los dos modelos clásicos con las mismas veintinueve variables y guarda una predicción por crédito. También produce los coeficientes estandarizados de la Regresión Logística y el resumen SHAP de XGBoost. Otro proceso conserva las cuatro configuraciones locales de Qwen mediante avances recuperables y huellas verificables. La etapa de resultados admite solamente seis combinaciones autorizadas y rechaza cualquier combinación de modelo y datos ajena al estudio.

Kedro organiza la generación de los cuadros y las figuras a partir del conjunto de 4.201 casos y de la tabla común de predicciones. Las pruebas controlan el número y el orden de las variables, la ausencia de columnas prohibidas, la definición escrita del objetivo, el horizonte de 150 días, la selección de ejemplos y la cantidad de predicciones por modelo.

### 7.4 Consulta completa de un registro real anonimizado

Este anexo muestra la entrada completa correspondiente al registro del Cuadro 6. El caso pertenece al conjunto de entrenamiento. La clave anonimizada y el desenlace se omiten. La transcripción se regeneró con la misma función de serialización utilizada por las cuatro configuraciones de Qwen.

El mensaje de sistema fue:

```text
Estima el riesgo de mora de un microcrédito.
```

El mensaje de usuario fue:

```text
DATOS DEL SOLICITANTE: el monto en mora agregado de obligaciones no hipotecarias en créditos financieros al mes M=08: 0.00, la mora ponderada en créditos financieros en el mes M=01 (índice): 6.25, el plazo inferido agregado (relación de saldo sobre cuota mínima) en el mes M=03: sin historial, la magnitud de utilización de obligaciones retail en los últimos 24 meses (índice 0–600): sin historial, la magnitud total de todas las obligaciones en los últimos 24 meses (índice 0–600): sin historial, la magnitud del exceso de pago inferido agregado no hipotecario en 24 meses (índice 0–600): sin historial, los meses desde la más reciente apertura bancaria en cuotas: sin historial, la utilización de obligaciones bancarias sin garantía de mediano plazo (últimos 12 meses): sin historial, el número de obligaciones retail vigentes y al día con 24 meses o más de antigüedad: sin historial, el exceso de pago inferido en cuentas hipotecarias en los últimos 6 meses: sin historial, el saldo máximo en obligaciones de telecomunicaciones (últimos 12 meses) en pesos: sin historial, el saldo asignado a obligaciones identificadas como transactor al mes 9: sin historial, el porcentaje de obligaciones aperturadas en los últimos 24 meses sobre el total: 50.00, los meses desde la más reciente cuenta de ahorros aperturada: 20.00, el porcentaje de obligaciones vigentes y al día del total de obligaciones: 100.00, el número de obligaciones de telecomunicaciones vigentes y al día: sin historial, la utilización de obligaciones vigentes en créditos financieros (últimos 12 meses): 96.00, el porcentaje de obligaciones que alguna vez estuvo en mora: 20.00, el monto agregado en mora en el mes M=16: sin historial, la mora ponderada en las obligaciones en el mes M=06 (índice): 0.00, la edad del solicitante en años: 43.00, el número de dependientes del solicitante: 0.00, los gastos familiares mensuales en pesos: 100000.00, los ingresos mensuales del negocio en pesos: 1600000.00, los egresos mensuales del negocio en pesos: 600000.00, los ingresos diarios del negocio en pesos: 100000.00, el capital inicial del negocio en pesos: 0.00, el arriendo mensual del negocio en pesos: sin historial, la antigüedad del negocio en años: 4.00, descripción del negocio: Comidas preparadas

Analiza el caso y estima la probabilidad de que este crédito presente una mora mayor de 60 días dentro de los primeros 150 días de observación. Usa solamente los datos proporcionados. Razona brevemente y termina tu respuesta con una única línea con este formato EXACTO:
PROBABILIDAD_DE_MORA: <entero de 0 a 100>
```

## 8. Bibliografía

Akiba, T., Sano, S., Yanase, T., Ohta, T. y Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701

Basel Committee on Banking Supervision (BCBS). (2006). *International Convergence of Capital Measurement and Capital Standards*. Bank for International Settlements.

Bergmeir, C. y Benítez, J. M. (2012). On the Use of Cross-validation for Time Series Predictor Evaluation. *Information Sciences*, 191, 192–213. https://doi.org/10.1016/j.ins.2011.12.028

Chen, T. y Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Chioda, L., Gertler, P., Higgins, S. y Medina, P. (2024). FinTech Lending to Borrowers with No Credit History (NBER Working Paper No. 33208). National Bureau of Economic Research. https://doi.org/10.3386/w33208

Cornelli, G., Frost, J., Gambacorta, L. y Jagtiani, J. (2022). The Impact of Fintech Lending on Credit Access (BIS Working Papers No. 1041). Bank for International Settlements.

Demirgüç-Kunt, A., Klapper, L., Singer, D. y Ansar, S. (2025). *The Global Findex Database 2025: Connectivity and Financial Inclusion in the Digital Economy*. Banco Mundial. https://www.worldbank.org/en/publication/globalfindex/report

Feng, D., Dai, Y., Huang, J., Zhang, Y., Xie, Q., Han, W., Chen, Z., Lopez-Lira, A. y Wang, H. (2023). Empowering Many, Biasing a Few: Generalist Credit Scoring through Large Language Models. *arXiv:2310.00566*. https://doi.org/10.48550/arXiv.2310.00566

Financial Stability Board. (2017). *Financial Stability Implications from FinTech: Supervisory and Regulatory Issues that Merit Authorities’ Attention*. https://www.fsb.org/2017/06/financial-stability-implications-from-fintech/

Golec, M. y Alabduljalil, M. (2026). Interpretable LLMs for Credit Risk: A Systematic Review and Taxonomy. *Expert Systems with Applications*, 306, 130941. https://doi.org/10.1016/j.eswa.2025.130941

Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data? *Advances in Neural Information Processing Systems*, Datasets and Benchmarks Track.

Hand, D. J. y Henley, W. E. (1997). Statistical Classification Methods in Consumer Credit Scoring: A Review. *Journal of the Royal Statistical Society: Series A*, 160(3), 523–541.

Hegselmann, S., Buendia, A., Lang, H., Agrawal, M., Jiang, X. y Sontag, D. (2023). TabLLM: Few-shot Classification of Tabular Data with Large Language Models. *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics*, 5549–5581.

Lessmann, S., Baesens, B., Seow, H.-V. y Thomas, L. C. (2015). Benchmarking State-of-the-Art Classification Algorithms for Credit Scoring: An Update of Research. *European Journal of Operational Research*, 247(1), 124–136.

Lundberg, S. M. y Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.

Qwen Team. (2025). *Qwen3 Technical Report*. arXiv:2505.09388.

Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards* (2.ª ed.). Wiley.

Thomas, L. C., Edelman, D. B. y Crook, J. N. (2002). *Credit Scoring and Its Applications*. SIAM.

TransUnion. (2024). *Diccionario de Variables — Experto CreditVision* (V3.8). Documento técnico de atributos, uso bajo contrato.
