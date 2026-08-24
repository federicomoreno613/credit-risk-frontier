

> # **Comparación de predicción de riesgo crediticio entre  Modelos de Lenguaje razonadores y Machine Learning Clásico:**

> Un comparativo con Microfinanzas Latinoamericanas.

> # 

# 

Tesis presentada para optar al título de Magister en Explotación de Datos y Descubrimiento de Conocimiento

**Tesista:**		Lic. Federico Nicolás Moreno 

**Director**: 		Mgs. Boris Dorian Da Silva,

**Co/Director**:	Dr. Cristian Bravo

**Buenos Aires**, 29 de marzo de 2026

# **1\. Introducción y estado del arte**

## **1.1 El crédito y el problema de la decisión**

Se define como crédito al monto de dinero que una institución financiera presta a un consumidor bajo el compromiso de que sea repagado con intereses en cuotas a intervalos regulares (Thomas et al 2002). Prestar dinero implica clasificar bajo incertidumbre. Al entregar el capital, la institución asume el desafío de estimar qué tan probable es que el solicitante rompa su compromiso de pago (Hand & Henley, 1997); esa probabilidad es lo que la industria denomina riesgo crediticio.

Para cuantificar el riesgo crediticio, las instituciones construyen modelos predictivos que asignan a cada solicitud del préstamo un puntaje numérico que refleja la probabilidad estimada de incumplimiento. Con esta probabilidad se puede tomar una decisión de otorgar o rechazar la solicitud de crédito. Se distingue en este trabajo: “riesgo crediticio” para referirse al fenómeno que se mide, y “modelo de riesgo crediticio” para referirse al artefacto predictivo que lo estima. 

Esta definición simplifica el problema. En la práctica, la información con la que cuenta la institución es incompleta. Hand y Henley (1997) determinan que la probabilidad de que un aplicante incumpla debe ser estimada por la información que provee al momento de requerir el préstamo, es decir, el modelo opera sobre una representación incompleta de la realidad del deudor. 

Alrededor de la decisión central intervienen otras preguntas que exceden la medición del riesgo crediticio pero lo condicionan. La más inmediata es la veracidad de la información declarada, dentro de esta rama estan los trabajos que estudian el Fraude crediticio. Bolton y Hand (2002) señalan que la estadística y el *machine learning* proveen herramientas efectivas para la detección del fraude, un problema que en la práctica se puede resolver con motores de reglas o modelos complementarios. Otro factor importante es la capacidad de pago futura, Campbell y Cocco (2015) encontraron que la relación cuota/ingreso del hogar afecta directamente al riesgo, y Bellotti y Crook (2009) demostraron que incluir variables macroeconómicas como covariables temporales mejora la estimación de la probabilidad de incumplimiento.

Este trabajo se centra en la construcción de un modelo de riesgo crediticio: un modelo que, a partir de las variables disponibles al momento de la solicitud, clasifique el crédito como probable de pago o de incumplimiento.

## **1.2 Enfoque de evaluación crediticia.**

Se llama política crediticia, al conjunto de reglas, umbrales y procedimientos que determinan a quién se le otorga crédito, en qué monto y bajo qué condiciones (Thomas et al.,2002). Pueden existir instituciones que heredan estas políticas de una normativa determinada por una institución financiera superior, tal como, los Bancos Centrales. En instituciones no reguladas, muchas *fintechs* de microprestamos, estas prácticas pueden variar. 

Una de estas prácticas encontrada en la literatura son las tarjetas de puntuación (*scorecards*) que consisten en una puntaje fijo por cada característica del solicitante*,* Siddiqi (2017) las describe como el estándar operativo en instituciones reguladas.

Los burós de crédito son entidades que recopilan y centralizan el historial de endeudamiento y comportamiento de pago de personas y empresas a partir de información reportada por múltiples instituciones financieras (Thomas et al.,2002). Algunas de las empresas presentes en América Látina son las internacionales TransUnion y Equifax o locales como Veraz y Nosis en Argentina o DataCrédito en Colombia. Estas presuponen que el cliente tiene un historial crediticio previo.

La práctica que nos interesa es la de el desarrollo de modelos propietarios. Algunos pueden utilizar variables pertenecientes a buros de crédito y/o recolectadas por las mismas empresas. Como variable objetivo, se utiliza la ausencia de pago real de la empresa que lo desarrolla. (Thomas et al., 2002; Lessmann et al., 2015).

En las microfinanzas de mercados emergentes, donde el historial formal suele ser inexistente, los puntajes de buró pierden relevancia. En estos escenarios, desarrollar un modelo propio aparece cómo una necesidad. Según el Banco Mundial, aproximadamente un tercio de la población adulta global permanece sin acceso a servicios financieros formales (Demirgüç-Kunt et al., 2025), y en América Latina esta cifra es aún mayor entre trabajadores independientes, hogares de menores ingresos y población rural.

## **1.3 Definición de mora y construcción de la variable objetivo**

Un crédito se clasifica como default (incumplimiento) si el deudor excede un umbral de días de atraso en el pago de sus cuotas.La variable objetivo de un modelo de riesgo crediticio se construye a partir de la observación del comportamiento de pago de créditos ya otorgados.

La definición de ese umbral no es trivial ni universal. El Comité de Basilea sobre Supervisión Bancaria, en el marco regulatorio conocido como Basilea II, establece que un deudor se considera en default cuando cumple alguna de estas condiciones: (a) el banco considera improbable que el deudor pague la totalidad de sus obligaciones sin recurrir a la ejecución de garantías, o (b) el deudor tiene un atraso de más de 90 días en cualquier obligación crediticia material (BCBS, 2006, párr. 452).

En microfinanzas los compromisos de pago son más cortos (típicamente quincenales o mensuales) y los montos son menores respecto a la banca tradicional. En la práctica, umbrales de 60 y 90 días de atraso son los más frecuentes en *fintechs* y créditos al consumo (Cornelli et al., 2022; Chioda et al., 2025). Un umbral de 60 días de atraso suele ser equivalente a aproximadamente dos cuotas mensuales impagas.

Calcular, la variable objetivo, la mora, tiene su complejidad. Los días de atraso se miden desde la fecha de vencimiento de cada cuota, no desde la fecha de desembolso del crédito. Un problema común puede ser analizar la mora desde el desembolso (cosechas o *vintages*) pero dentro de la misma cosecha la fecha de la primera cuota puede variar, por cuestiones contractuales o comerciales, es decir, algunos créditos desembolsados a principio de mes vencen en el mismo mes y otros desembolsados a fin de mes vencen el siguiente. Como consecuencia, un umbral de 60 días de atraso puede representar entre 60 y 90 días calendario desde el desembolso, dependiendo de cuándo cayó la primera cuota. 

## **1.4 *Machine Learning* clásico para datos tabulares**

En datos tabulares, los métodos de gradient boosting dominan los comparativos de clasificación, tanto en riesgo crediticio (Lessmann et al., 2015\) como en dominios más amplios (Grinsztajn et al., 2022; Crook et al., 2007). El estado del arte práctico es XGBoost (Chen & Guestrin, 2016), y es contra este modelo que se define el comparativo.

**1.4.1 *Transformers* diseñados para datos tabulares**

Grinsztajn et al. (2022) determinan que la ventaja de los árboles de gradient boosting aumenta a medida que crece el volumen de datos. En datasets pequeños, por debajo de las 10.000 muestras, la ventaja se reduce. En esta dirección podemos encontrar una arquitectura de red neuronal denominada *transformer* (Vaswani et al., 2017). Un transformer es una red neuronal basada en un mecanismo de *atención* que permite ponderar la relevancia relativa de cada elemento de la entrada respecto a los demás. Aplicados a datos tabulares, estos modelos *tokenizan* cada variable como un *embedding*, esto es una representación numérica densa, y utilizan la atención para aprender interacciones entre variables sin necesidad de especificarlas manualmente, algo que los árboles de gradient boosting logran solo de forma aproximada mediante splits sucesivos.

Un trabajo relevante es el de TabPFN v2 (Hollmann et al., 2025), un transformer preentrenado sobre *datasets* tabulares sintéticos que realiza clasificación mediante in-context learning en un solo forward pass  sin necesidad de entrenamiento. Fue publicado en enero del 2025 en la revista Nature y superó a los *ensembles*  de los *baselines* más fuentes en datasets con menos de diez mil muestras y con ventajas en tiempos. 

Dado esto aparece cómo natural integrar a TabPFN como comparativo en el dataset que tenemos cómo caso de estudio.

**1.5 *Large Language Models* para datos tabulares**

Los modelos clásicos de gradient boosting no operan en el significado de las variables, cuando el modelo recibe que el rubro de un solicitante es "Peluquería y Manicuría" y su nivel educativo es "técnico", XGBoost aprende la asociación estadística entre esas etiquetas y el resultado de pago  pero no comprende qué es una peluquería ni qué implica un nivel técnico de educación.

Los modelos de lenguaje de gran escala (*Large Language Models*, LLMs) ofrecen una alternativa a esta limitación. Un LLM es una red neuronal con miles de millones de parámetros que se construye en dos etapas. La primera, denominada preentrenamiento, consiste en exponer al modelo a grandes volúmenes de texto  libros, artículos, páginas web, código, en múltiples idiomas  entrenado para predecir la siguiente palabra en una secuencia. Como resultado, el modelo adquiere una representación interna del significado de las palabras y sus relaciones, puede encontrar que una peluquería es un comercio minorista, que un nivel educativo técnico implica una formación profesional específica, o que un ingreso en pesos colombianos tiene un orden de magnitud distinto a uno en dólares. La segunda etapa, denominada postentrenamiento, ajusta el modelo preentrenado para que siga instrucciones, responda preguntas y razone sobre problemas concretos. Esta etapa típicamente incluye *fine-tuning* supervisado sobre pares de instrucción-respuesta y alineamiento mediante retroalimentación humana (Ouyang et al., 2022), y es lo que transforma un modelo que simplemente predice texto en uno capaz de resolver tareas específicas cuando se le formula una pregunta.

Hegselmann et al. (2023) serializan datos tabulares como texto y los pasan a un LLM, el modelo logra clasificaciones no triviales incluso en *zero-shot* (sin datos de entrenamiento), explotando el conocimiento adquirido durante el preentrenamiento. En el régimen de pocas muestras, esta técnica resultó competitiva con XGBoost, aunque con muchos datos de entrenamiento los árboles *boosteados* recuperan su ventaja.

Dado un conjunto de datos de cinco mil filas donde las variables incluyen rubros de negocio y niveles educativos en español, ¿puede un LLM que comprende el significado de esos valores aportar capacidad predictiva que XGBoost no captura?

Los reguladores demandan cada vez más transparencia y auditabilidad en los modelos de riesgo crediticio (Bücker et al., 2022), y la regulación europea clasifica los sistemas de IA para evaluación crediticia como de alto riesgo (Parlamento Europeo y Consejo de la Unión Europea, 2024). La generación de explicaciones en lenguaje natural por parte de los LLMs responde este requerimiento.

Pero esta ventaja semántica tiene un límite: el conocimiento del mundo que el LLM aplica al interpretar categorías depende de los datos sobre los cuales fue preentrenado. Sus inferencias sobre el perfil económico asociado a una ocupación o un nivel educativo reflejan lo que era cierto al momento del preentrenamiento, no necesariamente lo que es cierto ahora. En el contexto de *zero-shot*, donde el modelo opera exclusivamente desde su conocimiento preentrenado, este sesgo podría introducir distorsiones si la población crediticia tiene características que divergen de lo que el modelo "aprendió" como típico. Las estrategias de *few-shot* y *fine-tuning* mitigan este riesgo porque exponen al modelo a los resultados reales de pago observados en la cartera, lo que le permite ajustar sus priors al contexto específico. La evaluación empírica  que compara directamente *zero-shot* contra *few-shot* y *fine-tuning*  permite medir exactamente cuánto de la predicción proviene del conocimiento preentrenado y cuánto de los datos observados.

Existen evaluaciones previas de LLMs como modelos de riesgo crediticio  GPT-4, ChatGPT, Llama 1/2 y Bloomz en Feng et al. (2023); un LLM genérico en Chen (2025)  pero estos pertenecen a una generación anterior con capacidades muy inferiores a las de los modelos actuales, o son modelos propietarios cuyo costo los hace impracticables para una *fintech* de micropréstamos.

Los modelos evaluados por Feng et al. fueron publicados entre 2022 y 2023\. En menos de tres años, la capacidad de los LLMs *open-source* avanzó al punto de superar a modelos propietarios que eran hasta entonces el estado del arte. Esta generación mejoró en razonamiento, eficiencia de memoria y soporte multilingüe, e incorporó un modo de razonamiento explícito ausente en la generación previa.

En la revisión sistemática más reciente sobre LLMs en riesgo crediticio (Golec & AlabdulJalil, 2025), que analiza 60 trabajos publicados entre 2020 y 2025 con metodología PRISMA, no registra evaluaciones de modelos con *thinking mode* nativo en clasificación tabular financiera. El *survey* de Fang et al. (2024), que cubre el espectro completo de LLMs en datos tabulares, tampoco reporta esta combinación.

La generación de LLMs que evaluaron Feng et al. respondía de forma directa: recibía una pregunta y emitía una respuesta. Lo que cambió en los modelos actuales no es solo la escala sino la forma en que procesan antes de responder. En septiembre de 2024, OpenAI lanzó o1 con la capacidad de razonar explícitamente durante la inferencia, aunque sin revelar cómo lo había logrado. La explicación técnica llegó meses después, en enero de 2025, cuando DeepSeek publicó DeepSeek-R1 (DeepSeek-AI, 2025\) y mostró que ese comportamiento podía inducirse con aprendizaje por refuerzo aplicado directamente al modelo base, sin necesidad de que humanos anotaran ejemplos de razonamiento. El mecanismo consiste en recompensar al modelo únicamente por la corrección de su respuesta final, y como efecto emergente aparecen patrones de autorreflexión, verificación de pasos y adaptación de estrategia. 

En el benchmark AIME 2024, que reúne problemas de matemática competitiva, DeepSeek-R1 pasó de 15,6% a 71,0% de precisión solo con este entrenamiento, y llegó a 86,7% con votación por mayoría, igualando a o1.

Esta capacidad se denomina thinking mode. Antes de emitir su respuesta, el modelo produce una cadena de razonamiento interna donde descompone el problema, evalúa la evidencia y llega a una conclusión. La diferencia con respecto a un modelo estándar no es menor en el contexto del riesgo crediticio. Un LLM convencional recibe los datos de un solicitante y emite directamente una clasificación. Uno con thinking mode razona primero: considera que el solicitante tiene un negocio de peluquería con 36 meses de antigüedad lo que puede sugerir cierta estabilidad, pero que su puntaje de buró es bajo y sus deudas activas son elevadas, y recién entonces clasifica. Esto importa porque en crédito las variables no son independientes. Un ingreso alto puede ser irrelevante si el historial de atrasos es recurrente. Una deuda elevada no pesa igual en un emprendimiento que recién arranca que en uno consolidado. El thinking mode permite al modelo ponderar estas interacciones de forma explícita, algo que un árbol de decisión aproxima con splits sucesivos pero que un LLM puede articular en lenguaje natural.

Para esta evaluación se propone utilizar Qwen3.5-9B (Qwen Team, 2026), un modelo de 9 mil millones de parámetros lanzado en marzo de 2026 bajo licencia Apache 2.0.. Qwen3.5-9B pertenece a una familia construida sobre una arquitectura híbrida de *Gated Delta Networks* y atención estándar, y a pesar de su tamaño supera en benchmarks independientes a GPT-OSS-120B de OpenAI, un modelo trece veces más grande. Soporta 201 idiomas incluyendo español, lo que contrasta con los 8 idiomas de Llama 3.1, uno de los modelos evaluados por Feng et al. en la generación anterior. Integra thinking mode de forma nativa, activado por defecto pero desactivable mediante configuración, lo que permite comparar dentro del mismo modelo si el razonamiento explícito aporta capacidad predictiva o no —una comparación que sería imposible entre modelos distintos. Por último, corre en hardware de consumo con cuantización de 4 bits, con un requerimiento de memoria de aproximadamente 5 GB para inferencia, lo que lo hace viable para una fintech con recursos limitados. No se encontraron trabajos previos que evalúen un modelo con estas características en clasificación tabular financiera (Golec & AlabdulJalil, 2025; Fang et al., 2024).

## **1.5.1 LLM cloud como techo de calidad**

Proponemos además utilizar GPT-5.4 de OpenAI evaluado vía API en modo zero-shot y few-shot, sin fine-tuning. Este modelo cumple la función de tener una comparación importante sin limitaciones de recursos. Siendo esta versión de Openai el modelo de frontera las conclusiones serán bastante contundentes si no supera al Xgboost, además es una comparación directa a los resultados de Feng et al. (2023) que evaluaron GPT-4 sobre Lendings.

## **1.6 El gap: LLMs en riesgo crediticio de mercados emergentes**

La evidencia disponible establece que *XGBoost* domina la clasificación tabular cuando hay datos suficientes (Lessmann et al., 2015; Grinsztajn et al., 2022; Fang et al., 2024). La metodología elegida para utilizar LLMs con datos tabulares sera el framework aportado por Hegselmann et al. (2023) nombrado TabLLM. Consiste en serializar cada fila de una tabla como una oración en lenguaje natural  por ejemplo,

| Edad | Educación | Salario |
| :---: | :---: | :---: |
| 42 | Master | 594 |

Sera traducida a:

"la persona tiene 42 años, su educación es *Master*, su ganancia es 594 dólares"  

Esto es lo que se utilizara de entrada al modelo de lenguaje junto a una instrucción sobre cómo entender los datos y por lo tanto generar la clasificación. El modelo entonces predice la variable objetivo no sólo utilizando los datos presentes en esta serialización sino también en el conocimiento obtenido durante el preentrenamiento. En el trabajo de Hegeselmann et al. (2023) se evaluó esta técnica sobre nueve conjuntos de datos públicos y encontraron que en el régimen de *few-shot* (pocas muestras etiquetadas), los LLMs resultaron competitivos o superiores a XGBoost. Con muchos datos de entrenamiento, XGBoost recupera su ventaja  pero la brecha se cierra a medida que el *conjunto de datos* se reduce. Este resultado constituye la motivación central del presente trabajo.

Los conjuntos de datos evaluados por TabLLM eran anglosajones y de dominios ajenos al riesgo crediticio. Las condiciones del caso de estudio son diferentes. El conjunto de datos de esta tesis tiene 5.351 créditos de una *fintech* latinoamericana de micropréstamos  un orden de magnitud por debajo de LendingClub y dentro del rango donde TabLLM reporta ventajas para los LLMs. Los datos incluyen variables semánticas en español (rubro del negocio, nivel educativo, tipo de vivienda) cuyos valores fueron ingresados por los solicitantes en un formulario de texto, lo que genera una multiplicidad de variantes no estandarizadas. XGBoost trata estos valores como categorías opacas, sin capturar la similitud entre ellos  mientras que un LLM preentrenado en español puede inferir que "Peluquería y Manicuría" y "Salón de Belleza" son actividades equivalentes, o que "nivel educativo: técnico" sugiere un perfil ocupacional específico. Finalmente, la población objetivo de micro emprendedores no bancarizados de un mercado emergente  tiene cobertura limitada en los burós de crédito, lo que reduce la señal disponible para los modelos clásicos y puede aumentar el valor del conocimiento previo que aporta el LLM.

# **2\. Propuesta de trabajo, recursos y objetivos**

## **2.1 Pregunta de investigación**

¿Puede un LLM *open-source* compacto (\~9B parámetros), con *thinking mode* nativo y capacidad multilingüe, alcanzar o superar la performance de métodos clásicos de machine learning como modelo de riesgo crediticio con datos tabulares  y en qué medida el *chain-of-thought*, los ejemplos *in-context* y el *fine-tuning* cierran la brecha?

Para responder se compararán ocho configuraciones partiendo desde una regresión logística como baseline hasta el LLM con QLoRA fine-tuning sobre los mismos datos, con split temporal, evaluando capacidad discriminativa (ROC-AUC), calibración de probabilidades, interpretabilidad y costo computacional.

La evaluación se realiza sobre dos datasets: uno público y anglosajón (*LendingClub*) y otro propietario de una fintech latinoamericana de micropréstamos, para distinguir los efectos del modelo de los efectos del contexto.

## **2.2 Diseño experimental**

Se propone un comparativo controlado de ocho modelos, organizados en una progresión que va desde el *baseline* hasta el *fine-tuning* del modelo:

| \# | Modelo | Tipo | Entrenamiento | Justificación |
| ----- | ----- | ----- | ----- | ----- |
| 1 | Regresión logística | Baseline clásico | Full training | Estándar de la industria (Hand & Henley, 1997\) |
| 2 | XGBoost | ML clásico | Optuna, 500 trials | SOTA en datos tabulares (Lessmann et al., 2015\) |
| 3 | TabPFN v2 | Transformer tabular | Sin entrenamiento (forward pass) | SOTA en ≤10K muestras (Hollmann et al., 2025\) |
| 4 | GPT-5.4 zero-shot | LLM propietario | Sin entrenamiento | Techo de calidad; comparación con Feng et al. (2023) |
| 5 | Qwen3.5-9B *zero-shot* | LLM *open-source* | Sin entrenamiento | Conocimiento del mundo sin datos de training |
| 6 | Qwen3.5-9B \+ *thinking* | LLM razonando | Sin entrenamiento | *Thinking mode* nativo (Qwen Team, 2026\) |
| 7 | Qwen3.5-9B *few-shot* | LLM con ejemplos | 8–32 ejemplos *in-context* | *In-context learning* (Hegselmann et al., 2023\) |
| 8 | Qwen3.5-9B *fine-tuned* | LLM adaptado | QLoRA sobre train set | *Fine-tuning* eficiente (Hu et al., 2022; Dettmers et al., 2023\) |

**Tabla 1: Diseño experimental  8 modelos a comparar.**

Este diseño experrimental permite hacer preguntas entre cada transición: 

1. ¿XGBoost mejora significativamente sobre la regresión logística en este dataset (2 vs. 1)?

2. ¿Un transformer tabular nativo ya supera a XGBoost en un dataset de \~5K muestras (3 vs. 2)?

3. ¿Un LLM propietario frontier sin fine-tuning supera al transformer tabular (4 vs. 3)?

4. ¿Qwen open-source compite con GPT propietario (5 vs. 4)?

5. ¿Cuánto aporta el conocimiento preentrenado del LLM sin datos del dominio (5 vs. 1)?

6. ¿El thinking mode mejora la clasificación (6 vs. 5)?

7. ¿Los ejemplos in-context cierran la brecha con XGBoost (7 vs. 2)?

8. ¿El fine-tuning la cierra del todo (8 vs. 2)?

## 

## **2.2.1 Elección del modelo  ¿Por qué Qwen3.5-9B?**

La justificación detallada está en la sección 1.5. En resumen, es el mejor modelo open-source en su rango de parámetros, tiene thinking mode nativo, soporta español, y corre en hardware de consumo.

## **2.2.2 Fine-tuning con QLoRA**

El fine-tuning se realizará con QLoRA (Dettmers et al., 2023): cuantización 4-bit del modelo base con adaptadores de bajo rango (rank 16–64) entrenados sobre el conjunto de training. Se utilizará la librería *Unsloth* para eficiencia en *Apple Silicon*. 

El formato de entrenamiento será instruction tuning: cada ejemplo consiste en un prompt con la serialización tabular y la instrucción de clasificación, y una respuesta con la clase predicha y opcionalmente una explicación breve.

## 

## **2.3 Datasets**

Se usan dos datasets que permiten separar el efecto del modelo del efecto del contexto:

**Dataset 1  LendingClub** (comparativo público). Dataset público de préstamos personales en Estados Unidos, ampliamente utilizado como comparativo en riesgo crediticio con LLMs (Feng et al., 2023; AlMarri et al., 2025). Contiene más de 2 millones de registros de créditos otorgados entre 2007 y 2018, con 151 variables cuantitativas y cualitativas. La distribución del target presenta un desbalance natural: aproximadamente el 80% de los créditos fueron pagados en su totalidad y el 20% restante resultó en default, lo que refleja una tasa de incumplimiento propia de una plataforma de préstamos personales en un mercado bancarizado. Se utilizará el mismo subset y preprocesamiento que Feng et al. (2023) para permitir comparación directa con los resultados de GPT-4, Llama y Bloomz. Este dataset cumple el rol de control: si Qwen3.5-9B supera a los LLMs de 2023 sobre LendingClub, la mejora es atribuible al modelo; si la mejora es mayor sobre el dataset latinoamericano que sobre LendingClub, la diferencia es atribuible al contexto.

**Dataset 2**  Fintech latinoamericana (caso de estudio). El segundo dataset proviene de una plataforma latinoamericana de micropréstamos para microempresarios no bancarizados. Contiene 5.351 créditos con 50 columnas seleccionadas de tres fuentes:

| Fuente | Features | Cobertura | Rol |
| :---: | :---: | :---: | :---: |
| TransUnion (buró de crédito) | 20 variables | 95,8% | Principal predictor numérico |
| Formulario de la app | 18 variables | 100% | Variables semánticas (rubro, educación, vivienda) |
| Metamap KYC | 12 variables | 97,9% | Verificación de identidad |

**Distribución del target.** La variable *target* se define como: *default* si mora \> 60 días; pago si mora ≤ 30 días. Los créditos con mora entre 30 y 60 días (zona gris) se excluyen para evitar clasificaciones ambiguas. La distribución resultante es 2.745 *defaults* (51%) y 2.606 pagadores (49%).Este balance cercano al 50/50 no fue buscado deliberadamente sino que es consecuencia natural de la definición del target y de las características de la cartera: la exclusión de la zona gris elimina casos intermedios, y la fintech opera en un segmento de alto riesgo (microempresarios no bancarizados) donde la tasa de mora es mucho más alta que en mercados bancarizados como LendingClub, cuya relación es aproximadamente 80/20. El balance resultante tiene dos implicancias para el diseño experimental. Por un lado, no requiere técnicas de re-balanceo artificial (*oversampling*, *undersampling*, SMOTE), lo cual simplifica la comparación entre modelos. Por otro, las probabilidades emitidas por los modelos estarán calibradas para esta distribución, no para la distribución real de la cartera en producción lo que lleva a la necesidad de la calibración incluida en la sección 2.4.

Los datos se dividen temporalmente en *train* (\~70%), validación (\~15%) y *test* (\~15%) para evitar *data leakage*.

Como ya mencionamos, este dataset es el caso de estudio principal: contiene las condiciones que la literatura identifica como favorables para LLMs dataset pequeño (Hegselmann et al., 2023), variables semánticas no estandarizadas (Cerda & Varoquaux, 2020), idioma no inglés, y buró de mercado emergente con historial financiero fragmentado.

## **2.4 Métricas de evaluación**

Los seis modelos se evalúan sobre cinco dimensiones:

Capacidad discriminativa. ROC-AUC como métrica primaria, con intervalos de confianza al 95% por *bootstrap* (1.000 iteraciones). Métricas complementarias: F1, precisión, recall.

**Calibración**. *Brier score, Expected Calibration Error (ECE)* y curvas de calibración. Los modelos de boosting producen probabilidades mal calibradas (Niculescu-Mizil & Caruana, 2005); se aplicará calibración isotónica post-hoc (Zadrozny & Elkan, 2002\) a todos los modelos.  No se encontraron demasiadas fuentes sobre la calibración de probabilidades a partir del output de un LLM  las probabilidades se derivan de logits sobre tokens de texto, lo que plantea un problema de calibración distinto al de los modelos tabulares que queda por explorar en el diseño experimental..

**Interpretabilidad**. *SHAP* values para XGBoost y regresión logística. Para el LLM, se comparan las explicaciones generadas en lenguaje natural contra las atribuciones empíricas de *SHAP*, siguiendo la metodología de AlMarri et al. (2025). Esta comparación permite evaluar si las explicaciones del LLM son fieles  es decir, si reflejan los factores que efectivamente determinan la predicción  o si son plausibles pero no fieles.

**Costo computacional**. Se evaluará el tiempo de inferencia por muestra, consumo de memoria, y costo total de entrenamiento/*fine-tuning.* En el caso del GPT-5.4 se reportará también el costo en dólares del uso de la API.

## **2.5 Objetivos y aportes esperados**

Objetivo general. Determinar si un L*LM open-source* compacto con thinking mode nativo puede competir con métodos clásicos de ML como modelo de riesgo crediticio en microfinanzas de mercados emergentes, evaluando no solo capacidad discriminativa sino también calibración e interpretabilidad.

**Objetivos específicos:**

(1) Comparar Qwen3.5-9B en modo zero-shot contra XGBoost y regresión logística sobre datos reales de microfinanzas latinoamericanas en español, para cuantificar el aporte del conocimiento preentrenado a datos del dominio.

(2) Medir el efecto del thinking mode nativo sobre la capacidad discriminativa, contrastando razonamiento explícito versus respuesta directa dentro del mismo modelo y contra los baselines clásicos.

(3) Evaluar si los ejemplos in-context (8, 16 y 32 muestras) cierran la brecha de performance entre el LLM y XGBoost sobre ambos datasets.

(4) Determinar si el fine-tuning con QLoRA permite a Qwen3.5-9B igualar o superar a los modelos clásicos entrenados sobre los mismos datos.

(5) Analizar si la comprensión semántica del LLM aporta capacidad predictiva sobre variables textuales (rubro, educación, vivienda) que XGBoost trata como categorías opacas.

(6) Evaluar si un transformer tabular nativo (TabPFN v2), diseñado específicamente para datasets pequeños, supera a XGBoost y a los LLMs en el régimen de \~5K muestras, y establecer el rol diagnóstico de la arquitectura transformer frente al conocimiento semántico.

(7) Comparar el rendimiento de un LLM propietario frontier (GPT-4o) contra el LLM open-source (Qwen3.5-9B) para determinar si la escala del modelo cierra la brecha con los métodos clásicos y cuánto avanzó la frontera respecto a los resultados de Feng et al. (2023).

Aportes esperados.

Si XGBoost mantiene su ventaja, que es lo que la literatura señala, el trabajo servirá para cuantificar cuánto gana y pierde cada enfoque, y eso es útil para otras fintechs evaluando la misma decisión. Si el LLM aporta algo en las variables semánticas o en el régimen de pocos datos eso ya es un aporte novedoso en la literatura.

## **2.6 Transferencia de resultados**

Las fintechs de micropréstamos obtendrán una evaluación empírica de si un LLM open-source compite con XGBoost a un costo computacional factible y en lo académico, el trabajo aporta el primer comparativo de LLMs con thinking mode contra ML clásico en microfinanzas latinoamericanas. También es importante para la explicabilidad que pueda aportar una LLM al riesgo crediticio. El EU AI Act clasifica la evaluación crediticia con IA como alto riesgo (Parlamento Europeo y Consejo de la Unión Europea, 2024).

## **2.7 Recursos**

* Lenguaje: Python 3.10+  
* Librerías ML: XGBoost, Optuna (Akiba et al., 2019), scikit-learn, pandas, numpy, matplotlib  
* Librerías LLM: transformers, unsloth, peft, bitsandbytes, mlx-lm  
* Modelo LLM: Qwen3.5-9B (Qwen Team, 2026), licencia Apache 2.0  
* Hardware: Mac M4 48 GB (inferencia MLX y QLoRA fine-tuning local)  
* Datos: dataset propietario de fintech latinoamericana (anonimizado y compartido en el repositorio) \+ LendingClub (público)  
* Repositorio público: [https://github.com/federicomoreno613/credit-risk-frontier/](https://github.com/federicomoreno613/credit-risk-frontier/)

## **2.8 Plan de trabajo y cronograma tentativo**

**Actividades específicas:**

1. Construcción del target y limpieza del dataset latinoamericano: procesamiento de pagos, umbral 60 días, exclusión zona gris, reducción de columnas, split temporal.  
2. Preparación del dataset LendingClub: replicar preprocesamiento de Feng et al. (2023) para comparabilidad.  
3. Entrenamiento de baselines clásicos: regresión logística y XGBoost con Optuna (500 trials).  
4. Serialización tabular: implementación del *Text Template* y *Table Format*, diseño de prompts para clasificación crediticia en español.  
5. Evaluación *zero-shot* y *thinking mode*: inferencia con Qwen3.5-9B sin entrenamiento, con y sin thinking, sobre ambos datasets.  
6. Evaluación *few-shot: in-context learning* con 8, 16 y 32 ejemplos, sobre ambos datasets.  
7. *Fine-tuning QLoRA*: entrenamiento del LLM sobre train set del dataset latinoamericano, selección de hiperparámetros con validación.  
8. Evaluación comparativa completa: métricas de discriminación, calibración, interpretabilidad y costo computacional para los 6 modelos × 2 datasets.  
9. Análisis de interpretabilidad: SHAP values para XGBoost vs. explicaciones del LLM, auditoría de fidelidad.  
10. Redacción de la tesis.

**Cronograma tentativo:**

| Actividad | Mes 1 | Mes 2 | Mes 3 | Mes 4 | Mes 5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1\. Target y limpieza dataset latinoamericano | X |  |  |  |  |
| 2\. Preparación LendingClub | X |  |  |  |  |
| 3\. *Baselines* clásicos | X |  |  |  |  |
| 4\. Serialización tabular | X | X |  |  |  |
| 5*. Zero-shot \+ thinking mode* |  | X |  |  |  |
| 6\. *Few-shot* |  | X |  |  |  |
| 7\. *QLoRA fine-tuning* |  | X | X |  |  |
| 8\. Evaluación comparativa |  |  | X | X |  |
| 9\. Análisis interpretabilidad |  |  |  | X |  |
| 10\. Redacción |  |  | X | X | X |

**3\. Referencias bibliográficas**

Akiba, T., Sano, S., Yanase, T., Ohta, T. y Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2623–2631.

AlMarri, S., Juhasz, K., Ravaut, M., Marti, G., Al Ahbabi, H. y Elfadel, I. (2025). Interpreting LLMs as credit risk classifiers: Do their feature explanations align with classical ML? CIKM 2025 FinAI Workshop. https://arxiv.org/abs/2510.25701

Basel Committee on Banking Supervision \[BCBS\]. (2006). International Convergence of Capital Measurement and Capital Standards: A Revised Framework. Bank for International Settlements. https://www.bis.org/publ/bcbs128.pdf

Bellotti, T. y Crook, J. (2009). Support vector machines for credit scoring and discovery of significant features. Expert Systems with Applications, 36(2), 3302–3308. https://doi.org/10.1016/j.eswa.2008.01.005

Bolton, R. J. y Hand, D. J. (2002). Statistical fraud detection: A review. Statistical Science, 17(3), 235–249. https://doi.org/10.1214/ss/1042727940

Bücker, M., Szepannek, G., Gosiewska, A. y Biecek, P. (2022). Transparency, auditability, and explainability of machine learning models in credit scoring. Journal of the Operational Research Society, 73(1), 70–90.

Campbell, J. Y. y Cocco, J. F. (2015). A model of mortgage default. The Journal of Finance, 70(4), 1495–1554. https://doi.org/10.1111/jofi.12252

Cerda, P. y Varoquaux, G. (2020). Encoding high-cardinality string categorical variables. IEEE Transactions on Knowledge and Data Engineering, 34(3), 1164–1176.

Chen, Q. (2025). Explore the use of prompt-based LLM for credit risk classification. Journal of Computer and Communications, 13(6), 33–46. https://doi.org/10.4236/jcc.2025.136003

Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

Chioda, L., Gertler, P., Higgins, S. y Medina, P. C. (2024). FinTech lending to borrowers with no credit history (NBER Working Paper No. 33208). National Bureau of Economic Research. https://doi.org/10.3386/w33208

Cornelli, G., Frost, J., Gambacorta, L. y Jagtiani, J. (2022). The impact of fintech lending on credit access for U.S. small businesses (BIS Working Papers No. 1041). Bank for International Settlements. https://www.bis.org/publ/work1041.pdf

Crook, J. N., Edelman, D. B. y Thomas, L. C. (2007). Recent developments in consumer credit risk assessment. European Journal of Operational Research, 183(3), 1447–1465.

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., … Liang, W. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Nature, 645, 633–638. https://doi.org/10.1038/s41586-025-09422-z

Dettmers, T., Pagnoni, A., Holtzman, A. y Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized language models. Proceedings of the 36th Conference on Neural Information Processing Systems.

Fang, X., Xu, W., Tan, F. A., Zhang, J., Hu, Z., Qi, Y., Nickleach, S., Socolinsky, D., Sengamedu, S. y Faloutsos, C. (2024). Large Language Models (LLMs) on tabular data: Prediction, generation, and understanding — A survey. Transactions on Machine Learning Research. https://arxiv.org/abs/2402.17944

Feng, D., Dai, Y., Huang, J., Zhang, Y., Xie, Q., Han, W., Lopez-Lira, A. y Wang, H. (2023). Empowering many, biasing a few: Generalist credit scoring through large language models. arXiv. https://arxiv.org/abs/2310.00566

Golec, M. y AlabdulJalil, M. (2025). Interpretable LLMs for credit risk: A systematic review and taxonomy. Expert Systems with Applications, 272, 126756\.

Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? Proceedings of the 36th Conference on Neural Information Processing Systems. https://arxiv.org/abs/2207.08815

Hand, D. J. y Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: A review. Journal of the Royal Statistical Society: Series A, 160(3), 523–541.

Hegselmann, S., Buendia, A., Lang, H., Agrawal, M., Jiang, X. y Sontag, D. (2023). TabLLM: Few-shot classification of tabular data with large language models. Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, 5549–5581.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. y Chen, W. (2022). LoRA: Low-rank adaptation of large language models. Proceedings of the International Conference on Learning Representations.

Klapper, L., Singer, D., Starita, L. y Norris, A. (2025). The Global Findex Database 2025: Connectivity and Financial Inclusion in the Digital Economy. World Bank. https://doi.org/10.1596/978-1-4648-2204-9

Lessmann, S., Baesens, B., Seow, H.-V. y Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring. European Journal of Operational Research, 247(1), 124–136.

Niculescu-Mizil, A. y Caruana, R. (2005). Predicting good probabilities with supervised learning. Proceedings of the 22nd International Conference on Machine Learning, 625–632.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J. y Lowe, R. (2022). Training language models to follow instructions with human feedback. Proceedings of the 36th Conference on Neural Information Processing Systems.

Parlamento Europeo y Consejo de la Unión Europea. (2024). Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). Official Journal of the European Union, L 2024/1689.

Qwen Team. (2025). Qwen3 Technical Report. arXiv:2505.09388.

Qwen Team. (2026). Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5

Siddiqi, N. (2017). Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards (2.ª ed.). Wiley.

Thomas, L. C., Edelman, D. B. y Crook, J. N. (2002). Credit Scoring and Its Applications. SIAM.

Zadrozny, B. y Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 694–699.

