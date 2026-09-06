# Informe de redacción — Capítulo 1 (v3 → v4)

Fecha: 2026-09-06. Fuente evaluada: `1-introduccion-v3.md` (= `TESIS-cap1-introduccion.docx` v3). Patrón de voz: introducción y marco teórico del E4 aprobado (`Entrega_aprobada_hasta ahora/Entrega 4 (1).md`) y `GUIA-REDACCION-Y-VALIDACION-TESIS-FINAL.md`.

## 1. Diagnóstico: qué se alejaba de tu voz

| Métrica | Tu E4 aprobado | Cap. 1 v3 | Cap. 1 v4 |
|---|---|---|---|
| Palabras por oración (media / mediana) | 23 / 21 | 28 / 27 | 23 / 21 |
| Oraciones de más de 40 palabras | 7 de 78 | 60 de 281 | 29 de 332 |
| Oraciones de más de 50 palabras | 0 | 15 | 3 |
| Rayas (—) cada 1.000 palabras | 1,1 | 4,8 | 0 |
| «acá» / «conviene» / «justamente» / «exactamente» | 0 / 0 / 0 / 0 | 6 / 10 / 6 / 4 | 0 / 2* / 0 / 0 |
| Palabras totales | — | 8.141 | 7.558 (~20,5 páginas Calibri 12, 1,5) |

\* Las dos apariciones que quedan son el verbo en sentido económico («aprobar conviene mientras…», «el esfuerzo conviene dirigirlo…»), no la muletilla «Conviene señalar que…».

Marcadores de redacción de IA detectados en la v3 y eliminados:

- Remates aforísticos ajenos a tu registro: «un retroceso disfrazado de avance», «Esta tesis trabaja con una.», «Conviene poner cara al segmento», «esta es la parte que suele quedar afuera de un trabajo técnico», «de punta a punta», «en definitiva».
- Imperativos formales de manual: «Considérese un crédito…», «llámese *p*…» → «Sea *M* el monto…».
- Incisos con rayas encadenados (hasta 3 por oración) → comas u oración nueva.
- Antítesis retórica «no es X sino Y» y «no solo… sino también» usadas como cierre de párrafo (reducidas de 8 a 5, las que quedan son distinciones de contenido).
- Cuatro oraciones en la v3 empezaban con gerundio; ninguna en la v4.
- Adjetivos de relleno: «interesante» (2), «sutil», «particularmente».

## 2. Correcciones de contenido (no de estilo)

1. **Findex mal enunciado.** La v3 decía «alrededor de un tercio de la población adulta mundial sigue sin acceso a servicios financieros formales». El dato de Demirgüç-Kunt et al. (2025) que usás en el E4 es: 79 % de los adultos del mundo con cuenta y ~70 % en América Latina y el Caribe **excluidas las economías de ingreso alto**. La v4 usa esa formulación.
2. **Párrafo duplicado en §1.5.** El argumento del subsidio cruzado con Edelberg (2006) aparecía dos veces (párrafo de la aritmética y párrafo «El costo de decidir mal tampoco se agota…»). Quedó una sola vez. Es la única cita que perdió una aparición; el conjunto de autores citados es idéntico entre v3 y v4 (verificado por script).
3. **«Código centinela»** reemplazado por «un código que indica ausencia de información», según tu convención de evitar jerga sin explicar (decisión de julio).
4. **Autoría Crook y Banasik (2004)** verificada contra Semantic Scholar, ScienceDirect e IDEAS/RePEc: el orden correcto es Crook y Banasik, como estaba en el capítulo. Había una nota de memoria de julio que decía lo contrario; quedó corregida.
5. **Adelanto de resultados eliminado** (tu pedido de hoy): §1.9 y el Cuadro 1 salen de la introducción. Los Cuadros 2–8 del resto del documento pasan a 1–7 en `reestructura-intro-marco-v3.md` (13 menciones renumeradas). Las «Tabla N» del capítulo 5 son una numeración aparte y no se tocaron.
6. **Anticipos del signo del resultado neutralizados.** Sin el adelanto, quedaban frases que igual revelaban el desenlace («es lo que explica el resultado principal», «el capítulo 5 muestra que esta cartera lo contradice», «Esta tesis lo muestra actuando: documenta la dirección invertida…», «Un resultado negativo bien establecido tiene valor de decisión»). Ahora remiten al capítulo 5 sin decir qué se encontró, y el aporte 4 está escrito en los dos sentidos («si supera… / si no supera…»).

## 3. Cambios estructurales

- **Apertura nueva** (dos párrafos antes de §1.1): la escena de la solicitud desde el celular con la frase «venta de frutas en plaza de mercado», el buró que devuelve casi nada, y la pregunta en una línea. Es la tesis narrativa que ya tenías en el E4 («le damos su mejor carta y aun así…»), sin el «aun así». Si preferís entrar directo en §1.1, se borran esos dos párrafos y nada más cambia.
- **§1.6 reordenado**: condición que no se cumple → costo de construir el dataset → escasez de etiquetas → el insumo que nunca falta (texto) → por qué el texto se destruye al codificarlo → modelos de lenguaje. Antes «escasez de etiquetas» quedaba intercalado entre dos párrafos sobre texto.
- Numeración final: 1.1–1.8 sin cambios, 1.9 Aporte esperado, 1.10 Organización.

## 4. Opinión editorial (lo que me preguntaste)

**¿Está bien introducido el tema?** Sí, y mejor que en el E4: el capítulo va de lo general a lo particular sin saltos (función del crédito → puntaje y política → quién queda afuera → dónde ocurre → qué cuesta decidir mal → por qué los métodos actuales tocan techo → conjetura). Cada sección deja una pieza que la siguiente necesita. El punto más fuerte es §1.1 con la distinción ocultamiento / incertidumbre compartida / ruido, porque justifica por sí sola por qué la descripción del negocio es el dato a explorar. Ese argumento es tuyo y no está en la literatura de esta forma; cuidalo.

**¿Está bien empezar por el crédito y no por machine learning?** Sí, es lo correcto para una tesis de la Maestría en Explotación de Datos: el jurado puede venir de estadística, economía o computación, y empezar por el problema del dominio le da a todos el mismo piso. Empezar por los modelos de lenguaje haría que el trabajo parezca «una prueba de una herramienta» y no «una respuesta a un problema». La única concesión al lector impaciente es la apertura nueva: en dos párrafos ya sabe qué se compara y por qué, y después acepta cinco secciones de contexto. Sin ese gancho, un lector técnico recién encuentra los modelos de lenguaje en la página 12, que es tarde.

**¿Falta algún tema?** Los temas que un jurado esperaría están: definición de mora y su umbral, buró y su circularidad, thin-file con cifras, fintech en la región, marco legal colombiano, sesgo de selección, costos asimétricos, equidad, explicabilidad, techo de los tabulares, serialización, contaminación de datos públicos. Dos cosas que yo agregaría o reforzaría:

1. **Un párrafo sobre por qué esta cartera y no un dataset público**, en §1.7 (alcance). Hoy la razón aparece dispersa (contaminación en §1.6, «cartera privada» en §1.9). Una frase que junte las tres razones (privada → sin contaminación; colombiana → thin-file real; con texto libre → la ventaja que se quiere medir) le da al lector el porqué del dato antes del capítulo 4.
2. **El lugar del capítulo dentro de la maestría**: una línea que diga que la tesis es un trabajo de minería de datos aplicada (comparación controlada de métodos sobre un problema real), no un desarrollo de modelos. Ubica al jurado en qué criterios usar para evaluarte.

Lo que **no** agregaría: más literatura de LLM. La introducción ya tiene 11 citas de esa línea y el marco teórico (§2.4–2.5) las desarrolla. Tampoco más cifras del Findex o del BID: con las que hay alcanza.

**¿Hay palabras fuera de lugar?** En la v3, las más visibles eran «acá» (coloquial rioplatense, seis veces; en el E4 nunca lo usás), «conviene» como muletilla de ensayista, «justamente/exactamente» como refuerzo, y «Considérese/llámese» que suenan a apunte de cátedra. También «ilegible» aplicado a una persona («lo vuelve ilegible para un sistema») es una metáfora fuerte; la dejé porque es exacta y legible, pero si te suena ajena cambiala por «invisible». «Trampa de la visibilidad crediticia» es una expresión que no está en las fuentes citadas; la mantuve porque describe bien el círculo, pero conviene no ponerla en cursiva ni entre comillas para no sugerir que es un término técnico establecido.

**¿Construcciones que parecen de IA?** En la v3, sí, y de un tipo concreto: (a) oraciones largas con dos o tres incisos entre rayas; (b) párrafos que cierran con una sentencia corta y redonda («Sin mejor información, el precio no resuelve el problema.», «Esta tesis trabaja con una.»); (c) tricolones abstractos («transparencia, auditabilidad y explicabilidad»); (d) la fórmula «no es X sino Y» como remate. La v4 corta (a) casi por completo, reduce (b) a los casos donde la frase corta es un dato y no un efecto, y deja (c) y (d) solo donde son enumeraciones reales. Queda un rasgo que vos también usás y que no toqué: la frase de dos palabras después de dos puntos («Muchas veces no oculta su situación: no la conoce.»). Es tuya, no de la máquina.

## 5. Lo que queda pendiente y es tuyo

- Lectura en voz alta de §1.5 (aritmética) y §1.6 (párrafo final sobre conocimiento previo): son los dos pasajes con más carga conceptual por oración.
- Decidir si la apertura nueva se queda.
- Los dos agregados sugeridos en el punto 4 (párrafo «por qué esta cartera» y línea de encuadre en la maestría) son redacción tuya según la división de trabajo de la guía; puedo editarlos después.

## 6. Adenda v5 (misma tarde)

Decisiones de Federico: sin adelanto de resultados, registro «de tesis» (sin escena narrativa), bloque intertemporal sí, fintech anónima.

- **Apertura formal** (tres párrafos antes de §1.1): informalidad laboral 47,6 % en LAC 2024 (OIT, 2025); brecha de financiamiento mipyme 5,2 + 2,9 billones USD, LAC ≈ 1,2 billones (IFC, 2017); EMICRON 2024: 60,0 % entidad regulada / 22,9 % gota a gota / 13,3 % familiares (DANE, 2025); párrafo que declara el objeto y la pregunta de la tesis. Salieron la escena del celular y el caso de la artesana. Cifras del pitch de Quipu descartadas por no tener fuente verificable: «25 % del PIB regional de micronegocios (OCDE)», «9 de cada 10 no bancarizados», «6 millones / 9 % con crédito».
- **Bloque intertemporal en §1.1**, entre el párrafo de la promesa de pago y el de información asimétrica: ecuación (1) restricción presupuestaria de Fisher (1930) con lectura de Varian (2010); ecuación (2) (1−p)(1+i) = 1+r con Freixas y Rochet (2008); cierre: fijar qué cobrar exige estimar p, con la precisión de que (2) requiere probabilidad calibrada y la tesis mide primero ordenamiento (§1.8). §1.5 enlaza: su aritmética «es la ecuación (2) con r = 0», y ahora dice explícitamente «sin contar el costo de fondeo». Las ecuaciones se exportan como OMML nativo de Word (2 en display, 11 inline).
- **Bibliografía**: seis fichas nuevas en `bibliografia-cap1-nuevas.md/.bib` (OIT 2025, IFC 2017, DANE 2025, Fisher 1930, Varian 2010, Freixas y Rochet 2008). Total 35 entradas nuevas del capítulo 1.
- Extensión: 8.033 palabras ≈ 21,8 páginas. Métricas de voz sin cambios respecto de la v4 (0 rayas, 0 «acá»).

## 7. Adenda v6

Decisión de Federico: el bloque intertemporal de Fisher «confunde y aleja». Se quitó completo de §1.1 (las dos ecuaciones y su lectura). §1.5 conserva la aritmética de la frontera con la aclaración «sin contar el costo de fondeo», sin remisión a una ecuación previa. Se retiraron de la bibliografía las tres fichas que solo sostenían ese bloque (Fisher 1930, Varian 2010, Freixas y Rochet 2008); quedan 32 fichas nuevas. La apertura formal con OIT, IFC y DANE se mantiene. Extensión: 7.719 palabras ≈ 21 páginas.
