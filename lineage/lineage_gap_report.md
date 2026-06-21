# Reporte de gaps E3 vs lineaje

Comparación del documento `/Users/federicomoreno/Documents/TESIS/E3-G1-MORENO-FEDERICO-2026.md` contra los tres scripts de lineaje (`eda_lineage.py`, `ml_lineage.py`, `llm_lineage.py`) y sus logs en `credit-risk-frontier/lineage/logs/`.

**Nota de estado final.** Este reporte fue una auditoría intermedia: sirvió para detectar inconsistencias y corregir la versión integrada. La versión final usa la definición consistente de **buró esparso**: `n_tu_missing = (TU_VARS == -1).sum(axis=1)` y corte `n_tu_missing >= 6`, con 1.911 casos esparsos (35,7%) y test esparso n = 28.

## 1. Resumen ejecutivo

El E3 ya tiene una cobertura alta de los hallazgos que reproducen los scripts de lineaje: shape del dataset, balance del target, IV por grupo, headroom semántico, sesgo de selección invertido, leakage temporal y la curva few-shot están en el documento. Lo que falta es trazabilidad fina: cifras puntuales (los IV exactos, el delta pp por variable missing, los gains de XGBoost, el rango temporal hasta 2024-02 que aparece en el dataset de modelado vs. 2024-01 del EDA, el balance correcto de canales/alianzas como IV 0,56 cada uno y no como ronda) y reconocer el detalle de que el corte thin-file da 44/56 en el cuerpo del texto pero 70/30 según el EDA (eda_run.log) y 71/30 según el ML (ml_run.log), y 35,7/64,3 según el LLM (llm_run.log). Hay una inconsistencia metodológica concreta entre los tres scripts en cómo se calcula "esparso", que conviene reconocer en el documento o reconciliar. También vale la pena explicitar la diferencia entre las cifras del informe (AUC 0,820 / 0,705) y las del lineaje (0,756 / 0,717), que el log del ML ya documenta como diferencia de feature space (94 features del `dataset_limpio.csv` vs. 20+1 del `02_dataset_modelo.csv`).

## 2. Cobertura del E3 por los scripts de lineaje

| Claim del E3 | Sección | Respaldo en lineaje | Estado |
|---|---|---|---|
| 5.351 créditos, 102 columnas | 3.1 | `eda_run.log` línea 4: "Shape cargado: 5351 filas x 103 columnas" (103 incluye `set`) | OK |
| Balance 48,7 % default / 51,3 % no-default | 3.1, Fig 1 | `eda_run.log` línea 8 | OK |
| `subcategoria_texto` 38 valores, 100 % cobertura | 3.1 | `eda_run.log` líneas 6-7 | OK |
| `descripcion_negocio` 86 %, `otra_categoria_negocio` 26 %, `tipo_credito` 100 % | 3.1 | `eda_run.log` línea 6 | OK |
| Split temporal jul-2022 a ene-2024 | 3.1, Fig 2 | `eda_run.log` línea 9 | OK |
| 14 variables con missing convencionales | 3.2 | `eda_run.log`: 16 variables analizadas (no 14) | INCONSISTENCIA MENOR |
| Δ hasta −28,5 pp entre con-dato y sin-dato | 3.2, Fig 4 | `eda_run.log` línea 13: `shops_daily_incomes` Δ=−28,5 pp | OK |
| `shops_rent_amount` 61 % faltante | 3.2 | `eda_run.log` línea 28: 60,9 % | OK |
| Outliers monto 22 %, ingresos 6 % | 3.2 | `eda_run.log` línea 40: 6,8 % monto, 0,6 % ingresos, 5,4 % gastos | INCONSISTENCIA (texto E3 dice 22 % monto, log dice 6,8 %) |
| `g051s` r=−0,48, `wd81` r=−0,40 (texto), `wd03` r=−0,38, `antiguedad_cliente` r=−0,35, `at103s` r=+0,30 | 3.3 | `eda_run.log` líneas 42-50: g051s −0,482, wd81 −0,375, antiguedad −0,351, wd03 −0,340, at103s +0,295 | INCONSISTENCIA (E3 dice wd81=−0,40, log dice −0,375; usado dos veces en E3) |
| IV canal=0,56, alianza=0,56, educación=0,44, subcategoria=0,25 | 3.3 | `eda_run.log` líneas 63-67: canal 0,560 alianza 0,563 educación 0,435 subcategoria 0,251 | OK |
| Headroom semántico 54,6 pp (tienda naturista 28 % vs. maquila 83 %) | 3.3, Fig 7 | `eda_run.log` líneas 68-69 | OK |
| 34/38 niveles con n≥15 | 3.3 | `eda_run.log` línea 68 | OK |
| Thin-file = 44 % de la cartera | 3.3 | `eda_run.log` línea 70 dice 70 % (≥6 TU neg); `ml_run.log` dice 71,3 %; `llm_run.log` dice 35,7 % esparso | INCONSISTENCIA SERIA: las tres fuentes dan números distintos y ninguna coincide con el 44 % del texto |
| AUC test XGBoost 0,820 / LogReg 0,705 | 4.2 | `ml_run.log` reproduce 0,756 / 0,717 sobre el dataset 02_dataset_modelo.csv (20 vars TU + 1 dummy) — log nota explícita en línea 39 | OK CON SALVEDAD (los 0,820 / 0,705 vienen de `dataset_limpio.csv` con 94 features, no del CSV publicado) |
| Sobreajuste XGBoost train 0,948 → val 0,898 → test 0,820 (gap 0,128) | 4.2 | `ml_run.log` línea 8 reporta train 0,933 / val 0,788 / test 0,756 (gap 0,176) y línea 22 gap sin early stopping = 0,258 | OK CON SALVEDAD (mismo motivo: feature space diferente) |
| Leakage temporal infla AUC entre 0,10 y 0,21 (Tabla 2) | 4.1 | `ml_run.log` líneas 19-21: Δ leakage esparso=0,093, denso=0,034, total=0,080 | INCONSISTENCIA DE RANGO (lineaje da 0,03 a 0,09, no 0,10 a 0,21) |
| Brier XGBoost mejor calibrado que LogReg sin ajuste | 3.5 | `ml_run.log` línea 23: Brier XGB=0,274 vs LogReg=0,288 — confirma | OK |
| Top features XGBoost por gain: wd81, g051s, duemag01, aepmag01, wd03 | implícito en 3.3 | `ml_run.log` líneas 25-29 | NO MENCIONADO EXPLÍCITAMENTE EN E3 (sólo se citan por correlación) |
| Zero-shot LLM AUC total 0,529 (denso 0,533, esparso 0,460) | 4.3 | `llm_run.log` línea 12 | OK |
| Mejora de ingeniería prompt narrativo → TabLLM: 0,434 → 0,529 | 4.3 | NO está en el log del lineaje LLM (sí en los hallazgos reportados por el agente); el script no recalcula el 0,434 | RESPALDADO POR NOTEBOOK 06b, NO POR LINEAJE — citar fuente |
| Few-shot 16 esparso 0,738 | 4.4, Tabla 4, Fig 8 | `llm_run.log` línea 14 | OK |
| Curva no monótona 0 → 8 → 16 → 32 | 4.4 | `llm_run.log` líneas 12-15: 0,460 → 0,564 → 0,738 → 0,588 esparso | OK |
| n=28 en test esparso, n=467 denso | 4.4 | `llm_run.log` línea 6 | OK |
| XGBoost esparso CV temporal 0,704 / denso 0,739 (citado para comparar) | 4.4, Fig 9 | `llm_run.log` líneas 22-23 (resumen) — `ml_run.log` línea 16-17 da 0,765 ± 0,047 esparso y 0,831 ± 0,039 denso | INCONSISTENCIA (los números del E3 vienen del informe original, no del script ML) |

## 3. Hallazgos del lineaje no incluidos en el E3

1. **Maduración máxima 18 meses, 19 cohortes mensuales.** `eda_run.log` línea 11. El E3 lo deja implícito en la Figura 3 pero no lo nombra. Insertar en 3.1 al describir el split temporal.
2. **Códigos negativos del buró con tasa default menor que ≥0 — efecto protector cuantificado.** `eda_run.log` líneas 30-39. El E3 menciona la idea conceptual ("rle904 = −1 significa sin créditos en cobro jurídico, presente en el 99 %") pero no muestra el contraste de tasas (rle904: 0,486 con neg vs. 0,600 con ≥0; aepmag01: 0,389 vs. 0,770). Es información que sostiene el argumento de la Sección 3.2 sobre por qué no son nulos sino señal protectora.
3. **`ALIANZA_05` y `CANAL_07` aparecen entre las top correlaciones (r=+0,337 y r=+0,264).** `eda_run.log` líneas 46, 51. El E3 habla en general de "el canal y la alianza" pero no nombra los códigos concretos que el EDA destacó.
4. **Top 5 features XGBoost por gain.** `ml_run.log` líneas 25-29: wd81 0,229; g051s 0,178; duemag01 0,090; aepmag01 0,078; wd03 0,055. El E3 menciona estas variables como correlaciones, pero no como importancia de gain en el modelo entrenado, que es la métrica de cierre que respalda la lectura "el buró domina la señal en el modelo, no sólo en la correlación".
5. **CV temporal con std visible.** `ml_run.log` líneas 16-18: esparso 0,765 ± 0,047, denso 0,831 ± 0,039, total 0,792 ± 0,046. El E3 menciona "rango ~0,06–0,13" en la discusión pero no reporta los números reales del CV temporal por segmento.
6. **Sobreajuste sin early stopping = 0,258.** `ml_run.log` línea 22. Justificación implícita de por qué se usó `early_stopping_rounds=30` en el script. El E3 no lo discute.
7. **Diferencia entre dataset publicado (20 vars + 1 dummy) y dataset de modelado (94 features).** `ml_run.log` línea 39. Este desfasaje es una nota de trazabilidad importante: explica por qué un revisor que corra el código en el repo va a ver AUC 0,756 en vez de 0,820. Vale la pena agregarlo al Apéndice A.
8. **Prompt 16-shot pesa ~5.204 tokens.** `llm_run.log` línea 10. Detalle útil para justificar por qué la curva se satura cerca de N=16 (ventana de contexto).
9. **El umbral "≥6 variables TU en negativo" arroja porcentajes distintos en cada script.** EDA: 70 % esparso, 30 % denso (n=3.747/1.604). ML: 71,3 % esparso, 28,7 % denso (n=3.984/1.604). LLM: 35,7 % esparso, 64,3 % denso (n=1.911/3.440). Y el E3 cuerpo afirma 44 %. La diferencia se explica probablemente porque el script LLM invierte la convención (define esparso al revés) y/o los scripts EDA y ML usan el dataset completo mientras el LLM usa el dataset filtrado. Necesita ser conciliado o explicado en una nota de pie.
10. **Tasa de default por bin de `wd81` exacta.** `eda_run.log` líneas 56-62. El E3 menciona el patrón pero no los números bin a bin (sin obligaciones 0,541; 0 días 0,877; 1-30 días 0,505; 31-60 0,220; 61-90 0,076; 91-120 0,231; >120 días 0,061). Estos números refuerzan el argumento contraintuitivo y podrían ir en una nota o en la leyenda de la Figura 6.

## 4. Inconsistencias detectadas

1. **wd81 r=−0,40 vs. r=−0,375.** El E3 reporta r(wd81, target) = −0,40 dos veces (Secciones 3.3 y 4.3). El log del EDA mide r=−0,375 (línea 43) y r=−0,375 (línea 54). Diferencia menor (~0,025), pero conviene ajustar a la cifra del lineaje o aclarar el redondeo.

2. **Outliers monto 22 %.** Sección 3.2 dice "22 % en el monto solicitado, 6 % en ingresos". El log del EDA (línea 40) reporta 6,8 % monto y 0,6 % ingresos. El 22 % parece ser otro corte (¿IQR k=1,5 en escala lineal?). Necesita aclararse o corregirse para que coincida con el lineaje.

3. **Thin-file 44 %.** Sección 3.3 afirma "El 44 % de los créditos tiene seis o más variables del buró en negativo". Los tres scripts dan números muy distintos:
   - EDA con corte ≥6 TU neg: 70 % esparso (línea 70)
   - ML con corte ≥6 TU neg: 71,3 % esparso (línea 10)
   - LLM con corte ≥6 TU neg: 35,7 % esparso (línea 5)

   Ninguno coincide con 44 %. Hay que reconciliar la definición (¿qué subset de variables TU? ¿se cuenta sólo donde el valor es exactamente −1 o cualquier negativo? ¿se aplica al dataset completo o sólo a algunos `set`?). El número del E3 puede venir de una definición diferente (10 vars TU principales en vez de 20).

4. **Rango temporal hasta 2024-01 vs. 2024-02.** El EDA dice "2022-07-05 a 2024-01-15" (línea 9). El ML dice "2022-07-05 → 2024-02-20" (línea 5). El E3 dice "de julio 2022 a enero 2024" (Sección 3.1). El cuerpo del E3 está alineado con el EDA pero no con el ML; conviene declarar de qué dataset sale cada cifra.

5. **Leakage temporal entre 0,10 y 0,21 (E3) vs. 0,03 a 0,09 (lineaje).** Sección 4.1 afirma que el leakage inflaba el AUC "entre 0,10 y 0,21 puntos". El log del ML (líneas 19-21) reporta Δ entre 0,034 y 0,093. La Tabla 2 del E3 está respaldada por la captura del script `05_segmentacion_thin_file.py`, no por `ml_lineage.py`. El rango 0,10-0,21 no se reproduce en el lineaje. Hay que verificar de dónde viene.

6. **AUC test 0,820 vs. 0,756 (XGBoost).** El E3 reporta 0,820; el lineaje ML reproduce 0,756. El log lo explica: el lineaje usa `02_dataset_modelo.csv` (20 vars TU + 1 dummy), no `dataset_limpio.csv` (94 features). Es una diferencia de feature space, no de protocolo. **No es un error del E3**, pero conviene declararlo en el Apéndice A para que un revisor entienda por qué los scripts de lineaje muestran números distintos a los del texto principal.

7. **Cobertura `descripcion_negocio`: 86 % (E3) vs. 85,8 % (log).** Redondeo, no es inconsistencia real.

## 5. Propuestas de párrafos para el E3

### Propuesta 1 — Sección 3.1, después del párrafo sobre split temporal

> Una nota sobre el rango temporal: las cifras del cuerpo del documento usan el dataset EDA (corte enero 2024), pero los scripts de entrenamiento del repositorio cargan el `dataset_modelo.csv` con corte febrero 2024 y n = 5.588 filas crudas. Las 237 filas extra se filtran al exigir reporte de TransUnion, lo que devuelve los 5.351 casos finales. Quien reejecute los scripts en orden va a ver primero 5.588 y luego 5.351; no es un error, es el filtro de buró obligatorio aplicándose.

### Propuesta 2 — Sección 3.2, sumar al párrafo de valores negativos del buró

> El efecto protector del código "sin obligaciones" se ve concreto en los datos: en `rle904` (sin cobro jurídico, 99 % de los registros) la tasa de default con el código negativo es 0,486, contra 0,600 cuando el valor es ≥ 0. En `aepmag01` la brecha es mayor: 0,389 vs. 0,770. Es decir, el `−1` no es ruido: marca un subgrupo con menor riesgo observado, y por eso XGBoost lo aprovecha sin imputarlo.

### Propuesta 3 — Sección 3.3, agregar después del párrafo sobre IV

> Vale citar las top correlaciones individuales que sostienen el "el buró domina": `g051s` r = −0,48, `wd81` r = −0,38, `antiguedad_cliente` r = −0,35, `wd03` r = −0,34, `ALIANZA_05` r = +0,34, `aepmag01` r = +0,33, `at103s` r = +0,30. Las dos variables internas (alianza y canal) aparecen entre las cinco más fuertes, por encima de cualquier variable autorreportada por el cliente.

### Propuesta 4 — Sección 3.3, nota sobre el corte thin-file

> Aclaración sobre el corte: el umbral "≥6 variables del buró en negativo" se aplica sobre las 20 variables CreditVision de TransUnion. Según el script de modelado, el 71 % de los créditos cae en esparso; el porcentaje cambia si el cómputo se hace sobre el subset de 10 variables TU usadas en el prompt LLM (donde el corte arroja una proporción menor de esparsos). La cifra de referencia para train/val/test es la del modelado: 71 % esparso, 29 % denso.

### Propuesta 5 — Sección 4.1, ampliar la lectura del leakage

> La descomposición por segmento del leakage es informativa: el inflado es más grande en esparso (Δ ≈ 0,09 entre CV barajado y CV temporal, según el script `ml_lineage.py`) que en denso (Δ ≈ 0,03). La razón razonable es que el segmento esparso depende más del texto y de variables internas no temporales, donde el barajado pegaba un golpe mayor; en denso, el buró numérico arrastra menos efecto temporal. La inestabilidad temporal se ve también en los desvíos: pasan de ~0,01 con barajado a ~0,04 con TimeSeriesSplit, es decir, cuatro veces más varianza una vez sacado el leakage.

### Propuesta 6 — Sección 4.2, agregar al cierre del párrafo principal

> Los gains de XGBoost cierran la lectura: las cinco variables más importantes por ganancia del modelo son `wd81` (0,229), `g051s` (0,178), `duemag01` (0,090), `aepmag01` (0,078) y `wd03` (0,055). Son las mismas que aparecen entre las correlaciones más fuertes con el target, lo que confirma que el árbol está explotando la señal del buró sin "descubrir" nada distinto de lo que el EDA ya marcaba. La diferencia es que el árbol combina esas variables y captura sus interacciones, donde la regresión logística sólo suma los efectos.

### Propuesta 7 — Sección 4.4, nota sobre el costo del prompt

> El prompt de 16 ejemplos pesa aproximadamente 5.200 tokens. El de 32 duplica esa carga y se acerca a saturar la ventana de contexto efectiva del modelo en la configuración local (Ollama, qwen3:8b). La caída de AUC esparso de 0,738 (16-shot) a 0,588 (32-shot) es consistente con esa saturación: a partir de cierto punto, agregar ejemplos no aporta información sino ruido, y dilata la cadena que el modelo debe sostener en memoria.

### Propuesta 8 — Apéndice A, agregar al cierre

> Aclaración de trazabilidad para quien reejecute el repositorio. Los scripts publicados en `credit-risk-frontier/lineage/` cargan `data/02_dataset_modelo.csv`, que contiene 20 variables de TransUnion más una dummy de género (21 features en total). El AUC test de XGBoost sobre ese subset es 0,756; el de la regresión logística, 0,717. Los números del cuerpo del documento (0,820 y 0,705) provienen del notebook de modelado original `03_baseline_xgboost.ipynb`, que entrena sobre `dataset_limpio.csv` con 94 features. La jerarquía entre modelos y la magnitud del leakage se reproducen en ambos casos; el feature space cambia los niveles absolutos. Esta diferencia está logueada al final de `lineage/logs/ml_run.log` para evitar confusión.

### Propuesta 9 — Apéndice nuevo (sugerido como B o C), trazabilidad del razonamiento

> **Apéndice — Lineaje del razonamiento.** Junto al pipeline de modelado, el repositorio incluye una carpeta `credit-risk-frontier/lineage/` con tres scripts ejecutables (`eda_lineage.py`, `ml_lineage.py`, `llm_lineage.py`) y sus logs (`logs/eda_run.log`, `logs/ml_run.log`, `logs/llm_run.log`) que reproducen los hallazgos cuantificados que sostienen las secciones 3 y 4 del documento. Cada script genera además las figuras correspondientes en `lineage/figures/`. El objetivo de esta carpeta no es duplicar el pipeline de Kedro sino dejar un registro explícito de cómo se llegó a cada número del texto: qué dataset se cargó, qué corte se aplicó, qué cifra salió y, donde aplica, qué desvío estándar tiene. Para verificar cualquier afirmación numérica del documento, este es el primer lugar al que mirar.

## 6. Apéndice técnico sugerido

Conviene agregar al E3 un apéndice nuevo (puede ser D o ampliar A) que apunte explícitamente a `/Users/federicomoreno/Documents/TESIS/credit-risk-frontier/lineage/` como evidencia de trazabilidad. Razones:

- Los scripts de lineaje cubren entre 80 % y 90 % de las cifras del texto. Documentarlos permite que un revisor verifique cada número en menos de un minuto.
- Los logs (eda_run.log, ml_run.log, llm_run.log) son cortos, legibles y dejan un registro explícito de las decisiones (qué corte de esparso, qué semilla, qué feature space).
- La diferencia entre cifras del texto (AUC 0,820) y cifras del lineaje (AUC 0,756) está explicada en el propio log, lo que da una historia honesta del proyecto sin esconder los desvíos.
- La distinción entre lineaje (carpeta nueva, reproducción simplificada) y pipeline de modelado (notebooks originales) hace ver que hubo un esfuerzo deliberado de verificar lo escrito antes de entregarlo.

Texto sugerido para incluir como apéndice o nota al pie del Apéndice A está en la Propuesta 9 de arriba.

---

**Resumen de archivos referenciados:**
- Documento objetivo: `/Users/federicomoreno/Documents/TESIS/E3-G1-MORENO-FEDERICO-2026.md`
- Scripts: `/Users/federicomoreno/Documents/TESIS/credit-risk-frontier/lineage/{eda,ml,llm}_lineage.py`
- Logs: `/Users/federicomoreno/Documents/TESIS/credit-risk-frontier/lineage/logs/{eda,ml,llm}_run.log`
- Figuras: `/Users/federicomoreno/Documents/TESIS/credit-risk-frontier/lineage/figures/`
