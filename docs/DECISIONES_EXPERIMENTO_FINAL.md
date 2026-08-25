# Registro de decisiones — experimento del plan final (agosto 2026)

Documento de trazabilidad para la redacción de la tesis. Registra qué se
decidió, por qué, y qué implica al escribir. El experimento de referencia
anterior es la entrega intermedia (`entregas/ENTREGA-INTERMEDIA-G1-MORENO-
FEDERICO-2026.md`); este registro cubre el rediseño posterior.

## 0. Ejes discursivos de la tesis

Los experimentos de este registro alimentan siete líneas argumentales. Al
redactar, cada resultado debe colgarse de uno de estos ejes:

1. **Competencia LLM vs. ML clásico en igualdad de información** (pregunta
   central, PLAN §2.1). Los ocho modelos reciben las mismas 29 variables y la
   misma partición temporal; el LLM solo agrega texto donde el diseño lo
   declara. Evidencia: tabla comparativa del monitoreo, progresión de los
   8 modelos (§2.2 del PLAN).
2. **El valor del contexto cualitativo humano.** ¿Leer texto aporta
   ordenamiento? Se responde con la escalera de perfiles: 29 variables →
   + descripción → + rubro/tipo/otra categoría/objetivo declarado (perfil
   full). La entrega mostró ΔAUC negativo con solo la descripción; el perfil
   full testea si el contexto completo revierte eso.
3. **Razonamiento como objeto de estudio, no solo como mecanismo.** El corpus
   de thinking de Qwen (guardado caso por caso) permite auditar QUÉ variables
   pondera el modelo y contrastarlo con SHAP (fidelidad vs. plausibilidad de
   las explicaciones, AlMarri et al. 2025; relevante para el EU AI Act que
   clasifica el scoring como alto riesgo).
4. **Thin-file e inclusión financiera.** Todas las métricas se reportan por
   segmento esparso/denso del buró: ¿el conocimiento preentrenado del LLM
   compensa la falta de historial donde el ML clásico se degrada?
5. **De la discriminación a la decisión económica.** AUC no alcanza para
   operar: matrices de confusión, umbral de costo mínimo y función de costos
   con montos y tasas reales muestran que un modelo que ordena bien puede
   decidir mal si está descalibrado (XGBoost: umbral óptimo 0,01, Brier 0,26).
6. **Robustez temporal y sesgo de selección.** El corrimiento de mora
   (train 0,70 → test 0,39) y los PSI altos condicionan toda lectura; la
   partición temporal congelada es la garantía metodológica y también la
   fuente de la dificultad.
7. **Reproducibilidad y costo computacional.** Todo corre en hardware de
   consumo (M4, Ollama local) o con costo de API medido y publicado; el
   contraste con LendingClub sumará el eje de generalización y contaminación
   de preentrenamiento.

## 1. Arquitectura: de Kedro a pipeline por carpetas

**Decisión.** Se reemplazó el flujo Kedro por `pipeline/`: un contrato único
(`contrato.py`) y una carpeta por etapa/modelo (preprocesamiento, regresión
logística, xgboost, tabulares, qwen, gpt, finetuning, lendingclub), con un
módulo común de monitoreo (`monitoreo.py`).

**Por qué.** Simplicidad y legibilidad para la defensa: cada modelo del PLAN
§2.2 vive en una carpeta con su script; la partición, las variables y el
target se declaran UNA vez y todos los pasos los importan. Kedro se eliminó
por completo del repositorio (código, configuración, dependencias, tests y
layout de datos por capas) bajo la regla de trabajo "lo que no se usa se
saca, no se aclara"; del dominio sobrevivieron solo `utils.py` y
`cohorte.py`. La única mención restante vive en la entrega intermedia
publicada, que es evidencia histórica y no se modifica.

**Para la redacción.** La sección de reproducibilidad puede describir el
flujo como seis etapas: cohorte → variables → humanización → modelos →
predicción → monitoreo, todas gobernadas por un contrato explícito.

## 2. Partición temporal 70/15/15 (cambio respecto de la entrega)

**Decisión.** Split temporal por `fecha_desembolso`: train 2.940 / val 630 /
test 631 (70/15/15), alineado con el PLAN §2.3. La entrega intermedia usó
80/10/10 (3.360/420/421).

**Por qué.** Consistencia con el PLAN y test más grande (mejores intervalos).

**Implicancia crítica.** Los resultados de este experimento NO son
comparables caso a caso con los publicados en la entrega. Al escribir, citar
cada cifra con su experimento. Validación cruzada realizada: el pipeline
nuevo, corrido con el split viejo, reprodujo la Regresión Logística publicada
(AUC test 0,757) y quedó cerca en XGBoost (0,771 vs. 0,761, distinta búsqueda
de hiperparámetros).

**Dato para la discusión.** Con 70/15/15 el corrimiento temporal se acentúa:
tasa de mora train 0,705 / val 0,778 / test 0,393, y varias variables con PSI
muy alto (shops_daily_incomes 3,56; duemag01 3,33). Este sesgo de selección
temporal castiga más a los modelos entrenados que a los zero-shot; hay que
decirlo al comparar familias.

## 3. Few-shot: 8 y 16 ejemplos (se descarta 32)

**Decisión.** `SHOTS = (0, 8, 16)`. La entrega usó 0 y 8; el PLAN mencionaba
8–32.

**Por qué.** 32 ejemplos casi duplican el costo de contexto por caso y la
evidencia de la entrega mostró que 8 ejemplos ya degradaban (concentración de
puntajes en 0/1). 16 permite estudiar la dosis intermedia sin pagar 32.

## 4. Serialización y perfiles de texto (humanización)

**Decisión.** Paso propio (`03_humanizar.py`): cada crédito se convierte una
sola vez en texto en lenguaje natural (estilo TabLLM, Hegselmann et al. 2023)
y TODOS los LLM (Qwen y GPT) leen exactamente el mismo texto. Tres perfiles:

1. `tu_form` — 29 variables estructuradas en palabras.
2. `tu_form_description` — 29 + descripción libre del negocio (idéntico al
   diseño de la entrega; preserva la comparabilidad del ΔAUC de descripción).
3. `tu_form_description_full` — NUEVO: 29 + descripción + rubro del negocio
   (`subcategoria_texto`), tipo de crédito, otra categoría declarada y
   objetivo declarado del crédito (`objetivo_credito`, ver §5).

**Verificación de integridad.** Al agregar el perfil 3 se comprobó que los
textos de los perfiles 1 y 2 quedaron byte-idénticos (los caches de
inferencia previos siguen válidos).

**Para la redacción.** El perfil 3 es un experimento ADICIONAL al diseño de
la entrega: presentarlo como extensión ("todo el contexto cualitativo del
solicitante"), sin mezclarlo con el ΔAUC original de la descripción. En
few-shot, los ejemplos KNN usan la serialización compacta del perfil 2; las
cualitativas extra solo van en el caso a evaluar.

**Inventario exhaustivo de texto.** Se auditó el dataset original completo
(1.146 columnas, 388 de texto): las únicas columnas de texto humano útiles
son `descripcion_negocio` (82,8% cobertura, ~64 caracteres, 3.350 únicos),
`otra_categoria_negocio` (29,5%), `subcategoria_texto` (categórica, 37
rubros), `tipo_credito` (binaria) y `credits_credit_alternative_goal`
(54,4%, 1.498 únicos). No existe ninguna otra fuente de texto libre; redes
sociales (~13%) se descartaron por cobertura y por ser cuasi-PII.

## 5. Extracto del dataset original (paso 00)

**Decisión.** `00_extraer_original.py` filtra del dataset original-original
(1.146 columnas, con PII cruda) solo las filas de la cohorte, mapeando IDs
con la misma anonimización de la tesis (sha256 con sal, 12 hex). Cobertura:
4.201/4.201. El extracto vive en `data/00_original/` y está EXCLUIDO de git.

**Usos autorizados.** (a) Texto adicional del perfil full (`objetivo_credito`);
(b) montos y tasas reales para la función de costos (§7):
`credits_amount_granted` (mediana 250.000), `credits_interest_amount`.

**Límite metodológico.** Las condiciones finales del crédito (monto otorgado,
cuotas, interés) NO entran como predictores — son posteriores a la decisión
(fuga). Solo se usan para valorizar errores en la evaluación económica.

## 6. Razonamientos guardados (thinking / reasoning)

**Decisión.** Toda inferencia LLM persiste su razonamiento completo, caso por
caso, en JSONL reanudables (`data/pipeline/razonamientos/`):
- Qwen (Ollama, thinking nativo): campo `thinking`, ~3.000-4.000 caracteres
  por caso, más respuesta, probabilidad, tokens y timestamp.
- GPT (OpenAI Responses API, `reasoning summary "auto"`): campo `reasoning`.
  Limitación observada: la organización de OpenAI no está verificada para
  resúmenes de razonamiento, por lo que el campo puede venir vacío; la
  respuesta y el `usage` (costo) se guardan siempre.

**Por qué.** La entrega intermedia NO conservó las respuestas de Qwen (§3.11)
y eso impidió auditar el razonamiento. Este corpus habilita la comparación
explicaciones-en-lenguaje-natural vs. SHAP (PLAN §2.4, metodología AlMarri
et al. 2025) y la mejora de prompts. Contiene datos de casos: NO se publica.

## 7. Evaluación: métricas, matrices de confusión y función de costos

**Decisión.** El monitoreo común reporta dos familias:
- Sin umbral (comparación entre modelos): AUC, Gini, KS, Brier, por segmento
  esparso/denso.
- Con umbral (decisión): matriz de confusión, accuracy, precision, recall,
  F1 — a umbral 0,5 y al umbral que minimiza el costo esperado.

**Función de costos.** `COSTOS = {costo_fn: 5, costo_fp: 1}` como razón
inicial (aprobar un moroso cuesta 5× rechazar un buen pagador). Es un
placeholder DECLARADO: la versión definitiva usará montos y tasas reales del
extracto original (costo FN = capital no recuperado; costo FP = interés
perdido), trabajo asignado al agente `pipe-xgb-costos`.

**Hallazgo ya observado (primera corrida).** El umbral 0,5 es engañoso con el
corrimiento de prevalencia (train 70% mora, test 39%): XGBoost a 0,5 da
recall 0,12 y costo 1.103; a su umbral de costo mínimo (0,01) da recall 0,96
y costo 339. Además su Brier alto (0,26-0,27) indica probabilidades
descalibradas: ordena bien pero la magnitud no es una probabilidad.
**Pendiente declarado:** calibración isotónica post-hoc (Zadrozny & Elkan
2002) antes de reportar métricas por umbral definitivas — coincide con lo
previsto en el PLAN §2.4.

## 8. XGBoost: búsqueda Optuna

**Decisión.** Optuna TPE, objetivo AUC de validación, early stopping en val;
500 trials ejecutados.

**Hallazgo.** 15 trials: AUC val 0,758 / test 0,754. 500 trials: AUC val
0,780 / test 0,751. La búsqueda larga mejoró validación pero NO test: la
señal está saturada y el techo lo pone el dato, no el ajuste. Dato honesto
para la discusión (y consistente con Grinsztajn et al. sobre tabular).

## 9. Modelos pendientes y decisiones de implementación

- **Tabulares (#3):** TabPFN v2 y/o TabFM (google-research/tabfm; in-context,
  API scikit-learn, ~100 filas de contexto con ensamblado, pesos de licencia
  NO comercial — citarla). TabFM NO es un modelo de series de tiempo.
- **Fine-tuning (#8):** en Apple Silicon se usa MLX-LM (`mlx_lm.lora`), no
  Unsloth (requiere CUDA; corregir la mención del PLAN §2.2.2). Publicación:
  fuse → GGUF → `ollama create` con la MISMA cuantización que el Qwen base,
  para que #8 vs. #5-7 aísle el efecto del fine-tuning sin cambiar runtime.
- **Series de tiempo (extensión, no priorizada):** Chronos/TimesFM sobre la
  secuencia de pagos por crédito, misma partición del contrato.
- **LendingClub (dataset 2):** repetir el flujo completo. Punto a discutir sí
  o sí: contaminación de preentrenamiento (dataset público que Qwen/GPT
  pudieron ver); comparar el desempeño zero-shot relativo entre datasets. A
  esa escala usar la Batch API de OpenAI (50% de descuento).

## 10. Costos de API (medidos, no estimados)

Con `usage` real de la primera corrida GPT: ~670 tokens input y ~590 output
por caso (72% del output es razonamiento). A precios de la familia GPT-5:
zero-shot completo (2 perfiles × 631) ≈ USD 8-9; experimento GPT completo
(3 perfiles × 3 dosis de ejemplos) ≈ USD 30-35. Se decidió mantener
`reasoning effort medium` porque el razonamiento es objeto de estudio.

## 11. Reproducibilidad y publicación

- Repo público `federicomoreno613/credit-risk-frontier`, rama `master`.
- Se versionan datos anonimizados, parquets, modelos y razonamientos
  (decisión explícita de agosto 2026: máxima reproducibilidad; los IDs son
  hashes con sal y no individualizan personas).
- NO se versionan: `data/00_original/` (PII cruda), `.env` (credenciales),
  `nbs/`, `knowledge/`, configuración de agentes.
- La partición está congelada en `data/pipeline/01_manifiesto_particion.json`
  (hashes SHA-256 de los IDs por conjunto).
- Corridas LLM reanudables con huella por configuración; tablero de avance en
  `pipeline/estado.py`.

## 12. Correcciones pendientes al PLAN (para la versión final)

1. §2.3: el split dice 70/15/15 — ahora implementación y PLAN coinciden
   (la entrega intermedia queda documentada con 80/10/10).
2. §2.2: few-shot "8-32" → "8 y 16".
3. §2.2.2: "Unsloth en Apple Silicon" → "MLX-LM (LoRA) y publicación vía
   GGUF/Ollama"; Unsloth requiere CUDA.
4. §2.2: sumar el perfil cualitativo completo como configuración adicional
   de los LLM (extensión del diseño de 6 configuraciones de la entrega).
5. Métricas: explicitar que accuracy/precision/recall se reportan a umbral de
   costo mínimo además de 0,5, y que la calibración isotónica precede a esas
   métricas.
