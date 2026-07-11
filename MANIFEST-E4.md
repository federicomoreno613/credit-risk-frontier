# Entregables E4 — credit-risk-frontier

Manifiesto de los artefactos correspondientes a la Entrega 4 (comparación de modelos de
lenguaje contra aprendizaje automático clásico en puntuación de crédito de microfinanzas).
Documento final: `E4-G1-MORENO-FEDERICO-2026.docx` (raíz de TESIS).

## Scripts (fuente autoritativa, orden de ejecución)

| Script | Rol |
|---|---|
| `scripts/01_preparar_dataset.py` | Construcción del dataset (n = 4.897), partición temporal train/val/test. |
| `scripts/03_baseline_xgboost.py` | XGBoost. Baseline citable **sin filtración: AUC test 0,7516** (ver nota de baseline). |
| `scripts/04_baseline_logreg.py` | Regresión Logística. Baseline citable **sin filtración: AUC test 0,6605**. |
| `scripts/05_segmentacion_thin_file.py` | Segmentación historial escaso / denso. |
| `scripts/06b_llm_prompting.py` | Serialización TabLLM y prompting base del modelo de lenguaje. |
| `scripts/07_comparacion_final.py` | Integración de métricas y tabla comparativa. |
| `scripts/08_llm_fewshot.py` | Few-shot con recuperación de vecinos (kNN) por clase. |
| `scripts/09_llm_thinking.py` | Corridas en **modo de razonamiento** (Qwen3-8B, Gemma 4 12B vía Ollama). |
| `scripts/09b_diagnostico_un_caso.py` | Diagnóstico de un caso: fila, prompt exacto, respuesta y score (chequeo previo). |
| `scripts/09c_llm_gpt.py` | Corrida GPT-5.4 vía API de OpenAI (mismo protocolo, endurecido tras revisión adversarial). |
| `scripts/09d_f1_posthoc.py` | Cálculo de F1 post-hoc sobre las probabilidades guardadas. |
| `scripts/10_figuras_llm.py` | Figuras de resultados del modelo de lenguaje. |
| `scripts/12_ablacion_fuentes_ml.py` | Ablación de fuentes de variables en los modelos clásicos. |

## Nota de baseline (con / sin filtración)

La cifra citable de los modelos clásicos es la **sin filtración (leakage)**: excluye las
cinco variables de crédito *otorgado* (monto, cuota, intereses, aval, instrumentación
digital), que son posteriores a la aprobación y no están disponibles al decidir. El
modelo de lenguaje tampoco las ve, de modo que la comparación es simétrica.

- **Sin filtración (citable):** XGBoost **0,7516**, Regresión Logística **0,6605**.
  Es el control `all_full` de `scripts/12_ablacion_fuentes_ml.py` y lo que reproduce el
  pipeline Kedro (`models/xgboost_metrics.json`, `models/logreg_metrics.json`).
- **Con filtración (solo transparencia):** XGBoost 0,7776, Regresión Logística 0,6910.
  Corresponde a la configuración previa (variables de otorgamiento incluidas), etiquetada
  `con_leakage_previo` en `models/baseline_metrics_noleak.json`. Se conserva como fixture
  y como variantes `xgb_leak`/`logreg_leak` del pipeline, y aparece en la tabla comparativa
  a título de contraste; **no es la cifra citable**.

## Pipeline Kedro (reproducibilidad)

El repositorio kedriza toda la evidencia siguiendo la organización canónica de Kedro
(cuatro pipelines: `data_processing`, `data_science`, `tabllm`, `reporting`; ver README).
`kedro run` reproduce offline los modelos clásicos, la tabla comparativa y las figuras;
`kedro run --pipeline tabllm` recomputa las métricas del modelo de lenguaje **desde los
caches de predicciones** (evidencia congelada) sin re-inferir. La integridad de esos
caches está registrada en `lineage/llm_cache_manifest.sha256`.

## Figuras del documento

**Cuerpo:**
- `figures/fig1_sesgo_wd81.png` — Fig 1: el sesgo de selección invierte la relación con el buró (wd81).
- `figures/fig2_margen_semantico.png` — Fig 2: el rubro del negocio (texto libre) separa la mora.
- `figures/fig_pipeline_ejemplo.png` — Fig 3: recorrido de un caso por el modelo de lenguaje.
- `figures/fig_interpretabilidad.png` — Fig 4: interpretabilidad de los modelos clásicos.

**Anexo (EDA, n = 4.897):**
- `figures/figA1_target.png` — A1: balance de la variable objetivo (56,1 % / 43,9 %).
- `figures/figA2_temporal.png` — A2: volumen mensual de desembolsos y mora por cohorte.
- `figures/figA4_faltantes.png` — A4: faltantes por variable (> 1 %).
- `figures/figA5_atipicos.png` — A5: distribución y atípicos de variables del negocio (escala symlog).

*Nota:* A3 (maduración de cohortes / vintage) no es construible con estos datos, que registran
el estado final del target pero no el momento del incumplimiento.

## Métricas (JSON versionables en `models/`)

Modo de razonamiento, protocolo sin filtración de información, test n = 495:
- `models/llm_metrics_thinking_{zero,few16,few32}_test.json` — Qwen3-8B (zero 0,569 / few16 0,503).
- `models/gemma4_12b_llm_metrics_thinking_{zero,few16}_test.json` — Gemma 4 12B (zero 0,318 / few16 0,443).
- `models/gpt54_gpt_metrics_{zero,few16}_test.json` — GPT-5.4 (zero 0,332 / few16 0,464).

## Excluido deliberadamente (política de datos, `.gitignore`)

- **Parquets de respuestas** (`models/gpt/*.parquet`, `models/gemma_fix/*.parquet`) — contienen la
  serialización textual de legajos de clientes; no se publican.
- **Logs de corrida** y binarios de modelos más allá de los baseline `.pkl` ya versionados.
- El dataset con datos de clientes permanece fuera del control de versiones público.
