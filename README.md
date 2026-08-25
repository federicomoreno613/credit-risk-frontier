# Credit Risk Frontier — Tesis UBA 2026

Repositorio de la tesis de Federico Nicolás Moreno (Maestría en Explotación de
Datos y Descubrimiento de Conocimiento, FCEyN-UBA). Pregunta de investigación
(PLAN §2.1): ¿puede un LLM open-source compacto con *thinking mode* nativo
alcanzar o superar a los métodos clásicos de ML en riesgo crediticio de
microfinanzas latinoamericanas?

El repositorio sigue la estructura del documento rector:
[`entregas/PLAN de tesis Federico Moreno.md`](entregas/PLAN%20de%20tesis%20Federico%20Moreno.md).

## Flujo de ejecución: `pipeline/`

El experimento se corre por carpetas, con un contrato único. Detalle y
comandos en [`pipeline/README.md`](pipeline/README.md):

```
pipeline/contrato.py             declaración única: variables, target, split, modelos, costos
pipeline/monitoreo.py            módulo común: métricas, matrices de confusión, costos, PSI
pipeline/01_preprocesamiento/    data original -> cohorte + variables + texto humanizado
pipeline/02_regresion_logistica/ modelo #1        pipeline/06_gpt/         modelo #4
pipeline/03_xgboost/             modelo #2        pipeline/07_finetuning/  modelo #8 (pend.)
pipeline/04_tabulares/           modelo #3 (pend.) pipeline/08_lendingclub/ dataset 2 (pend.)
pipeline/05_qwen/                modelos #5-7
```

```bash
poetry install
for paso in pipeline/0*.py; do poetry run python "$paso"; done
```

Regla central: la partición temporal se declara UNA vez (paso 01, columna
`set`) y todos los modelos comparten exactamente los mismos train/val/test.

## Diseño experimental (PLAN §2.2): los 8 modelos y dónde viven

| # | Modelo | Dónde | Estado |
|---|--------|-------|--------|
| 1 | Regresión logística | `pipeline/04_entrenar.py` | corrido (AUC val 0.758) |
| 2 | XGBoost | `pipeline/04_entrenar.py` | corrido (AUC val 0.759); Optuna 500 trials pendiente |
| 3 | TabPFN v2 | pendiente — esbozo en `nbs/06_tabpfn.py` | pendiente |
| 4 | GPT zero/few-shot (API) | `pipeline/05_predecir.py --gpt` | listo, requiere `openai` + API key |
| 5 | Qwen3 zero-shot | `pipeline/05_predecir.py --qwen --shots 0` | verificado en muestra |
| 6 | Qwen3 + thinking | mismo comando; el *thinking* se guarda SIEMPRE | verificado en muestra |
| 7 | Qwen3 few-shot (8 y 16) | `pipeline/05_predecir.py --qwen --shots 8` (o `16`) | listo |
| 8 | Qwen3 fine-tuned (QLoRA) | pendiente — esbozo en `nbs/08_sft_qlora.py` | pendiente |

### Razonamientos guardados (insumo de §2.4 Interpretabilidad)

Cada inferencia LLM persiste su razonamiento completo como texto en
`data/pipeline/05_razonamientos/*.jsonl` (un registro por crédito: thinking de
Qwen o reasoning de GPT, respuesta y probabilidad). Ese corpus alimenta la
comparación explicaciones-en-lenguaje-natural vs. SHAP prevista en el plan y
el análisis de calidad de prompts. Contiene datos de casos: NO se publica.

### Experimento previo de referencia (entrega intermedia)

El último experimento cerrado es
[`entregas/ENTREGA-INTERMEDIA-G1-MORENO-FEDERICO-2026.md`](entregas/ENTREGA-INTERMEDIA-G1-MORENO-FEDERICO-2026.md):
split 80/10/10 (3.360/420/421), 6 configuraciones, resultados congelados en la
prueba temporal — XGBoost 0,761, LogReg 0,757, mejor Qwen 0,654; ΔAUC de la
descripción negativo y few-8 sin mejora. Diferencias con el pipeline vigente:

- **Split**: la entrega usó 80/10/10; el pipeline usa 70/15/15 (PLAN §2.3).
  Las cifras de ambos experimentos NO se mezclan.
- **Razonamientos**: la entrega (§3.11) no conservó las respuestas de Qwen;
  el pipeline guarda thinking/reasoning completos por caso.
- **Validación cruzada**: corrido con el split viejo, el pipeline reprodujo la
  LogReg publicada (AUC test 0,757) y quedó cerca en XGBoost (0,771 vs. 0,761,
  distinta búsqueda de hiperparámetros).

## Datasets (PLAN §2.3)

| Dataset | Rol | Dónde |
|---|---|---|
| Fintech LATAM (caso de estudio) | principal: 4.201 créditos modelables, mora >60d en 150d | `data/01_raw/` (local, fuera de Git) |
| LendingClub (comparativo público) | benchmark vs. Feng et al. (2023) | `data/lendingclub/`, `nbs/10–11` |

Partición vigente del caso de estudio: temporal 70/15/15
(train 2.940 / val 630 / test 631), alineada con el PLAN §2.3 y congelada en
`data/pipeline/01_manifiesto_particion.json`. Decisión 2026-08: few-shot con
8 y 16 ejemplos (32 se descartó por costo/beneficio).

## Métricas (PLAN §2.4)

Discriminación (AUC/Gini/KS), calibración (Brier/ECE), interpretabilidad
(SHAP vs. razonamientos), costo computacional y estabilidad. El paso 06 las
materializa en `data/pipeline/06_monitoreo/` (métricas por segmento
esparso/denso, PSI train vs. test, tasa de mora por partición).

## Extensiones declaradas (no priorizadas)

- **Foundation model tabular**: TabPFN v2 y/o
  [TabFM](https://github.com/google-research/tabfm) (Google Research,
  in-context, API scikit-learn, pesos con licencia no comercial) como
  candidatos para el modelo #3 del PLAN. TabFM no es de series de tiempo.
- **Series de tiempo preentrenado** (Chronos/TimesFM) sobre la secuencia de
  pagos por crédito, con la misma partición del contrato. Ver
  `pipeline/contrato.py`.


## Mapa del resto del repositorio

| Ruta | Qué es |
|---|---|
| `entregas/` | PLAN de tesis y entrega intermedia (fuente de cifras publicadas) |
| `nbs/` + `notebooks/` | notebooks de análisis y redacción (locales, fuera de Git) |
| `bibliografia/` | papers citados y `references.bib` |
| `docs/` | registro de decisiones e inventario de variables |
| `src/credit_risk_frontier/` | dominio compartido: `utils.py` (variables, serialización, métricas, Ollama) y `cohorte.py` (desenlace y partición) |
| `tests/` | contrato de datos del universo crudo |

## Privacidad

Los datos crudos con PII, los textos completos y los razonamientos por caso
son locales y están excluidos de Git. Solo se publican agregados y contratos.
