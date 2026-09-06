# Credit Risk Frontier — Tesis UBA 2026

Repositorio de la tesis de Federico Nicolás Moreno (Maestría en Explotación de
Datos y Descubrimiento de Conocimiento, FCEyN-UBA). Pregunta de investigación
(PLAN §2.1): ¿puede un LLM open-source compacto con *thinking mode* nativo
alcanzar o superar a los métodos clásicos de ML en riesgo crediticio de
microfinanzas latinoamericanas?

El documento rector es
[`entregas/PLAN de tesis Federico Moreno.md`](entregas/PLAN%20de%20tesis%20Federico%20Moreno.md).
Comandos y reglas del experimento: [`pipeline/README.md`](pipeline/README.md).

## Flujo: `pipeline/` + `src/`

```
src/credit_risk_frontier/        dominio: 29 variables, métricas, serialización, cohorte
pipeline/contrato.py             rutas, split, modelos, costos
pipeline/monitoreo.py            métricas, matrices, costos, PSI
pipeline/01_preprocesamiento/    crudo -> cohorte + variables + texto humanizado
pipeline/02_regresion_logistica/ modelo #1
pipeline/03_xgboost/             modelo #2
pipeline/04_tabulares/           modelo #3 (TabPFN)
pipeline/05_qwen/                modelos #5-7
pipeline/06_gpt/                 modelo #4
pipeline/07_finetuning/          modelo #8 (QLoRA, pendiente)
pipeline/08_lendingclub/         dataset 2 (Feng et al. 2023)
```

```bash
poetry install
poetry run python pipeline/01_preprocesamiento/01_cargar_datos.py
poetry run python pipeline/01_preprocesamiento/02_crear_variables.py
poetry run python pipeline/01_preprocesamiento/03_humanizar.py
poetry run python pipeline/02_regresion_logistica/entrenar_y_predecir.py
poetry run python pipeline/03_xgboost/entrenar_optuna.py --trials 50
poetry run python pipeline/04_tabulares/predecir.py
poetry run python pipeline/05_qwen/predecir.py
poetry run python pipeline/06_gpt/predecir.py
poetry run python pipeline/monitoreo.py
```

La partición temporal se declara una vez (paso 01, columna `set`). Nadie
vuelve a particionar.

## Modelos (PLAN §2.2)

| # | Modelo | Dónde | Estado |
|---|--------|-------|--------|
| 1 | Regresión logística | `pipeline/02_regresion_logistica/entrenar_y_predecir.py` | corrido |
| 2 | XGBoost | `pipeline/03_xgboost/entrenar_optuna.py` | corrido |
| 3 | TabPFN v2 | `pipeline/04_tabulares/predecir.py` | corrido |
| 4 | GPT zero/few-shot | `pipeline/06_gpt/predecir.py` | corrido (API) |
| 5-7 | Qwen3 zero/few/thinking | `pipeline/05_qwen/predecir.py` | corrido |
| 8 | Qwen3 fine-tuned (QLoRA) | `pipeline/07_finetuning/` | pendiente |

Razonamientos por caso: `data/pipeline/razonamientos/*.jsonl`. Métricas:
`data/pipeline/monitoreo/`.

La entrega intermedia usó split 80/10/10 y no conservó respuestas de Qwen.
Este pipeline usa 70/15/15 y guarda thinking: las cifras no se mezclan. Ver
[`entregas/ENTREGA-INTERMEDIA-G1-MORENO-FEDERICO-2026.md`](entregas/ENTREGA-INTERMEDIA-G1-MORENO-FEDERICO-2026.md).

## Datasets (PLAN §2.3)

| Dataset | Rol | Dónde |
|---|---|---|
| Fintech LATAM | 4.201 créditos modelables, mora >60d en 150d | `data/crudo/` |
| LendingClub | benchmark vs. Feng et al. (2023) | `pipeline/08_lendingclub/` |

Partición vigente: temporal 70/15/15 (train 2.940 / val 630 / test 631),
congelada en `data/pipeline/01_manifiesto_particion.json`.

## Mapa

| Ruta | Qué es |
|---|---|
| `entregas/` | PLAN y entrega intermedia |
| `docs/` | decisiones e inventario de variables |
| `src/credit_risk_frontier/` | dominio compartido |
| `tests/` | contrato del universo crudo y del split congelado |
| `redaccion/` | material de tesis |
| `bibliografia/` | `references.bib` |

## Privacidad

El extracto original con PII (`data/00_original/`) es solo local. Los datos
anonimizados, predicciones y razonamientos de `data/pipeline/` se versionan
para reproducir el experimento; los razonamientos no se republican como
corpus.
