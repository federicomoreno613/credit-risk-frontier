# Credit Risk Frontier — Tesis UBA 2026

Repositorio de trabajo para mi tesis de Maestría en Data Mining & Knowledge Discovery (FCEyN, UBA): **comparación de modelos clásicos de machine learning y modelos de lenguaje para predicción de riesgo crediticio en microfinanzas latinoamericanas**.

**Autor:** Federico Nicolás Moreno  
**Director:** Mgs. Boris Dorian Da Silva  
**Co-director:** Dr. Cristian Bravo  
**Año:** 2026

## Resumen

La pregunta central es simple: en una cartera real de microcréditos, ¿un modelo de lenguaje puede competir con modelos clásicos de *credit scoring* como XGBoost o Regresión Logística?

La respuesta que aparece en esta etapa es matizada:

- **XGBoost gana claramente en el promedio**, porque aprovecha mejor la señal tabular del buró y de las variables internas.
- **El LLM zero-shot no alcanza**, porque llega con reglas generales sobre riesgo crediticio que no siempre aplican a una cartera ya filtrada por aprobación.
- **El LLM few-shot muestra una señal interesante en clientes con buró esparso**, es decir, casos donde TransUnion aporta menos profundidad histórica y el texto del negocio puede pesar más.

Este repositorio guarda la evidencia reproducible de esa comparación: dataset anonimizado, notebooks, scripts, resultados, figuras, modelos clásicos curados y el documento final integrado.

## Estado actual del repo

El repo ya tiene una primera capa Kedro real para ordenar la versión de investigación: mantiene notebooks + scripts + resultados curados, pero ahora agrega pipelines reproducibles para validar el dataset, auditar EDA, orquestar modelos clásicos, segmentación, comparación y chequeos de artefactos de tesis.

La kedrización v1 prioriza **no romper la evidencia publicada**. Por defecto reutiliza métricas y modelos ya curados en `models/` y `results/`; si se quiere recalcular, los parámetros permiten activar entrenamiento o inferencia local de forma explícita.

## Qué contiene

```text
credit-risk-frontier/
├── docs/                    # documento final integrado en MD y DOCX
├── data/dataset_tesis.csv   # base anonimizada usada por scripts/notebooks
├── notebooks/               # notebooks de exploración, modelos y comparación
├── scripts/                 # versiones ejecutables del flujo experimental
├── src/credit_risk_frontier/ # pipelines Kedro v1
├── conf/base/               # catálogo y parámetros Kedro
├── tests/                   # checks de contrato del dataset y buró esparso
├── results/                 # métricas y tablas finales en CSV
├── models/                  # modelos clásicos curados y métricas JSON
├── figures/                 # figuras usadas en el documento final
├── lineage/                 # scripts/logs para auditar cifras clave
├── bibliografia/references.bib
├── MANIFEST.md              # detalle de qué se sube y qué queda fuera
├── requirements.txt
└── pyproject.toml
```

## Documento final

La versión integrada de la entrega está en:

- `docs/integrado.md`
- `docs/integrado.docx`

El Markdown usa rutas relativas a `figures/`, así que puede leerse desde GitHub sin depender de rutas locales de mi máquina.

## Flujo experimental

Los notebooks principales son:

| Notebook | Rol |
|---|---|
| `notebooks/03_baseline_xgboost.ipynb` | Entrena y evalúa XGBoost con split temporal. |
| `notebooks/04_baseline_logreg.ipynb` | Baseline lineal interpretable. |
| `notebooks/05_segmentacion_thin_file.ipynb` | Segmentación por buró esparso y CV temporal. |
| `notebooks/06b_llm_prompting.ipynb` | Qwen3-8B zero-shot con serialización tipo TabLLM. |
| `notebooks/08_llm_fewshot.ipynb` | Experimentos few-shot con 8, 16 y 32 ejemplos. |
| `notebooks/07_comparacion_final.ipynb` | Consolidación de métricas y figuras finales. |

Los scripts en `scripts/` siguen la misma lógica de los notebooks, pero son más fáciles de correr como pipeline simple.

## Resultados principales

Sobre el test temporal más reciente:

| Modelo | AUC test | Lectura |
|---|---:|---|
| XGBoost | 0,820 | Mejor desempeño global. |
| Regresión Logística | 0,705 | Baseline lineal razonable, pero inferior. |
| Qwen3-8B zero-shot | 0,529 | Cerca del azar; falla por priors generales. |
| Qwen3-8B few-shot 16 | 0,527 | No mejora el agregado, pero sí muestra señal por segmento. |

El resultado más interesante aparece al mirar clientes con **buró esparso**: con 16 ejemplos en contexto, Qwen3-8B alcanza **AUC = 0,738** en ese segmento. Es un hallazgo prometedor, pero debe leerse con cautela porque el test esparso tiene solo **n = 28** casos.

## Decisiones metodológicas importantes

- El split train/validación/test respeta `fecha_desembolso`; no se usa split aleatorio para la comparación principal.
- La validación por segmento usa `TimeSeriesSplit`, no `StratifiedKFold(shuffle=True)`, para evitar *data leakage* temporal.
- Los códigos `-1` del buró no se tratan como nulos comunes: representan ausencia de registros específicos y pueden ser informativos.
- En el texto final uso **buró esparso** en vez de “thin-file” porque describe mejor el problema práctico: no faltan todos los datos, falta profundidad de historial crediticio formal.
- La inferencia LLM requiere Ollama local y el modelo `qwen3:8b`; los caches por crédito no se publican.

## Lineaje y auditoría de cifras

La carpeta `lineage/` contiene scripts y logs cortos que sirven para auditar los números principales del documento:

- `lineage/eda_lineage.py`
- `lineage/ml_lineage.py`
- `lineage/llm_lineage.py`
- `lineage/logs/*.log`

No reemplazan a los notebooks originales, pero ayudan a seguir el razonamiento: qué dataset se cargó, qué corte se aplicó, qué métrica salió y qué figura se generó.

## Cómo reproducir el entorno base

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Luego, el flujo conceptual es:

1. preparar dataset modelable;
2. entrenar modelos clásicos;
3. evaluar por segmento con split temporal;
4. correr inferencia LLM local con Ollama;
5. consolidar métricas y figuras;
6. contrastar cifras con `lineage/`.

## Kedro v1

La primera versión Kedro separa el flujo en pipelines nombrados:

| Pipeline | Rol |
|---|---|
| `public_repro` | Validación del dataset, EDA liviano, métricas clásicas, segmentación, comparación y chequeo de artefactos. Es el default seguro. |
| `classic_models` | XGBoost y Regresión Logística. Por defecto reutiliza artefactos existentes; puede recalcularse por parámetros. |
| `segment_analysis` | Validación temporal por clientes con buró esparso/denso. |
| `llm_local` | Zero-shot y few-shot con Ollama/cache local. Queda fuera del default. |
| `full_local` | Todo lo anterior, incluyendo LLM local si se activan los parámetros. |

Comandos básicos:

```bash
kedro registry list
kedro catalog describe-datasets --pipeline public_repro
kedro run --pipeline public_repro
pytest -q
```

Para recalcular modelos clásicos o LLM local, cambiar parámetros en `conf/base/parameters_data_science.yml` o usar overrides de Kedro. La idea es que `kedro run` no dispare Ollama ni entrenamientos largos por accidente.

Los tests de `pytest -q` no sólo chequean funciones sueltas: también corren `public_repro` y `llm_local` de punta a punta y comparan los outputs Kedro contra los artefactos curados del repo normal (`models/` y `results/`).

La etapa siguiente de tesis no es volver a rehacer esta entrega, sino extender sobre esta base: dataset público/control, TabPFN, LLM frontier/cloud, thinking on/off, QLoRA, calibración, interpretabilidad y costos.

## Qué no se publica

Por privacidad y trazabilidad, quedan fuera:

- datos crudos con PII;
- claves de anonimización;
- caches de predicciones por crédito;
- PDFs bibliográficos completos;
- configuraciones locales de asistentes o herramientas.

El detalle exacto está en `MANIFEST.md`.
