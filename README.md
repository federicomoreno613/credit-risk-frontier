# Credit Risk Frontier — Entrega III

Repositorio de apoyo para la tesis de Maestría en Data Mining & Knowledge Discovery (FCEyN, UBA): **comparación de modelos clásicos de machine learning y LLMs para credit scoring en microfinanzas latinoamericanas**.

**Autor:** Federico Nicolás Moreno  
**Director:** Mgs. Boris Dorian Da Silva  
**Co-director:** Dr. Cristian Bravo  
**Año:** 2026

## Qué contiene esta versión

Esta carpeta acompaña la Entrega III. Está pensada como evidencia reproducible del análisis. Se publica un **CSV base anonimizado** (`data/dataset_tesis.csv`) porque es el insumo mínimo para reproducir los notebooks; quedan fuera los datos crudos, las claves de anonimización y los caches de inferencia por crédito.

Estructura curada:

```text
credit-risk-frontier/
├── data/dataset_tesis.csv   # base anonimizada usada por scripts/notebooks
├── notebooks/               # notebooks ejecutables de la entrega
├── scripts/                 # scripts equivalentes, pensados para pipeline
├── results/                 # CSVs resumidos de métricas y perfiles
├── models/                  # modelos clásicos y métricas JSON curadas
├── figures/                 # figuras finales incluidas en el documento
├── bibliografia/references.bib
├── MANIFEST.md              # qué se sube y qué se excluye
├── requirements.txt
└── pyproject.toml
```

## Notebooks principales

| Notebook | Rol en la entrega |
|---|---|
| `notebooks/03_baseline_xgboost.ipynb` | Entrena y evalúa XGBoost con split temporal. |
| `notebooks/04_baseline_logreg.ipynb` | Baseline lineal interpretable. |
| `notebooks/05_segmentacion_thin_file.ipynb` | Segmentación por densidad de buró y CV temporal. |
| `notebooks/06b_llm_prompting.ipynb` | Experimento Qwen3-8B zero-shot con serialización TabLLM. |
| `notebooks/08_llm_fewshot.ipynb` | Experimentos few-shot con 8/16/32 ejemplos. |
| `notebooks/07_comparacion_final.ipynb` | Consolida métricas y comparación final. |

## Reproducibilidad

Los scripts en `scripts/` reflejan la misma lógica de los notebooks y exponen funciones reutilizables. El flujo conceptual es:

1. preparar dataset modelable;
2. entrenar baselines clásicos (`03`, `04`);
3. evaluar por segmento con `TimeSeriesSplit` (`05`);
4. correr inferencia LLM local con Ollama (`06b`, `08`);
5. consolidar métricas y figuras (`07`).

Ejemplo de instalación del entorno:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

La inferencia LLM requiere Ollama local y el modelo `qwen3:8b` descargado. Las celdas de inferencia LLM que dependen de caches locales pueden tardar mucho o requerir Ollama; los caches por crédito no se publican.

## Decisiones metodológicas visibles en la entrega

- El split train/validación/test respeta el orden temporal de `fecha_desembolso`.
- La comparación principal reporta AUC de test temporal.
- El desglose por segmento clásico usa `TimeSeriesSplit`, no `StratifiedKFold(shuffle=True)`, para evitar *data leakage* temporal.
- El LLM usa serialización TabLLM con variables textuales del negocio y top features numéricas.
- El resultado few-shot del segmento *thin-file* se reporta con cautela porque el test esparso tiene n = 28.

## Qué no se publica

No subir datos crudos (`data/01_raw/`, `data/03_primary/`), `data/00_keys/`, predicciones `.parquet`, configuraciones de asistentes (`CLAUDE.md`, `.claude/`) ni PDFs bibliográficos completos si el repositorio va a ser público. Los modelos clásicos curados (`models/xgboost_baseline.*`, `models/logreg_baseline.pkl`) sí quedan como evidencia reproducible.
