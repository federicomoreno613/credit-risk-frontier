# MANIFEST — Entrega III

Este manifiesto define el recorte del repositorio `credit-risk-frontier` que acompaña la Entrega III de la tesis. El criterio es simple: subir evidencia reproducible del trabajo humano —dataset base anonimizado, notebooks, scripts, modelos clásicos, métricas, figuras y documentación— sin publicar datos crudos, claves ni rastros de herramientas de asistencia.

## Subir

| Ruta | Motivo |
|---|---|
| `README.md` | Explica el objetivo del repo, el flujo y las limitaciones de reproducibilidad. |
| `MANIFEST.md` | Deja explícito qué se publica y qué queda fuera. |
| `requirements.txt` / `pyproject.toml` | Permiten reconstruir el ambiente base. |
| `data/dataset_tesis.csv` | CSV base anonimizado usado por los notebooks y scripts de E3. |
| `results/*.csv` | Tablas de métricas y perfil del dataset, livianas y auditables. |
| `models/xgboost_baseline.json` / `models/xgboost_baseline.pkl` | Modelo XGBoost final curado. |
| `models/logreg_baseline.pkl` | Baseline lineal final curado. |
| `models/*metrics*.json` | Métricas reproducibles sin predicciones por cliente. |
| `notebooks/03_baseline_xgboost.ipynb` | Baseline principal y métrica honesta de test. |
| `notebooks/04_baseline_logreg.ipynb` | Baseline lineal interpretable. |
| `notebooks/05_segmentacion_thin_file.ipynb` | Segmentación *thin-file* y corrección de leakage temporal. |
| `notebooks/06b_llm_prompting.ipynb` | Qwen3-8B zero-shot con serialización TabLLM. |
| `notebooks/08_llm_fewshot.ipynb` | Few-shot 8/16/32 y hallazgo del segmento esparso. |
| `notebooks/07_comparacion_final.ipynb` | Tabla final y figuras comparativas. |
| `scripts/03_baseline_xgboost.py` | Versión ejecutable del baseline XGBoost. |
| `scripts/04_baseline_logreg.py` | Versión ejecutable de regresión logística. |
| `scripts/05_segmentacion_thin_file.py` | CV temporal por segmento. |
| `scripts/06b_llm_prompting.py` | Inferencia LLM zero-shot. |
| `scripts/08_llm_fewshot.py` | Inferencia LLM few-shot. |
| `scripts/07_comparacion_final.py` | Consolidación de métricas. |
| `figures/*.png` | Figuras y capturas terminal usadas en el documento. |
| `bibliografia/references.bib` | Referencias bibliográficas sin redistribuir PDFs completos. |

## No subir

| Ruta / patrón | Motivo |
|---|---|
| `data/01_raw/`, `data/03_primary/`, `data/04_model_input/` | Datos crudos/intermedios y derivados no curados. |
| `data/00_keys/` | Mapeos de anonimización reversibles. |
| `data/07_model_output/` | Caches y salidas por crédito. |
| `*.parquet`, `*.joblib`, `*.pickle` | Artefactos derivados con posible información sensible o peso innecesario. |
| `*.pkl` salvo `models/xgboost_baseline.pkl` y `models/logreg_baseline.pkl` | Solo se publican los modelos clásicos curados. |
| `CLAUDE.md`, `.claude/` | Configuración de asistentes; no aporta a la entrega académica. |
| `bibliografia/*.pdf` | PDFs completos: mejor citar vía `references.bib`, salvo repositorio privado autorizado. |
| notebooks con rutas locales o warnings largos | Limpiar outputs antes de publicar si contienen `/Users/...`, trazas o HTML pesado. |

## Comando sugerido antes de publicar

```bash
git status --short
python -m nbstripout notebooks/*.ipynb  # opcional: si se decide limpiar outputs
```

Si ya hubiera datos o PDFs trackeados por Git, quitarlos del índice sin borrar los archivos locales:

```bash
git rm --cached data/01_raw/*.csv data/03_primary/*.csv bibliografia/*.pdf
```
