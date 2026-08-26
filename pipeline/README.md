# Pipeline por carpetas

Una carpeta por etapa/modelo, un contrato único y un módulo común de
monitoreo. La partición temporal train/val/test (70/15/15) se declara UNA vez
en `contrato.py`, se materializa en el preprocesamiento (columna `set`) y
todos los modelos la comparten.

```
contrato.py               declaración única: rutas, 29 variables, target, split, modelos, costos
monitoreo.py              módulo COMÚN (métricas, PSI, matrices de confusión, costos) + consolidación
01_preprocesamiento/      carga de datos -> variables -> humanizado (texto para LLMs)
02_regresion_logistica/   modelo #1 del PLAN — baseline
03_xgboost/               modelo #2 — búsqueda Optuna (varias pruebas)
04_tabulares/             modelo #3 — TabPFN/TabFM (pendiente)
05_qwen/                  modelos #5-7 — local, educativo, guarda el thinking
06_gpt/                   modelo #4 — API, guarda el reasoning
07_finetuning/            modelo #8 — QLoRA (pendiente)
08_lendingclub/           dataset 2 del PLAN — subset de Feng et al. 2023 vía
                          Hugging Face (TheFinAI/cra-lendingclub, sin Kaggle);
                          SOLO Qwen zero/few-shot + tests de contaminación
                          (Bordt et al. 2024) antes de interpretar resultados
```

## Correr

```bash
# 1. Preprocesamiento (una vez; regenerar solo si cambia el contrato)
poetry run python pipeline/01_preprocesamiento/01_cargar_datos.py
poetry run python pipeline/01_preprocesamiento/02_crear_variables.py
poetry run python pipeline/01_preprocesamiento/03_humanizar.py

# 2. Modelos (independientes entre sí)
poetry run python pipeline/02_regresion_logistica/entrenar_y_predecir.py
poetry run python pipeline/03_xgboost/entrenar_optuna.py --trials 50
poetry run python pipeline/05_qwen/predecir.py --demo        # ver prompt+thinking
poetry run python pipeline/05_qwen/predecir.py               # 6 configs, reanudable
poetry run python pipeline/06_gpt/predecir.py --limite 50    # requiere openai + key

# 3. Monitoreo común: consolida todo y mide
poetry run python pipeline/monitoreo.py

# 4. LendingClub (independiente del resto; solo Qwen)
poetry run python pipeline/08_lendingclub/01_descargar.py
poetry run python pipeline/08_lendingclub/02_contaminacion.py
poetry run python pipeline/08_lendingclub/03_qwen_eval.py --limite 500
```

## Reglas

- Nadie re-particiona: el split vive en `set` desde el preprocesamiento.
- El test solo se usa para predecir/medir; los hiperparámetros se eligen en val.
- Cada modelo escribe `data/pipeline/predicciones/<nombre>.parquet` con el
  esquema `C.COLUMNAS_PREDICCION` (vía `C.guardar_predicciones`).
- Los LLM leen el MISMO texto humanizado y guardan razonamiento completo por
  caso en `data/pipeline/razonamientos/*.jsonl` (reanudable; PII, no se publica).
- Las métricas por umbral (accuracy, precision, recall, matriz de confusión)
  se reportan a umbral 0,5 y al umbral óptimo por costos (`C.COSTOS`);
  AUC/Gini/KS no dependen del umbral.
- Si cambia el contrato: borrar los JSONL de razonamientos y regenerar todo.

## Salidas (`data/pipeline/`)

```
01_base.parquet, 01_manifiesto_particion.json    cohorte y split congelado
02_variables.parquet, 02_contrato_variables.json
03_humanizado.parquet
modelos/            artefactos (.joblib, trials Optuna, info JSON)
predicciones/       un parquet por modelo
razonamientos/      JSONL con thinking (Qwen) / reasoning (GPT)
predicciones_consolidadas.parquet
monitoreo/          metricas.csv, matrices_confusion.csv, psi_variables.csv, resumen.json
```
