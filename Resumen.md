# Resumen

## NLP de columnas textuales (`nbs/01b_nlp_texto.py`)

Notebook aparte del EDA, sin categorías temáticas manuales. Aprende vocabulario y bigramas con TF-IDF y descubre temas con NMF solo en train; elige resolución por estabilidad entre inicializaciones y guarda términos, textos representativos, deriva temporal y contraste externo con `subcategoria_texto`. Jupytext: `notebooks/01b_nlp_texto.ipynb`. Smoke y corrida full en 0.

Hallazgo: emergen 7 temas reproducibles, pero no forman una taxonomía fuerte (información mutua normalizada con la subcategoría: 0,139; pureza: 0,506). La mezcla temática cambia en prueba (Jensen-Shannon 0,197). La similitud mediana descripción–subcategoría es apenas 0,016. `tipo_credito` sigue siendo casi constante y `otra_categoria_negocio` es muy redundante.

Kedro de la entrega quedó en tres tramos: `intermediate_classics`, `intermediate_evaluate`, `intermediate_reporting`. `__default__` sigue siendo solo clásicos. Mismos nodos y nombres.

Guía: `.pi/GUIA-NOTEBOOKS.md`. Prompt: `.pi/PROMPT-SESION-AGENTES.md`.

## EDA humanizado en `01_eda.py`

Reescribí el notebook en chunks cortos: markdown conceptual + pandas lineal. El recorrido sigue la entrega §3.6 (target → texto → subcategoría → buró → correlaciones → extremos → esparso → faltantes → tiempo×historial). Mismas tablas; smoke y full en 0. Jupytext regeneró `notebooks/01_eda.ipynb` (26 celdas: 13 md + 13 code). Saqué `plt.close`: cada chunk deja `fig` o la tabla para que Jupyter las muestre. Esparso se calcula en el notebook (20 TU `< 0`, corte 6). Sumé chunks que terminan en la tabla: edad/antigüedad, dependientes, montos, arriendo, rubro, objetivo, canal×set, género. Los markdown del EDA son preguntas de cero (sin citar entrega ni otros scripts). Jupytext regeneró el `.ipynb`.


## Oleada 2 (Pi CLI)

Quedaron `06_tabpfn`, `07_gpt`, `08_sft_qlora`, `10_lendingclub_prep` y `11_lendingclub_benchmark`. Smoke OK salvo GPT: el mock no tenía `.completions`; lo envolví en `ChatClient`.

## Oleada 1 (Pi CLI)

`pi -p` escribió y el smoke pasó en `02_hipotesis`, `04_xgboost` y `05_qwen_prompt`. En `03_logreg` corregí el import de `calibration_curve` y `keep_empty_features=True` (en smoke `rle904` viene toda vacía).

## Validación de `01_eda.py`

**PASA** el smoke. El subagente de Pi escribió un EDA usable.

- Jupytext percent, sin `print`/`display`, cita plan §2.3 y entrega §3.5–3.6.
- Asociaciones y extremos solo en `train`. 29 variables del contrato.
- `TESIS_SMOKE=1 poetry run python notebooks/01_eda.py` termina en 0 y genera 9 CSV + 6 figuras.
- Falló el import (`from notebooks import _runtime` no anda como script). Lo corregí con `sys.path` + `notebooks/__init__.py`. El resto de agentes ya tienen esa regla.
- En smoke (80 filas) `mora_por_subcategoria_train` queda vacía: el corte de 40 créditos por rubro no se cumple. En corrida full no debería pasar.

## Cómo lanzo los agentes desde acá

Sí se pueden correr sin abrir la TUI:

```text
cd /Users/federicomoreno/Documents/TESIS/credit-risk-frontier
pi -p --approve --no-session "Usá subagent_run con agent nb-hipotesis en mode task. Task: Escribí notebooks/02_hipotesis.py. Importá _runtime con sys.path. No toques otro archivo."
```

Oleadas: mismos textos que abajo, con `pi -p --approve`.

## Prompts en la TUI (`pi` + `/reload`)

### Oleada 1 (sin eda, ya está)

```text
Usá subagent_run en mode background con agents:
nb-hipotesis, nb-logreg, nb-xgboost, nb-qwen-prompt
Task: Cada uno escribe SOLO su .py. Import: sys.path + from notebooks import _runtime. No toquen 01_eda.py.
```

### Oleada 2

```text
Usá subagent_run en mode background con agents:
nb-tabpfn, nb-gpt, nb-sft, nb-lendingclub-prep, nb-lendingclub-bench
Task: Cada uno escribe SOLO su .py asignado. Lean los archivos de su rol. No se pisen.
```

### Comparación y smoke

```text
Usá subagent_run con agent nb-comparacion en mode task.
Task: Escribí notebooks/09_comparacion.py. Si faltan salidas, usá stubs del smoke.
```

```text
Usá subagent_run con agent nb-smoke-runner en mode task.
Task: Corré TESIS_SMOKE=1 sobre los notebooks que existan. Falla si hay print/display o si el import no corre como script.
```
