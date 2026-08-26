# Mapa de la tesis por unidades

Cada fila es una sesión de escritura (ver método en `README.md`).
Estados: `pendiente` → `ficha` (IA preparó evidencia) → `borrador` (Federico
escribió) → `revisada` → `cerrada`. Las citas son claves de
`bibliografia/references.bib`.

## Cap. 1 — Introducción y pregunta de investigación

| Unidad | Evidencia / insumo | Figuras | Citas | Estado |
|---|---|---|---|---|
| 1.1 Contexto: microfinanzas y exclusión thin-file | `entregas/PLAN de tesis…` §1; E3 | — | demirguc2025, cornelli2022, chioda2025 | pendiente |
| 1.2 Pregunta e hipótesis (LLM vs ML clásico) | PLAN §2.1 | — | feng2023, hegselmann2023 | pendiente |
| 1.3 Contribuciones y alcance | PLAN; `docs/DECISIONES_EXPERIMENTO_FINAL.md` | — | — | pendiente |

## Cap. 2 — Marco teórico y trabajos relacionados

| Unidad | Evidencia / insumo | Figuras | Citas | Estado |
|---|---|---|---|---|
| 2.1 Credit scoring clásico y regulación | bibliografía | — | thomas2002, siddiqi2017, bcbs2006, hand1997, crook2007 | pendiente |
| 2.2 Benchmarks tabulares: árboles vs deep | — | — | lessmann2015, grinsztajn2022, chen2016, hollmann2025 | pendiente |
| 2.3 LLMs para datos tabulares y scoring | — | — | hegselmann2023, feng2023, fang2024, qwen2025, qwen2026 | pendiente |
| 2.4 Calibración y decisiones con costos | — | — | zadrozny2002, niculescu2005, bellotti2009 | pendiente |
| 2.5 Contaminación de preentrenamiento | `pipeline/08_lendingclub/02_contaminacion.py` (metodología) | — | bordt2024 | pendiente |

## Cap. 3 — Datos

| Unidad | Evidencia / insumo | Figuras | Citas | Estado |
|---|---|---|---|---|
| 3.1 Cohorte, target y partición temporal | `data/pipeline/01_manifiesto_particion.json`; contrato | `01_eda_target_temporal.png` | — | pendiente |
| 3.2 Las 29 variables (TU + formulario) y faltantes | `02_contrato_variables.json` | `03_eda_faltantes_estructuradas.png`, `04_eda_asociaciones_train.png` | cerda2020 | pendiente |
| 3.3 Segmentos esparso vs denso (thin-file) | contrato (`CORTE_ESPARSO`) | `05_eda_segmentos_buro.png`, `06_eda_bivariado_buro.png`, `07_eda_mora_set_segmento.png` | — | pendiente |
| 3.4 Texto libre: qué dice la descripción del negocio | nb `01b_nlp_texto` | `02_eda_campos_texto.png`, `02b_eda_palabras_descripcion.png`, `08_nlp_senales_campos.png`, `08b_nlp_temas_emergentes.png` | — | pendiente |
| 3.5 Sesgo de selección (solo aprobados) y límites | E3 feedback; DECISIONES | — | bolton2002, campbell2015 | pendiente |

## Cap. 4 — Metodología

| Unidad | Evidencia / insumo | Figuras | Citas | Estado |
|---|---|---|---|---|
| 4.1 Diseño experimental: 8 modelos, mismos splits | `pipeline/README.md`; contrato | diagrama a crear | — | pendiente |
| 4.2 Baselines: logreg y XGBoost+Optuna (+costos) | `pipeline/02…/03…` | `03_logreg_roc.png`, `03_logreg_calibracion.png` | akiba2019, chen2016 | pendiente |
| 4.3 TabPFN | `pipeline/04_tabulares/` | — | hollmann2025 | pendiente |
| 4.4 LLMs: serialización, prompts, zero/few-shot, thinking | `pipeline/05_qwen/`, `06_gpt/`; `utils.py` | ejemplo de prompt (caja) | qwen2025, hegselmann2023, ouyang2022 | pendiente |
| 4.5 QLoRA | `pipeline/07_finetuning/` | — | hu2022, dettmers2023 | pendiente |
| 4.6 Métricas, costos y calibración (pendiente isotónica) | `monitoreo.py`; DECISIONES | — | zadrozny2002, niculescu2005 | pendiente |

## Cap. 5 — Resultados

| Unidad | Evidencia / insumo | Figuras | Citas | Estado |
|---|---|---|---|---|
| 5.1 Tabla maestra: AUC/Gini/KS por modelo y segmento | `monitoreo/metricas.csv` | tabla + forest plot a crear | — | pendiente |
| 5.2 Tabulares: XGBoost vs TabPFN vs logreg | `metricas.csv`; nb12 | figura a crear | — | pendiente |
| 5.3 Qwen zero-shot por perfil; efecto del texto libre | `metricas.csv`; nb13 | figura a crear | — | pendiente |
| 5.4 Few-shot EMPEORA (0,67→0,58): diagnóstico | checkpoint 621ac0b; nb15 | figura a crear | hegselmann2023 | pendiente |
| 5.5 GPT vs Qwen; thinking no mejora (2x costo) | `metricas.csv`; commit 00d3407; nb14 | figura a crear | — | pendiente |
| 5.6 Thin-file: ¿el LLM gana en esparso? | `metricas.csv` (segmento) | figura clave de la tesis | — | pendiente |
| 5.7 Decisiones con costos: matrices y umbral óptimo | `matrices_confusion.csv` | figura a crear | bellotti2009 | pendiente |
| 5.8 QLoRA (cuando corra) | pendiente de corrida | — | hu2022 | pendiente |

## Cap. 6 — Validación externa: LendingClub

| Unidad | Evidencia / insumo | Figuras | Citas | Estado |
|---|---|---|---|---|
| 6.1 Por qué LendingClub y por qué SOLO Qwen | `pipeline/08_lendingclub/lc.py` (docstring); manifiesto | — | feng2023 | pendiente |
| 6.2 Tests de contaminación (3 tests + control) | `salidas/contaminacion.json` | figura brecha orig-control | bordt2024 | pendiente |
| 6.3 Zero/few-shot original vs perturbado vs Feng | `salidas/metricas_qwen.csv` | tabla comparativa | feng2023 | pendiente |

## Cap. 7 — Discusión y conclusiones

| Unidad | Evidencia / insumo | Figuras | Citas | Estado |
|---|---|---|---|---|
| 7.1 Respuesta a la pregunta; comparativa honesta | cap. 5 y 6 | — | lessmann2015, feng2023 | pendiente |
| 7.2 Limitaciones (sesgo selección, calibración, n=631, placeholder de costos) | DECISIONES; README guía | — | — | pendiente |
| 7.3 Implicancias para microfinanzas y trabajo futuro | — | — | chioda2025, golec2025 | pendiente |

## Figuras a crear (backlog)

1. Diagrama del diseño experimental (cap. 4.1).
2. Forest plot AUC por modelo × segmento (5.1/5.6 — la figura central).
3. Barras zero vs few-shot por perfil (5.4).
4. Brecha original-control de contaminación (6.2).
5. Caja con un prompt real y su thinking (4.4, anexo).
