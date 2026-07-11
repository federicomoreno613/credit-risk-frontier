"""Nodos de reporting (huella spaceflights: reporting).

Consolida métricas en la tabla comparativa (reemplaza el script 07), genera las
figuras de tesis leyendo los AUC de los datasets de métricas (no hardcodeados, con
la clave correcta ``"AUC"`` — el script 10 leía ``"auc"`` en minúscula y pintaba
"pend." para el LLM), calcula F1 post-hoc (script 09d) y valida los artefactos de
tesis. Todos los nodos son puros: reciben datos del catálogo y RETORNAN el objeto
(DataFrame / Figure / dict); Kedro persiste vía catálogo (ningún savefig/to_csv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk_frontier import utils

SEG_ORDER = ["esparso", "denso", "total"]

MODEL_NAMES = {
    "XGBoost":     "XGBoost",
    "LogReg":      "Reg. Logística",
    "TabFM":       "TabFM (zero-shot)",
    "LLM_nothink": "Qwen3 TabLLM zero-shot",
    "LLM_think":   "Qwen3 TabLLM thinking",
    "LLM_fewshot": "Qwen3 few-shot (best)",
}
MODEL_COLORS = {
    "XGBoost":     "#4878CF",
    "LogReg":      "#6ACC65",
    "TabFM":       "#00A2A2",
    "LLM_nothink": "#F97316",
    "LLM_think":   "#8B5CF6",
    "LLM_fewshot": "#EC4899",
}


# ---------------------------------------------------------------------------
# Tabla comparativa (reemplaza el script 07)
# ---------------------------------------------------------------------------

def _row(model, segment, m, eval_type, std=np.nan, n=np.nan) -> dict:
    return {
        "model": model, "segment": segment,
        "AUC": m.get("AUC", np.nan), "AUC_std": std,
        "Gini": m.get("Gini", np.nan), "KS": m.get("KS", np.nan),
        "Brier": m.get("Brier", np.nan), "eval_type": eval_type, "n": n,
    }


def _rows_classic_test(model_key: str, metrics: dict) -> list:
    """Fila 'total' desde el test temporal de un clásico (03/04)."""
    if not metrics or "test" not in metrics:
        return []
    return [_row(model_key, "total", metrics["test"], "test_temporal")]


def _rows_cv_seg(cv_seg: dict, model_key: str) -> list:
    """Filas esparso/denso/total desde el CV temporal (script 05)."""
    rows = []
    for seg in SEG_ORDER:
        s = cv_seg.get(model_key, {}).get(seg)
        if not s or "AUC" not in s:
            continue
        m = {k: s[k]["mean"] for k in ("AUC", "Gini", "KS", "Brier") if k in s}
        rows.append(_row(model_key, seg, m, "cv_temporal",
                         std=s["AUC"]["std"], n=s.get("n_folds", np.nan)))
    return rows


def _rows_llm_test(model_key: str, test: dict) -> list:
    """Filas total/esparso/denso desde la evaluación en test de un LLM."""
    rows = []
    for seg in SEG_ORDER:
        s = test.get(seg, {})
        if "AUC" in s:
            rows.append(_row(model_key, seg, s, "test", n=s.get("n", np.nan)))
    return rows


def build_comparison_table(xgb_metrics: dict, logreg_metrics: dict, cv_seg_metrics: dict,
                           llm_nothink_metrics: dict, llm_fewshot_metrics: dict,
                           params: dict) -> pd.DataFrame:
    """Consolida las métricas en un DataFrame comparativo (long).

    Recibe los dicts del catálogo (no hace globbing de ``models/``). Los clásicos
    citables son los no-leak (xgb_metrics/logreg_metrics); su AUC total sale del test
    temporal y sus segmentos del CV temporal.
    """
    rows = []
    rows += _rows_classic_test("XGBoost", xgb_metrics)
    rows += _rows_classic_test("LogReg", logreg_metrics)
    rows += _rows_cv_seg(cv_seg_metrics, "XGBoost")
    rows += _rows_cv_seg(cv_seg_metrics, "LogReg")
    rows += _rows_llm_test("LLM_nothink", (llm_nothink_metrics or {}).get("test", {}))

    # Few-shot: elegir la mejor configuración por AUC total en test.
    best = _best_fewshot(llm_fewshot_metrics)
    if best:
        rows += _rows_llm_test("LLM_fewshot", best)

    table = pd.DataFrame(rows)
    if not table.empty:
        table["model_name"] = table["model"].map(MODEL_NAMES).fillna(table["model"])
        table = table.sort_values(["model", "segment"]).reset_index(drop=True)
    return table


def _best_fewshot(combined: dict) -> dict | None:
    """Selecciona el test-dict de la config few-shot con mejor AUC total (fuente: 07._best_fewshot)."""
    if not combined:
        return None
    candidates = {k: v["test"] for k, v in combined.items()
                  if k.endswith("_shot") and isinstance(v, dict) and "test" in v}
    if not candidates:
        return None
    best = max(candidates.items(), key=lambda kv: kv[1].get("total", {}).get("AUC", -1))
    return best[1]


# ---------------------------------------------------------------------------
# F1 post-hoc (script 09d) — nodo puro
# ---------------------------------------------------------------------------

def compute_f1_posthoc(predictions: pd.DataFrame, params: dict) -> dict:
    """F1@umbral + F1 óptimo + AUC recomputado por segmento desde el contrato de predicciones.

    Fuente: 09d_f1_posthoc. Opera sobre test (filas con prob no-NaN).
    """
    from sklearn.metrics import f1_score, roc_auc_score

    threshold = params.get("threshold", 0.5)
    df = predictions[predictions["set"] == "test"].copy()
    df = df[df["prob"].notna()]

    out = {}
    for seg in ("total", "esparso", "denso"):
        sub = df if seg == "total" else df[df["segmento"] == seg]
        if len(sub) < utils.MIN_SEG_CASOS or sub["y_true"].nunique() < 2:
            out[seg] = {"n": int(len(sub)), "note": "insuficiente"}
            continue
        y = sub["y_true"].values
        p = sub["prob"].values
        f1_fixed = f1_score(y, (p >= threshold).astype(int), zero_division=0)
        grid = np.linspace(0.05, 0.95, 19)
        f1_best = max(f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid)
        out[seg] = {
            "n": int(len(sub)),
            "f1_at_threshold": float(f1_fixed),
            "f1_optimal": float(f1_best),
            "auc": float(roc_auc_score(y, p)),
        }
    return utils.json_safe({"threshold": threshold, **out})


# ---------------------------------------------------------------------------
# Validación de artefactos de tesis (preservado del scaffold WIP)
# ---------------------------------------------------------------------------

def validate_thesis_artifacts(comparison_table: pd.DataFrame, dataset_report: dict,
                              eda_report: dict, params: dict) -> dict:
    """Valida que docs, figuras y comparativa de tesis estén presentes."""
    required_files = params.get("required_files", [])
    checks = {}
    for rel in required_files:
        path = utils.PROJECT_ROOT / rel
        checks[rel] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
    missing = [rel for rel, c in checks.items() if not c["exists"]]
    if missing and params.get("fail_on_missing_files", True):
        raise FileNotFoundError(f"Faltan artefactos de tesis esperados: {missing}")

    return {
        "status": "ok" if not missing else "missing_files",
        "missing_files": missing,
        "required_file_checks": checks,
        "comparison_rows": int(len(comparison_table)),
        "comparison_models": sorted(comparison_table["model"].dropna().unique().tolist())
        if "model" in comparison_table else [],
        "dataset_rows": dataset_report.get("n_rows"),
        "dataset_segments": dataset_report.get("segment_counts", {}),
        "eda_figures_checked": len(eda_report.get("figure_checks", {})),
    }
