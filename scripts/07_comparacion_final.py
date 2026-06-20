# %% [markdown]
# # Comparación Final — XGBoost vs. LLM (zero-shot y few-shot)
# ## Tesis: LLMs vs. ML Clásico — UBA 2026
#
# Consolida las métricas de los modelos de la Entrega 3 en una tabla comparativa
# única (DataFrame), lista para la tesis. Scope E3: **3 modelos** —
# XGBoost, Qwen3-8B zero-shot (TabLLM) y Qwen3-8B few-shot. Se incluyen LogReg y el
# modo thinking si sus métricas están disponibles.
#
# ## Dos fuentes de evaluación (columna `eval_type`)
#
# Los modelos no comparten exactamente el mismo protocolo, y la tabla lo explicita:
#
# - **`test_temporal`** — métrica puntual sobre el conjunto de test (split por `set`,
#   honesto fuera de muestra). Es la métrica primaria de la tesis. Disponible para
#   todos los modelos en el segmento *total*, y para los LLM también por segmento.
# - **`cv_temporal`** — media ± std de `TimeSeriesSplit` (script 05). Aporta el
#   desglose por segmento (esparso/denso) de los modelos clásicos, con su variabilidad.
#
# No se mezcla con el viejo CV barajado (StratifiedKFold), que inflaba las métricas.
#
# **Requiere:** scripts 03, 04, 05, 06b ejecutados (08 opcional para few-shot).

# %% [markdown]
# ## Setup

# %%
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEG_ORDER = ["esparso", "denso", "total"]

MODEL_COLORS = {
    "XGBoost":     "#4878CF",
    "LogReg":      "#6ACC65",
    "LLM_nothink": "#F97316",
    "LLM_think":   "#8B5CF6",
    "LLM_fewshot": "#EC4899",
}
MODEL_NAMES = {
    "XGBoost":     "XGBoost",
    "LogReg":      "Reg. Logística",
    "LLM_nothink": "Qwen3 TabLLM zero-shot",
    "LLM_think":   "Qwen3 TabLLM thinking",
    "LLM_fewshot": "Qwen3 few-shot (best)",
}

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


# %% [markdown]
# ## Helpers de extracción
#
# Cada modelo guarda sus métricas en una forma distinta; estas funciones las
# normalizan a filas `(model, segment, AUC, AUC_std, Gini, KS, Brier, eval_type, n)`.

# %%
def _row(model, segment, m, eval_type, std=np.nan, n=np.nan):
    """Construye una fila normalizada de la tabla comparativa."""
    return {
        "model": model, "segment": segment,
        "AUC": m.get("AUC", np.nan), "AUC_std": std,
        "Gini": m.get("Gini", np.nan), "KS": m.get("KS", np.nan),
        "Brier": m.get("Brier", np.nan), "eval_type": eval_type,
        "n": n,
    }


def _rows_from_classic_test(model_key: str, path: Path) -> list:
    """Fila 'total' desde el test temporal de un modelo clásico (03/04)."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if "test" not in data:
        return []
    return [_row(model_key, "total", data["test"], "test_temporal")]


def _rows_from_cv_seg(cv_seg: dict, model_key: str) -> list:
    """Filas esparso/denso (y total) desde el CV temporal (script 05)."""
    rows = []
    segs = cv_seg.get(model_key, {})
    for seg in SEG_ORDER:
        s = segs.get(seg)
        if not s or "AUC" not in s:
            continue
        m = {k: s[k]["mean"] for k in ("AUC", "Gini", "KS", "Brier") if k in s}
        rows.append(_row(model_key, seg, m, "cv_temporal",
                         std=s["AUC"]["std"], n=s.get("n_folds", np.nan)))
    return rows


def _rows_from_llm_test(model_key: str, test: dict) -> list:
    """Filas total/esparso/denso desde la evaluación en test de un LLM (06b/08)."""
    rows = []
    for seg in SEG_ORDER:
        s = test.get(seg, {})
        if "AUC" in s:
            rows.append(_row(model_key, seg, s, "test", n=s.get("n", np.nan)))
    return rows


def _best_fewshot(metrics_dir: Path) -> tuple:
    """Selecciona la configuración few-shot con mejor AUC total en test.

    Retorna (n_shots, test_dict) o (None, None) si no hay métricas.
    """
    combined = metrics_dir / "llm_metrics_fewshot.json"
    candidates = {}
    if combined.exists():
        with open(combined) as f:
            data = json.load(f)
        for key, conf in data.items():
            if key.endswith("_shot") and isinstance(conf, dict) and "test" in conf:
                candidates[key] = conf["test"]
    else:
        for path in sorted(metrics_dir.glob("llm_metrics_fewshot_*.json")):
            with open(path) as f:
                conf = json.load(f)
            candidates[f"{conf.get('n_shots','?')}_shot"] = conf.get("test", {})
    if not candidates:
        return None, None
    best = max(candidates.items(),
               key=lambda kv: kv[1].get("total", {}).get("AUC", -1))
    return best[0], best[1]


# %% [markdown]
# ## Construcción de la tabla comparativa (nodo Kedro)

# %%
def compare_models(metrics_dir: str, output_dir: str) -> pd.DataFrame:
    """Consolida las métricas disponibles en un DataFrame comparativo.

    Args:
        metrics_dir: directorio con los JSON de métricas (03–08).
        output_dir: directorio para guardar `tabla_comparacion_final.csv` y figura.

    Returns:
        DataFrame long con una fila por (modelo, segmento, eval_type).
    """
    metrics_path = Path(metrics_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    figs_dir = output_path.parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    cv_seg_file = metrics_path / "cv_seg_metrics.json"
    cv_seg = json.load(open(cv_seg_file)) if cv_seg_file.exists() else {}

    rows = []
    # Clásicos: total desde test temporal + segmentos desde CV temporal
    rows += _rows_from_classic_test("XGBoost", metrics_path / "xgboost_metrics.json")
    rows += _rows_from_classic_test("LogReg", metrics_path / "logreg_metrics.json")
    rows += _rows_from_cv_seg(cv_seg, "XGBoost")
    rows += _rows_from_cv_seg(cv_seg, "LogReg")

    # LLM zero-shot (nothink / think) desde test
    for mode, key in [("nothink", "LLM_nothink"), ("think", "LLM_think")]:
        path = metrics_path / f"llm_metrics_tabllm_{mode}.json"
        if path.exists():
            with open(path) as f:
                rows += _rows_from_llm_test(key, json.load(f).get("test", {}))

    # LLM few-shot (mejor configuración) desde test
    n_shots, fewshot_test = _best_fewshot(metrics_path)
    if fewshot_test:
        rows += _rows_from_llm_test("LLM_fewshot", fewshot_test)

    table = pd.DataFrame(rows)
    if not table.empty:
        table["model_name"] = table["model"].map(MODEL_NAMES).fillna(table["model"])
        table = table.sort_values(["model", "segment"]).reset_index(drop=True)
    table.to_csv(output_path / "tabla_comparacion_final.csv", index=False)

    plot_comparison(table, figs_dir)
    return table


# %% [markdown]
# ## Figura — AUC por segmento (modelos disponibles)

# %%
def plot_comparison(table: pd.DataFrame, figs_dir: Path) -> None:
    """Barras de AUC por segmento para cada modelo disponible."""
    if table.empty:
        return
    # Una sola fila por (modelo, segmento): preferir test temporal > test > cv temporal
    pref = {"test_temporal": 0, "test": 1, "cv_temporal": 2}
    plot_tbl = (table.assign(_p=table["eval_type"].map(pref).fillna(9))
                .sort_values("_p")
                .drop_duplicates(["model", "segment"], keep="first"))

    models = [m for m in MODEL_NAMES if m in set(plot_tbl["model"])]
    x = np.arange(len(SEG_ORDER))
    n = max(len(models), 1)
    width = min(0.20, 0.9 / n)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)

    fig, ax = plt.subplots(figsize=(max(10, n * 2), 6))
    for offset, model_key in zip(offsets, models):
        sub = plot_tbl[plot_tbl["model"] == model_key].set_index("segment")
        aucs = [sub.loc[s, "AUC"] if s in sub.index else 0 for s in SEG_ORDER]
        stds = [sub.loc[s, "AUC_std"] if s in sub.index else 0 for s in SEG_ORDER]
        stds = [0 if (isinstance(v, float) and np.isnan(v)) else v for v in stds]
        bars = ax.bar(x + offset * width, aucs, width, color=MODEL_COLORS.get(model_key, "#888"),
                      alpha=0.85, label=MODEL_NAMES.get(model_key, model_key),
                      edgecolor="white", yerr=stds, capsize=4, error_kw={"linewidth": 1.2})
        for bar, val in zip(bars, aucs):
            if val and val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Bureau esparso", "Bureau denso", "Total"])
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.set_title("Comparación AUC-ROC por segmento — Tesis UBA 2026\n"
                 "(clásicos: segmentos por CV temporal; LLM: test)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs_dir / "22_comparacion_auc_segmentos.png", bbox_inches="tight")
    plt.close(fig)


# %% [markdown]
# ## Entry point

# %%
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    compare_models(metrics_dir=str(BASE / "models"), output_dir=str(BASE / "models"))
